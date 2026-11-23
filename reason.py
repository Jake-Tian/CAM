"""Answer M3-Bench-robot questions with the CAM graph agent."""

import argparse
from contextlib import redirect_stdout
import json
import os
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

from langchain_core.messages import HumanMessage, AIMessage

from utils.llm_gpt import generate_text_response
from utils.prompts import (
    prompt_agent_verify_answer_referencing,
)
from reasoning.agent import build_agent, DEFAULT_BUDGET
from utils.reasoning_trace import build_tool_rounds
from utils.general import atomic_output, find_pkl_files


# ---------------------------------------------------------------------------
# New agent-based reasoning
# ---------------------------------------------------------------------------

def reason(
    graph,
    video_name,
    question,
    budget=DEFAULT_BUDGET,
):
    """Answer one question and retain the tool observations and token count."""
    print("================================================")
    print("Question: ", question)

    app = build_agent(
        graph,
        video_name,
        budget=budget,
    )

    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "clip_history": [],
        "tool_call_history": [],
        "budget": budget,
        "total_tokens": 0,
    }

    result_state = app.invoke(initial_state)

    # Pull the final free-text answer: the last AIMessage without tool calls.
    final_ans = ""
    for msg in reversed(result_state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            final_ans = msg.text
            break

    if not final_ans:
        raise ValueError("The agent returned no final answer")

    total_tokens = result_state.get("total_tokens", 0)
    token_summaries = {"total": int(total_tokens or 0)}

    # Save both the tool call and its observation for error analysis.
    rounds = build_tool_rounds(result_state["messages"])

    print("Final Answer:", final_ans)
    print(f"Total Tokens: {total_tokens}")

    return {
        "question": question,
        "rounds": rounds,
        "final_answer": final_ans,
        "token_summaries": token_summaries,
    }

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_answer(question, ground_truth_answer, predicted_answer):
    prompt = prompt_agent_verify_answer_referencing.format(
        question=question,
        ground_truth_answer=ground_truth_answer,
        agent_answer=predicted_answer
    )

    response, _ = generate_text_response(prompt)
    verdict = response.strip().upper()
    if verdict.startswith("YES"):
        return True
    if verdict.startswith("NO"):
        return False
    raise ValueError(f"Unexpected judge response: {response!r}")


def main():
    parser = argparse.ArgumentParser(description="Answer the Robot questions for a saved graph.")
    parser.add_argument("videos", nargs="*", metavar="VIDEO")
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reasoning"))
    args = parser.parse_args()
    questions = json.loads(Path("data/robot.json").read_text(encoding="utf-8"))
    videos = args.videos or find_pkl_files(args.graph_dir)
    if not videos:
        parser.error("No video graphs found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = Path("data/logs")
    logs.mkdir(parents=True, exist_ok=True)
    for video in videos:
        items = questions[video]["qa_list"]
        if not items:
            raise ValueError(f"No questions for {video}")
        with (args.graph_dir / f"{video}.pkl").open("rb") as handle:
            graph = pickle.load(handle)
        results = {}
        output = args.out_dir / f"{video}.json"
        partial = output.with_suffix(f".{os.getpid()}.partial.json")
        with (logs / f"{video}_reason.log").open("w") as log, redirect_stdout(log):
            for item in tqdm(items, desc=f"reason {video}", file=sys.stderr):
                result = reason(graph, video, item["question"])
                result["evaluate_correct"] = evaluate_answer(
                    item["question"], item["answer"], result["final_answer"]
                )
                results[item["question_id"]] = {
                    "question": item["question"], "ground_truth_answer": item["answer"],
                    "reasoning": result, "timestamp": item.get("timestamp"), "type": item.get("type"),
                }
                with atomic_output(partial, "w") as handle:
                    json.dump(results, handle, indent=2, ensure_ascii=False)
        partial.replace(output)
        print(f"{video}: wrote {len(results)} answers to {output}")


if __name__ == "__main__":
    main()
