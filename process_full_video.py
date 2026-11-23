import argparse
from contextlib import redirect_stdout
import json
import pickle
import sys
import time
from pathlib import Path
from tqdm import tqdm
from classes.hetero_graph import HeteroGraph
from classes.output_structure import EpisodicFormat, TripleExtraction
from utils.llm_gpt import generate_text_response
from utils.mllm_gpt import generate_messages, get_response
from utils.prompts import prompt_generate_episodic_memory, prompt_extract_triples
from utils.general import atomic_output, merge_character_appearances
from utils.abstraction_config import AbstractionConfig
from utils.token_usage import (
    add_stage_usage,
    build_token_summary,
    usage_total,
)


def extract_behavior_triples(behaviors):
    behavior_prompt = (
        prompt_extract_triples
        + "\n"
        + json.dumps(behaviors, ensure_ascii=False)
    )
    response, tokens = generate_text_response(
        behavior_prompt,
        text_format=TripleExtraction,
    )
    triples = [
        [triple.source, triple.content, triple.target]
        for triple in response.triples
    ]
    return triples, tokens


def process_full_video(video_name):

    frames_dir = Path(f"data/frames/{video_name}")
    output_graph_path = Path("data/graphs") / f"{video_name}.pkl"
    output_json_path = Path("data/memorization") / f"{video_name}.json"
    if not frames_dir.exists() or not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    image_folders = sorted(
        [folder for folder in frames_dir.iterdir() if folder.is_dir()],
        key=lambda folder: int(folder.name)
    )
    if not image_folders:
        raise ValueError(f"No frame clip folders found in {frames_dir}")

    # tqdm stays visible because logs redirect stdout, not stderr.
    pbar = tqdm(total=len(image_folders) + 3, desc=f"memo {video_name}", file=sys.stderr)
    
    previous_conversation = False
    appearance_dict = dict()    # character name → [appearance description, embedding]
    episodic_memory = dict()
    graph = HeteroGraph()
    stage_usage = {}
    
    try: 
        for folder in image_folders:
            print("--------------------------------")
            print("Processing folder: ", folder)
            clip_id = int(Path(folder).name)
            current_images = sorted(folder.glob("*.jpg"), key=lambda path: int(path.stem))

            # Exclude embeddings from the appearance prompt.
            appearance_prompt_dict = {name: value[0] for name, value in appearance_dict.items()}
            prompt = "Character appearance from previous videos: \n" + json.dumps(appearance_prompt_dict) + "\n" + prompt_generate_episodic_memory
            messages = generate_messages(current_images, prompt)
            response, tokens = get_response(messages, EpisodicFormat)
            add_stage_usage(stage_usage, "mllm", tokens)
            if usage_total(stage_usage["mllm"]) > 7_000_000:
                raise RuntimeError("Video exceeded the memorization token limit")
            behaviors = response.behaviors
            conversation = response.conversation
            characters_appearance = response.characters_appearance
            scene = response.scene

            if behaviors and behaviors[0].startswith("Equivalence:"):
                equivalence_parts = behaviors[0].split(":")[1].split(",")
                print(equivalence_parts)
                if len(equivalence_parts) >= 2:
                    behaviors = behaviors[1:]
                    old_name = equivalence_parts[0].strip()
                    new_name = equivalence_parts[1].strip()
                    graph.rename_character(old_name, new_name)
                    if old_name in appearance_dict:
                        appearance_dict[new_name] = appearance_dict[old_name]
                        del appearance_dict[old_name]
                else:
                    print(f"Warning: Malformed equivalence line '{behaviors[0]}', skipping rename")

            if previous_conversation and len(conversation) == 0 and graph.current_conversation_id is not None:
                print(f"Extracting summary for completed conversation {graph.current_conversation_id}...")
                result, tokens = graph.extract_conversation_summary(graph.current_conversation_id)
                add_stage_usage(stage_usage, "conversation", tokens)
                print(f"✓ Conversation summary extracted. Attributes: {len(result['character_attributes'])}, Relationships: {len(result['characters_relationships'])}")

            if len(conversation) > 0:
                graph.update_conversation(clip_id, conversation, previous_conversation=previous_conversation)
                previous_conversation = True
            else:
                previous_conversation = False

            #--------------------------------
            # Graph Construction
            #--------------------------------
            if behaviors:
                triples, tokens = extract_behavior_triples(behaviors)
                add_stage_usage(stage_usage, "triples", tokens)
            else:
                triples = []
            
            graph.insert_triples(triples, clip_id, scene)
            print(f"Inserted {len(triples)} triples into graph for clip {clip_id}")

            equivalence_list = merge_character_appearances(characters_appearance, appearance_dict)
            if equivalence_list:
                for equivalence in equivalence_list:
                    graph.rename_character(equivalence[0], equivalence[1])

            episodic_memory[clip_id] = {
                "folder": str(folder),
                "characters_behavior": behaviors,
                "conversation": conversation,
                "characters_appearance": str(characters_appearance),
                "scene": scene,
                "triples": triples
            }
            pbar.update(1)

        if previous_conversation and graph.current_conversation_id is not None:
            print(f"Extracting summary for final conversation {graph.current_conversation_id}...")
            result, tokens = graph.extract_conversation_summary(graph.current_conversation_id)
            add_stage_usage(stage_usage, "conversation", tokens)
            print(f"✓ Final conversation summary extracted. Attributes: {len(result['character_attributes'])}, Relationships: {len(result['characters_relationships'])}")

        print("Inserting character appearances...")
        print("Number of edges: ", len(graph.edges))
        appearance_text_dict = {name: value[0] for name, value in appearance_dict.items()}
        graph.insert_character_appearances(appearance_text_dict)

        graph.node_embedding_insertion()
        pbar.set_postfix_str("appearances")
        pbar.update(1)

        abs_tokens = graph.run_abstraction(AbstractionConfig())
        for stage in ("attributes", "relationships"):
            add_stage_usage(stage_usage, stage, abs_tokens[f"{stage}_usage"])
        graph.insert_high_level_and_appearance_embeddings()
        pbar.set_postfix_str("abstraction")
        pbar.update(1)

        pbar.set_postfix_str("save")
        pbar.update(1)
    finally:
        pbar.close()

    token_summaries = build_token_summary(stage_usage)
    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(output_graph_path, "wb") as graph_file, atomic_output(output_json_path, "w") as memory_file:
        pickle.dump(graph, graph_file)
        json.dump({"memory_token_summaries": token_summaries, "episodic_memory": episodic_memory},
                  memory_file, indent=2)
    print(f"\n✓ Saved episodic memory and token summaries for {video_name} to {output_json_path}")
    return graph, episodic_memory, token_summaries


def main():
    parser = argparse.ArgumentParser(description="Build graph memory from prepared Robot frames.")
    parser.add_argument("videos", nargs="+", metavar="VIDEO")
    logs = Path("data/logs")
    logs.mkdir(parents=True, exist_ok=True)
    for video in parser.parse_args().videos:
        start = time.monotonic()
        with (logs / f"{video}_memo.log").open("w") as log, redirect_stdout(log):
            graph, memory, usage = process_full_video(video)
        print(f"{video}: {len(memory)} clips, {len(graph.edges)} edges, "
              f"{usage['total']} tokens, {time.monotonic() - start:.0f}s")


if __name__ == "__main__":
    main()
