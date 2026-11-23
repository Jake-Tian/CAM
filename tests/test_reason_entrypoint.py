import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage

import reason
from reasoning.agent import DEFAULT_BUDGET, build_agent


class _FakeAgent:
    def invoke(self, state):
        return {
            "messages": [*state["messages"], AIMessage(content="answer")],
            "total_tokens": 7,
        }


class ReasonEntrypointTest(unittest.TestCase):
    def test_default_reason_uses_five_tool_agent(self):
        graph = object()
        with patch.object(reason, "build_agent", return_value=_FakeAgent()) as build:
            result = reason.reason(graph, "video", "question")

        build.assert_called_once_with(graph, "video", budget=DEFAULT_BUDGET)
        self.assertEqual(result["final_answer"], "answer")
        self.assertEqual(result["rounds"], [])
        self.assertEqual(result["token_summaries"], {"total": 7})

    def test_missing_graph_returns_a_failure(self):
        with TemporaryDirectory() as graph_dir, patch.object(
            sys, "argv", ["reason.py", "bedroom_01", "--graph-dir", graph_dir, "--out-dir", graph_dir]
        ):
            with self.assertRaises(FileNotFoundError):
                reason.main()


if __name__ == "__main__":
    unittest.main()
