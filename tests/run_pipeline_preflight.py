"""Check Robot inputs and the model APIs before processing a whole video."""

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.embedding import get_embedding
from utils.llm_gpt import MODEL, generate_text_response
from utils.mllm_gpt import generate_messages, get_response


class Status(BaseModel):
    status: Literal["ok"]


def main():
    questions = json.loads((ROOT / "data/robot.json").read_text())
    if not questions["bedroom_01"]["qa_list"]:
        raise ValueError("No Robot sample questions found")
    prompt = 'Return {"status": "ok"}.'
    if not get_embedding("CAM preflight"):
        raise RuntimeError("Empty embedding")
    generate_text_response(prompt, text_format=Status)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    get_response(generate_messages(image, prompt), text_format=Status)
    print(f"Robot annotations, embeddings and {MODEL} text/vision requests passed.")


if __name__ == "__main__":
    main()
