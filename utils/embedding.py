"""OpenAI embedding helpers."""

from functools import lru_cache

from utils.llm_gpt import get_client


def _create_embeddings(texts):
    response = get_client().embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


@lru_cache(maxsize=4096)
def get_embedding(text):
    return _create_embeddings([text])[0]


def get_multiple_embeddings(texts):
    return _create_embeddings(texts)
