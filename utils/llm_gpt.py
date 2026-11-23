"""Shared OpenAI generation for text and sampled video frames."""

import os
from functools import lru_cache

from openai import OpenAI

from utils.token_usage import usage_from_response


MODEL = os.environ.get("CAM_MODEL", "gpt-5.2")


@lru_cache(maxsize=1)
def get_client(api_key=None, base_url=None):
    return OpenAI(api_key=api_key, base_url=base_url)


def get_response(messages, text_format=None, model=None, api_key=None,
                 base_url=None, temperature=None):
    model = model or MODEL
    completions = get_client(api_key, base_url).chat.completions
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature

    if text_format is None:
        response = completions.create(**kwargs)
        content = response.choices[0].message.content
    else:
        response = completions.parse(**kwargs, response_format=text_format)
        content = response.choices[0].message.parsed
    if content is None:
        raise ValueError(f"{model} returned an empty response")
    return content, usage_from_response(response, model, "openai")


def generate_text_response(prompt, text_format=None, model=None, api_key=None,
                           base_url=None, temperature=None):
    messages = [{"role": "user", "content": prompt}]
    return get_response(messages, text_format, model, api_key, base_url, temperature)
