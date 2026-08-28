"""
Open-source LLM backends for the offline augmentation pipeline.

Two interchangeable text-generation backends are provided:

* ``HFTransformersLLM`` — runs any open-source instruct model locally through
    🤗 ``transformers`` (e.g. ``meta-llama/Llama-3.2-3B-Instruct``,
    ``Qwen/Qwen3-8B``, ``mistralai/Mistral-7B-Instruct-v0.3``).
* ``VLLMBackend`` — same models but served through ``vllm`` for high-throughput
  batched generation (recommended for 39k users).

Both expose a single method::

    generate(system_prompt: str, user_prompts: list[str]) -> list[str]

so the rest of the pipeline is backend-agnostic. A tiny ``EchoBackend`` is also
included for dry-runs / CI without a GPU.
"""

from __future__ import annotations

import re
from typing import Protocol

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _strip_thinking(text: str) -> str:
    """Remove any <think>...</think> block (and stray tags) from model output."""
    text = _THINK_RE.sub("", text)

    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    text = text.replace("</think>", "").replace("<think>", "")
    return text.strip()

def _apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Render a chat prompt, disabling thinking mode when the template supports it."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

class LLMBackend(Protocol):
    def generate(self, system_prompt: str, user_prompts: list[str]) -> list[str]:
        ...

class HFTransformersLLM:
    """Local generation via 🤗 transformers chat template."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 160,
        temperature: float = 0.3,
        batch_size: int = 8,
        dtype: str = "bfloat16",
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device or "auto",
        )
        self.model.eval()
        self._torch = torch

    def generate(self, system_prompt: str, user_prompts: list[str]) -> list[str]:
        torch = self._torch
        outputs: list[str] = []
        for start in range(0, len(user_prompts), self.batch_size):
            batch = user_prompts[start : start + self.batch_size]
            texts = [
                _apply_chat_template(self.tokenizer, system_prompt, up)
                for up in batch
            ]
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=4096,
            ).to(self.model.device)

            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-5),
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            gen = gen[:, enc["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            outputs.extend(_strip_thinking(d) for d in decoded)
        return outputs

class VLLMBackend:
    """High-throughput generation via vLLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 160,
        temperature: float = 0.3,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
    ):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

    def generate(self, system_prompt: str, user_prompts: list[str]) -> list[str]:
        prompts = [
            _apply_chat_template(self.tokenizer, system_prompt, up)
            for up in user_prompts
        ]
        results = self.llm.generate(prompts, self.sampling)

        results = sorted(results, key=lambda r: int(r.request_id))
        return [_strip_thinking(r.outputs[0].text) for r in results]

class EchoBackend:
    """Deterministic stub that echoes a trimmed version of the prompt.

    Useful to smoke-test the pipeline end-to-end (I/O, alignment, encoding)
    without loading a real model.

    The echoed text is trimmed at a word boundary and terminated with a period.
    Cutting mid-word produced output that the format validator correctly
    rejected as truncated, which aborted every dry run before it could exercise
    the rest of the pipeline -- the opposite of what this backend is for.

    The stub deliberately marks itself. Earlier it prefixed every line with
    "This shopper's inferred profile:" regardless of which builder called it, so
    a dry run of the ITEM stage produced 23033 descriptions all claiming to
    describe a shopper -- text that reads like a plausible profile and is easy
    to mistake for a real run when skimming the audit log. The prefix is now
    ``[ECHO]``, which is unmistakably not model output and is greppable.
    """

    PREFIX = "[ECHO]"

    def __init__(self, max_chars: int = 200, **_ignored):
        self.max_chars = max_chars

    def generate(self, system_prompt: str, user_prompts: list[str]) -> list[str]:
        out = []
        for up in user_prompts:
            snippet = up.strip().replace("\n", " ")[: self.max_chars]
            snippet = snippet.rsplit(" ", 1)[0] if " " in snippet else snippet
            snippet = snippet.rstrip(" ,;:-").rstrip(".")
            out.append(f"{self.PREFIX} dry-run stub text: {snippet}.")
        return out

def build_backend(name: str, **kwargs) -> LLMBackend:
    """Factory: ``name`` in {"hf", "vllm", "echo"}."""
    name = name.lower()
    if name == "hf":
        return HFTransformersLLM(**kwargs)
    if name == "vllm":
        return VLLMBackend(**kwargs)
    if name == "echo":
        return EchoBackend(**kwargs)
    raise ValueError(f"unknown backend '{name}' (use hf | vllm | echo)")
