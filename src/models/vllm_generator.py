"""Zero-shot generation with a local Hugging Face model via vLLM."""

from __future__ import annotations

from typing import Any

from beartype import beartype

from src.data.qa import QAExample
from src.preprocessing.prompts import render_user_message
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VllmGenerator:
    """Batched instruct generation. vLLM is imported only when the model is loaded."""

    def __init__(self, config: dict[str, Any]) -> None:
        from vllm import LLM, SamplingParams

        model_cfg = config["model"]
        gen_cfg = config["generation"]
        model_name = str(model_cfg["name"])
        logger.info("loading local model %s", model_name)
        self._llm = LLM(
            model=model_name,
            gpu_memory_utilization=float(model_cfg["gpu_memory_utilization"]),
            max_model_len=int(model_cfg["max_model_len"]),
            enforce_eager=True,
        )
        self._sampling = SamplingParams(
            temperature=float(gen_cfg["temperature"]),
            max_tokens=int(gen_cfg["max_tokens"]),
        )
        self._template = str(config["prompt"])
        self._tokenizer = self._llm.get_tokenizer()

    def _chat_prompt(self, question: str) -> str:
        message = render_user_message(self._template, question)
        rendered: str = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered

    @beartype
    def predict(self, examples: list[QAExample]) -> list[str]:
        """Generate one short answer per question."""
        prompts = [self._chat_prompt(item.question) for item in examples]
        logger.info("generating %d answers with vLLM", len(prompts))
        outputs = self._llm.generate(prompts, self._sampling)
        return [output.outputs[0].text.strip() for output in outputs]
