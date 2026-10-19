"""LLM backend interface, local Ollama HTTP implementation, and test mock."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import requests

from src.config import Config, load_config

logger = logging.getLogger(__name__)


def approximate_token_count(text: str) -> int:
    """Return a simple whitespace token estimate when a backend gives no count."""
    return len(text.split())


@dataclass(frozen=True)
class BackendResponse:
    """Raw backend response and per-call telemetry."""

    text: str
    model: str
    prompt_tokens: int
    output_tokens: int
    latency_ms: float


class BackendCallError(RuntimeError):
    """Backend failure carrying the telemetry available for the failed call."""

    def __init__(
        self,
        message: str,
        *,
        model: str,
        prompt_tokens: int,
        output_tokens: int = 0,
        latency_ms: float,
    ) -> None:
        """Initialize a backend error and its call measurements."""
        super().__init__(message)
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms


class LLMBackend(ABC):
    """Abstract interface implemented by local and mock text-generation APIs."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model identifier."""

    @abstractmethod
    def generate(self, prompt: str) -> BackendResponse:
        """Generate raw text and measured token/latency telemetry."""


class OllamaBackend(LLMBackend):
    """Call a local Ollama `/api/generate` endpoint over HTTP."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize Ollama settings from the project config.

        Args:
            config: explicit config, or the default project config.
        """
        self.config = config or load_config()
        self._model_name = self.config.llm.model
        self.base_url = self.config.llm.base_url.rstrip("/")
        self.temperature = self.config.llm.temperature
        self.max_tokens = self.config.llm.max_tokens
        self.timeout_seconds = self.config.llm.timeout_seconds

    @property
    def model_name(self) -> str:
        """Return the Ollama model tag configured for this backend."""
        return self._model_name

    def generate(self, prompt: str) -> BackendResponse:
        """Generate one non-streaming JSON-mode response via Ollama.

        Args:
            prompt: prompt text to submit.

        Returns:
            Raw response and Ollama token counts plus measured latency.

        Raises:
            BackendCallError: when the HTTP request times out or fails.
        """
        started = time.perf_counter()
        estimate = approximate_token_count(prompt)
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            latency_ms = (time.perf_counter() - started) * 1000.0
            result = BackendResponse(
                text=str(payload.get("response", "")),
                model=str(payload.get("model", self.model_name)),
                prompt_tokens=int(payload.get("prompt_eval_count", estimate)),
                output_tokens=int(
                    payload.get("eval_count", approximate_token_count(str(payload.get("response", ""))))
                ),
                latency_ms=latency_ms,
            )
            self._log_call(result.prompt_tokens, result.output_tokens, latency_ms)
            return result
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as error:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._log_call(estimate, 0, latency_ms)
            raise BackendCallError(
                str(error),
                model=self.model_name,
                prompt_tokens=estimate,
                latency_ms=latency_ms,
            ) from error

    @staticmethod
    def _log_call(prompt_tokens: int, output_tokens: int, latency_ms: float) -> None:
        """Log the required token and latency measurements for one call."""
        logger.info(
            "LLM call telemetry prompt_tokens=%d output_tokens=%d latency_ms=%.3f",
            prompt_tokens,
            output_tokens,
            latency_ms,
        )


class MockBackend(LLMBackend):
    """Deterministic test backend returning queued strings or raising errors."""

    def __init__(
        self,
        responses: Sequence[str | BaseException],
        model_name: str = "mock-model",
    ) -> None:
        """Create a queued-response mock.

        Args:
            responses: strings or exceptions consumed one per call.
            model_name: model identifier exposed to cache keys and reports.
        """
        self.responses = list(responses)
        self._model_name = model_name
        self.prompts: list[str] = []
        self.calls = 0

    @property
    def model_name(self) -> str:
        """Return the test model identifier."""
        return self._model_name

    def generate(self, prompt: str) -> BackendResponse:
        """Return the next queued response or raise its queued exception."""
        started = time.perf_counter()
        self.calls += 1
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("MockBackend has no queued response")
        result = self.responses.pop(0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        prompt_tokens = approximate_token_count(prompt)
        if isinstance(result, BaseException):
            logger.info(
                "Mock LLM call telemetry prompt_tokens=%d output_tokens=0 latency_ms=%.3f",
                prompt_tokens,
                elapsed_ms,
            )
            if isinstance(result, TimeoutError):
                raise BackendCallError(
                    str(result),
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    latency_ms=elapsed_ms,
                ) from result
            raise result
        text = str(result)
        output_tokens = approximate_token_count(text)
        logger.info(
            "Mock LLM call telemetry prompt_tokens=%d output_tokens=%d latency_ms=%.3f",
            prompt_tokens,
            output_tokens,
            elapsed_ms,
        )
        return BackendResponse(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )
