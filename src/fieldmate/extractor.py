"""Constrained generation -> Pydantic validation -> one error-feedback retry -> graceful failure."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .client import GenerationMetrics, OllamaClient, OllamaError
from .prompts import extraction_messages, repair_message
from .schemas import SCHEMAS, json_schema_for

Kind = Literal["preflight", "defect"]


class Attempt(BaseModel):
    raw: str
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Always returned - never raises for model misbehaviour (graceful failure)."""

    ok: bool
    kind: Kind
    model: str
    data: dict[str, Any] | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    error: str | None = None

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def first_try_valid(self) -> bool:
        return bool(self.attempts) and self.attempts[0].error is None


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors()[:8]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)


def parse_and_validate(kind: str, raw: str) -> tuple[BaseModel | None, str | None]:
    """Return (model, None) on success or (None, human-readable error) on failure."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"- output is not valid JSON ({exc.msg} at char {exc.pos})"
    try:
        return SCHEMAS[kind].model_validate(obj), None
    except ValidationError as exc:
        return None, _format_validation_error(exc)


class Extractor:
    def __init__(self, client: OllamaClient, max_retries: int | None = None) -> None:
        self.client = client
        self.max_retries = client.settings.max_retries if max_retries is None else max_retries

    def extract(
        self,
        kind: Kind,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        constrained: bool = True,
        options: dict | None = None,
    ) -> ExtractionResult:
        """`constrained=False` sends format="json" (no schema) - used for the ablation study."""
        if kind not in SCHEMAS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(SCHEMAS)}")
        model_name = model or self.client.settings.model
        fmt: dict | str = json_schema_for(kind) if constrained else "json"
        messages = extraction_messages(kind, text)
        result = ExtractionResult(ok=False, kind=kind, model=model_name)

        for attempt_no in range(self.max_retries + 1):
            try:
                res = self.client.chat(
                    messages, model=model_name, fmt=fmt, temperature=temperature, options=options
                )
            except OllamaError as exc:
                result.error = f"ollama error: {exc}"
                return result  # infrastructure failure: fail fast, don't burn a retry
            parsed, err = parse_and_validate(kind, res.content)
            result.attempts.append(
                Attempt(raw=res.content, error=err, metrics=res.metrics.as_dict())
            )
            if parsed is not None:
                result.ok = True
                result.data = parsed.model_dump()
                result.error = None
                return result
            result.error = err
            if attempt_no < self.max_retries:
                messages = [
                    *messages,
                    {"role": "assistant", "content": res.content},
                    repair_message(err or "unknown error"),
                ]
        result.error = f"validation failed after {len(result.attempts)} attempt(s):\n{result.error}"
        return result


def total_metrics(result: ExtractionResult) -> GenerationMetrics:
    """Sum metrics over all attempts (retries cost real latency, so we count them)."""
    m = GenerationMetrics(model=result.model)
    for a in result.attempts:
        d = a.metrics
        m.prompt_tokens += d.get("prompt_tokens", 0)
        m.output_tokens += d.get("output_tokens", 0)
        m.load_s += d.get("load_s", 0.0)
        m.prompt_eval_s += d.get("prompt_eval_s", 0.0)
        m.eval_s += d.get("eval_s", 0.0)
        m.total_s += d.get("total_s", 0.0)
        m.wall_s += d.get("wall_s", 0.0)
    return m
