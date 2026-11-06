"""FastAPI wrapper: /health, /chat, /chat/stream (NDJSON), /extract."""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .client import GenerationMetrics, OllamaClient, OllamaError
from .extractor import ExtractionResult, Extractor
from .prompts import chat_messages


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    context: str | None = Field(default=None, max_length=20000, description="Optional fleet log")
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    answer: str
    model: str
    metrics: dict


class ExtractRequest(BaseModel):
    kind: Literal["preflight", "defect"]
    text: str = Field(min_length=1, max_length=8000)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


@lru_cache(maxsize=1)
def get_client() -> OllamaClient:
    return OllamaClient()


def create_app() -> FastAPI:
    app = FastAPI(
        title="FieldMate - offline drone-ops assistant",
        version=__version__,
        description="Small language models via Ollama. No internet required.",
    )

    @app.get("/health")
    def health(client: OllamaClient = Depends(get_client)):
        try:
            version = client.version()
            models = client.list_models()
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        default = client.settings.model
        return {
            "status": "ok",
            "ollama_version": version,
            "default_model": default,
            "default_model_available": default in models,
            "models": models,
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, client: OllamaClient = Depends(get_client)):
        try:
            res = client.chat(
                chat_messages(req.message, req.context),
                model=req.model,
                temperature=req.temperature,
            )
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ChatResponse(
            answer=res.content,
            model=req.model or client.settings.model,
            metrics=res.metrics.as_dict(),
        )

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest, client: OllamaClient = Depends(get_client)):
        """Newline-delimited JSON: {"delta": "..."} lines, then {"done": true, "metrics": {...}}."""

        def gen() -> Iterator[str]:
            try:
                for item in client.chat_stream(
                    chat_messages(req.message, req.context),
                    model=req.model,
                    temperature=req.temperature,
                ):
                    if isinstance(item, GenerationMetrics):
                        yield json.dumps({"done": True, "metrics": item.as_dict()}) + "\n"
                    else:
                        yield json.dumps({"delta": item}) + "\n"
            except OllamaError as exc:
                yield json.dumps({"done": True, "error": str(exc)}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/extract", response_model=ExtractionResult)
    def extract(req: ExtractRequest, client: OllamaClient = Depends(get_client)):
        result = Extractor(client).extract(
            req.kind, req.text, model=req.model, temperature=req.temperature
        )
        if result.error and result.error.startswith("ollama error"):
            raise HTTPException(status_code=502, detail=result.error)
        # Validation failure is a *graceful* 200 with ok=false so callers can fall back.
        return result

    return app


app = create_app()
