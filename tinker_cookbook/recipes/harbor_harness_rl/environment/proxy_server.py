"""FastAPI sidecar that translates OpenAI chat-completion requests to Tinker sampling.

Run inside the proxy sandbox via ``python -m ...proxy_server``; configuration is
read from the environment (see ``TinkerProxy.from_env``).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pickle
import time
from typing import Any

import tinker
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.third_party.litellm.provider import (
    _prepare_messages_with_tools,
    _sampling_result_to_chat_completion_dict,
    _SamplingResult,
)
from tinker_cookbook.third_party.openai_compat import openai_messages_to_tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger("tinker_proxy")


class TinkerProxy:
    """Serves an OpenAI-compatible API backed by a single Tinker sampling client.

    The tokenizer, renderer, and sampling client are built once in ``__init__``
    and reused for every request. Records each (request, response) pair plus raw
    token IDs; the dump endpoint is gated by an optional control token.
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        base_model: str,
        *,
        max_input_tokens: int,
        max_tokens: int,
        control_token: str = "",
    ) -> None:
        self.base_model = base_model
        self.max_input_tokens = max_input_tokens
        self.max_tokens = max_tokens
        self.control_token = control_token
        self.captures: list[dict[str, Any]] = []
        # The most recently captured prompt. Retries arrive back-to-back, so a
        # call whose prompt matches the previous one is skipped as a duplicate.
        self._last_prompt: tuple[int, ...] | None = None

        self.sampling_client = sampling_client
        self.renderer = renderers.get_renderer(
            get_recommended_renderer_name(base_model), get_tokenizer(base_model)
        )
        logger.info("ready: base_model=%s", base_model)

        self.app = FastAPI()
        self.app.add_api_route("/healthz", self.healthz, methods=["GET"])
        self.app.add_api_route("/__captured__", self.captured, methods=["GET"])
        self.app.add_api_route("/v1/models", self.models, methods=["GET"])
        self.app.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"])
        self.app.add_api_route("/chat/completions", self.chat_completions, methods=["POST"])

    @classmethod
    def from_env(cls) -> TinkerProxy:
        # A live SamplingClient pickled from the trainer
        sampling_client = pickle.loads(
            base64.b64decode(os.environ["PROXY_TINKER_SAMPLING_CLIENT_B64"])
        )
        return cls(
            sampling_client=sampling_client,
            base_model=sampling_client.get_base_model(),
            max_input_tokens=int(os.environ["PROXY_MAX_INPUT_TOKENS"]),
            max_tokens=int(os.environ["PROXY_MAX_TOKENS"]),
            control_token=os.environ.get("PROXY_CONTROL_TOKEN", ""),
        )

    def _authorized(self, request: Request) -> bool:
        return (
            not self.control_token
            or request.headers.get("x-proxy-control-token") == self.control_token
        )

    def _output_tokens(self, payload: dict[str, Any], prompt_len: int) -> int:
        """Output cap so prompt + output fits the context window (self.max_tokens)."""
        fits = self.max_tokens - prompt_len
        requested = payload.get("max_tokens") or payload.get("max_completion_tokens")
        output = min(int(requested), fits) if requested else fits
        return max(1, output)

    def _context_window_error(self, n_tokens: int) -> JSONResponse:
        """OpenAI-style 400 that OpenHands and litellm both classify as a
        context-window error.

        The message matches OpenHands' controller substring list ("context
        length exceeded") AND litellm's OpenAI context-window pattern ("maximum
        context length" / code ``context_length_exceeded``), so the agent
        reliably emits a CondensationRequestAction regardless of whether litellm
        reclassifies the 400.
        """
        msg = (
            f"This model's maximum context length is {self.max_tokens} tokens. "
            f"However, your messages resulted in {n_tokens} tokens. "
            "Context length exceeded - please reduce the length of the messages."
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": msg,
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "context_length_exceeded",
                }
            },
        )

    async def healthz(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "base_model": self.base_model,
            "max_input_tokens": self.max_input_tokens,
            "max_tokens": self.max_tokens,
            "captured": len(self.captures),
        }

    async def models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": self.base_model, "object": "model"}]}

    async def captured(self, request: Request) -> JSONResponse:
        if not self._authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return JSONResponse({"count": len(self.captures), "records": self.captures})

    async def chat_completions(self, request: Request):
        payload = await request.json()
        auth = request.headers.get("authorization", "")
        inbound_key = auth[7:] if auth[:7].lower() == "bearer " else auth

        # Render the prompt once, then size the output to fit the context window.
        tinker_messages = openai_messages_to_tinker(payload.get("messages", []))
        if payload.get("tools"):
            tinker_messages = _prepare_messages_with_tools(
                self.renderer, tinker_messages, payload["tools"]
            )
        model_input = self.renderer.build_generation_prompt(tinker_messages)
        prompt_token_ids = model_input.to_ints()
        prompt_len = len(prompt_token_ids)

        # Pre-check: input alone exceeds the reserved input budget. Reject before
        # sampling so OpenHands compacts and retries; this turn is not captured.
        if prompt_len >= self.max_input_tokens:
            logger.info(
                "context_length_exceeded (pre): prompt=%d >= max_input=%d",
                prompt_len,
                self.max_input_tokens,
            )
            return self._context_window_error(prompt_len)

        sample = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=float(payload.get("temperature", 1.0)),
                max_tokens=self._output_tokens(payload, prompt_len),
                top_p=float(payload.get("top_p", 1.0)),
                top_k=int(payload.get("top_k", -1)),
                stop=payload.get("stop") or self.renderer.get_stop_sequences(),
            ),
        )
        seq = sample.sequences[0]

        # Post-check: generation hit the context ceiling (input + output filled
        # the window). Discard this truncated turn and signal overflow so the
        # agent compacts; the turn is not captured for training.
        if prompt_len + len(seq.tokens) >= self.max_tokens:
            logger.info(
                "context_length_exceeded (post): prompt=%d + completion=%d >= max_tokens=%d",
                prompt_len,
                len(seq.tokens),
                self.max_tokens,
            )
            return self._context_window_error(prompt_len + len(seq.tokens))

        parsed_message, termination = self.renderer.parse_response(seq.tokens)
        result = _SamplingResult(
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=seq.tokens,
            logprobs=seq.logprobs,
            parsed_message=parsed_message,
            termination=termination,
            model_name=payload.get("model", "tinker"),
        )
        completion = _sampling_result_to_chat_completion_dict(result)

        record = {
            "ts": time.time(),
            "model": payload.get("model"),
            "inbound_key": inbound_key,
            "request_body": json.dumps(payload),
            "prompt_token_ids": result.prompt_token_ids,
            "completion_token_ids": result.completion_token_ids,
            "logprobs": result.logprobs,
            "finish_reason": completion["choices"][0]["finish_reason"],
            "prompt_cache_hit_tokens": sample.prompt_cache_hit_tokens,
            "response": completion,
        }
        # Dedup consecutive retries
        key = tuple(result.prompt_token_ids)
        if key != self._last_prompt:
            self._last_prompt = key
            self.captures.append(record)
        logger.info(
            "chat.completions prompt=%d completion=%d key=%r",
            len(result.prompt_token_ids),
            len(result.completion_token_ids),
            inbound_key,
        )

        if not payload.get("stream"):
            return JSONResponse(completion)
        return self._stream(completion)

    @staticmethod
    def _stream(completion: dict[str, Any]) -> StreamingResponse:
        """Minimal SSE: emit the whole message as one chunk, then [DONE]."""
        choice = completion["choices"][0]

        async def event_stream():
            chunk = {
                "id": completion["id"],
                "object": "chat.completion.chunk",
                "created": completion["created"],
                "model": completion["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": choice["message"],
                        "finish_reason": choice["finish_reason"],
                    }
                ],
            }
            yield "data: " + json.dumps(chunk) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def run(self, port: int) -> None:
        uvicorn.run(self.app, host="0.0.0.0", port=port, workers=1, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[proxy] %(message)s")
    TinkerProxy.from_env().run(port=int(os.environ.get("PROXY_PORT", "8000")))
