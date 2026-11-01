"""Parse Phonon's complete PCM response for AIPerf request-level metrics."""

from typing import Any

from aiperf.common.models import (
    BaseResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
    TextResponse,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


class PhononTTSEndpoint(BaseEndpoint):
    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        if not request_info.turns:
            raise ValueError("Phonon TTS requires a turn")
        turn = request_info.turns[-1]
        if not turn.texts or not turn.texts[0].contents:
            raise ValueError("Phonon TTS requires input text")
        payload = {
            "model": turn.model or request_info.model_endpoint.primary_model_name,
            "input": turn.texts[0].contents[0],
            "voice": "default",
            "response_format": "pcm",
        }
        if turn.extra_body:
            payload.update(turn.extra_body)
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        if not isinstance(response, TextResponse):
            return None
        if response.content_type != "application/x-phonon-pcm-size":
            return None
        byte_count = int(response.text)
        if byte_count == 0 or byte_count % 2:
            return None
        return ParsedResponse(perf_ns=response.perf_ns, data=BaseResponseData())
