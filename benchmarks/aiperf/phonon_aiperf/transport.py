"""Keep PCM bytes out of AIPerf's JSON-over-ZMQ record transport."""

from aiperf.common.enums import ConnectionReuseStrategy
from aiperf.common.hooks import on_init
from aiperf.common.models import BinaryResponse, TextResponse
from aiperf.transports.aiohttp_client import AioHttpClient
from aiperf.transports.aiohttp_transport import AioHttpTransport, ConnectionLeaseManager


class PhononHttpClient(AioHttpClient):
    async def _request(self, *args, **kwargs):
        record = await super()._request(*args, **kwargs)
        record.responses = [
            TextResponse(
                perf_ns=response.perf_ns,
                text=str(len(response.raw_bytes)),
                content_type="application/x-phonon-pcm-size",
            )
            if isinstance(response, BinaryResponse)
            and response.content_type
            and response.content_type.startswith("audio/pcm")
            else response
            for response in record.responses
        ]
        return record


class PhononHttpTransport(AioHttpTransport):
    @on_init
    async def _init_aiohttp_client(self) -> None:
        self.aiohttp_client = PhononHttpClient(
            timeout=self.model_endpoint.endpoint.timeout,
            tcp_kwargs=self.tcp_kwargs,
            collect_trace_chunks=self.model_endpoint.endpoint.collect_trace_chunks,
        )
        if (
            self.model_endpoint.endpoint.connection_reuse_strategy
            == ConnectionReuseStrategy.STICKY_USER_SESSIONS
        ):
            self.lease_manager = ConnectionLeaseManager(tcp_kwargs=self.tcp_kwargs)
