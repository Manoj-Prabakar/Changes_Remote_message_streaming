from typing import AsyncIterator, Callable, Optional

import asyncio
import logging
import httpx
from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    SendMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]

import inspect
from types import SimpleNamespace
from typing import AsyncIterator, Any


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents and support streaming calls."""

    def __init__(self, agent_card: AgentCard, agent_url: str):
        logger.info("agent_card: %s", getattr(agent_card, "name", str(agent_card)))
        logger.info("agent_url: %s", agent_url)
        # Use a single AsyncClient per RemoteAgentConnections
        self._httpx_client = httpx.AsyncClient(timeout=30)
        self.agent_client = A2AClient(self._httpx_client, agent_card, url=agent_url)
        self.card = agent_card
        self.conversation_name: Optional[str] = None
        self.conversation = None
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(
        self, message_request: SendMessageRequest
    ) -> SendMessageResponse:
        """
        Backwards-compatible one-shot message send. Returns a single SendMessageResponse.
        """
        return await self.agent_client.send_message(message_request)


# replace the previous send_streaming_message with this:
    async def send_streaming_message(
        self, message_request: SendStreamingMessageRequest
    ) -> AsyncIterator[Any]:
        """
        Start a streaming A2A call and return an async iterator that yields events.

        This function tries multiple possible method names / response shapes on
        the underlying A2AClient to be robust across client versions.
        """

        logger.debug("Attempting to send streaming message via A2AClient...")

        # Candidate method names to try (ordered by likelihood)
        candidate_names = [
            "send_streaming_message",
            "send_stream_message",
            "send_message_streaming",
            "send_message_stream",
            "streaming_send_message",
            "stream_message",
            "send_message",  # last resort: try the normal send_message with a streaming request
        ]

        # Keep diagnostic info if nothing works
        attempted = []
        last_exc = None

        for name in candidate_names:
            method = getattr(self.agent_client, name, None)
            if method is None:
                attempted.append(f"{name}=<missing>")
                continue

            attempted.append(name)
            try:
                # Call the method. It might be a coroutine or return an async iterator
                res = method(message_request)

                # If it's awaitable, await it
                if inspect.isawaitable(res):
                    res = await res

                # If the returned object is an async iterator / generator, return it
                if hasattr(res, "__aiter__"):
                    logger.debug("Using streaming method: %s (returns async iterator)", name)
                    return res  # caller can `async for` over this

                # If it has .events() async iterator, wrap that
                if hasattr(res, "events") and callable(getattr(res, "events")):
                    logger.debug("Using streaming method: %s (response.events())", name)

                    async def _iter_events():
                        async for ev in res.events():
                            yield ev

                    return _iter_events()

                # If it's a one-shot SendMessageResponse-like object (final), wrap into async iterator
                # so callers can still `async for` (they will see a single final event)
                # Heuristic: check for typical attributes (root/result or model_dump_json)
                if hasattr(res, "model_dump_json") or hasattr(res, "root"):
                    logger.debug("Method %s returned a non-iterator final response; wrapping into single-event iterator", name)

                    async def _single_final():
                        yield res

                    return _single_final()

                # If it's something else, try to treat it as synchronous iterable (rare)
                if hasattr(res, "__iter__"):
                    logger.debug("Method %s returned a synchronous iterable; wrapping into async iterator", name)

                    async def _wrap_sync_iter():
                        for ev in res:
                            yield ev

                    return _wrap_sync_iter()

                # Unknown shape: raise to try next candidate
                raise RuntimeError(f"Unsupported response shape from {name}: {type(res)}")

            except Exception as ex:
                logger.debug("Candidate method %s failed: %s", name, ex, exc_info=True)
                last_exc = ex
                # try next candidate

        # If we reached here, nothing worked
        client_methods = ", ".join([m for m in dir(self.agent_client) if not m.startswith("_")])
        raise RuntimeError(
            "A2AClient does not support a known streaming method. "
            f"Attempted: {attempted}. "
            f"Available client attributes (sample): {client_methods[:1000]!s}. "
            f"Last exception: {last_exc}"
        )

    async def close(self):
        """Close the underlying httpx client (call at shutdown)."""
        try:
            await self._httpx_client.aclose()
        except Exception as e:
            logger.debug("Error closing httpx client: %s", e)
