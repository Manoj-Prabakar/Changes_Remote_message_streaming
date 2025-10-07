from . import patch_genai_live_ws
from . import patch_adk_tracing 

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterable, List, Optional
import logging

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendStreamingMessageRequest,
    SendMessageResponse,
    SendStreamingMessageSuccessResponse,
    SendStreamingMessageResponse,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types


from .remote_agent_connection import RemoteAgentConnections

load_dotenv()
nest_asyncio.apply()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HostAgent:
    """The Host agent responsible for small talk and delegating tasks to remote agents."""

    def __init__(self) -> None:
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "test_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No friends found"

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: List[str],
    ) -> "HostAgent":
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    # ---------- ADK Agent wiring ----------

    def create_agent(self) -> Agent:
        return Agent(
            model="gemini-2.0-flash-exp", 
            name="test_agent",
            instruction=self.root_instruction,
            description="Streaming Host Agent for small talk and planning info",

            tools=[self.send_message_streaming, self.send_message],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        return f"""
**Role:** You are the Host Agent. You do small talk with the user and then tell
the information about the places using the remote agent - weather_agent.

**Core Directives:**
* Use the `send_message_streaming` tool to check the places details.
* If streaming yields nothing, a non-streaming `send_message` is also available.
* Main work will be done by the weather_agent.

<Available Agents>
{self.agents}
</Available Agents>
"""

    # ---------- One-shot A2A (fallback & direct tool) ----------

    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext) -> str:
        """Sends a task to a remote friend agent using one-shot A2A."""
        import traceback
        logger = logging.getLogger(__name__)

        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        state = tool_context.state or {}
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        print(f"\n🟢 Sending task to agent (one-shot): {agent_name}")
        print(json.dumps({"task": task, "task_id": task_id, "context_id": context_id}, indent=2))

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        # print request JSON safely
        try:
            print("\n📤 message_request:")
            print(json.dumps(json.loads(message_request.model_dump_json(exclude_none=True)), indent=2))
        except Exception:
            print("Could not pretty-print message_request; raw:", message_request)

        try:
            send_response: SendMessageResponse = await asyncio.wait_for(
                client.send_message(message_request), timeout=30
            )
        except asyncio.TimeoutError:
            logger.exception("Timeout sending message to %s", agent_name)
            return "(no response)"
        except Exception as e:
            logger.exception("Exception sending message to %s: %s", agent_name, e)
            text_err = str(e).lower()
            if "extra_headers" in text_err:
                logger.error("Detected 'extra_headers' incompatibility. See upgrade/monkeypatch suggestions.")
            if "not found" in text_err or "not supported for generatecontent" in text_err:
                logger.error("Model not found or not supported for generateContent. Check ListModels or use bidi/live.")
            return "(no response)"

        # Pretty-print response if possible
        try:
            resp_json = json.loads(send_response.model_dump_json(exclude_none=True))
            print("\n📥 send_response:")
            print(json.dumps(resp_json, indent=2))
        except Exception:
            print("\n📥 send_response (raw):", send_response)

        # Validate success shape and extract artifacts safely
        root = getattr(send_response, "root", None)
        if not isinstance(root, SendMessageSuccessResponse) or not isinstance(root.result, Task):
            print("⚠️ Received non-success or non-task response. Cannot proceed.")
            return "(no response)"

        try:
            response_content = root.model_dump_json(exclude_none=True)
            json_content = json.loads(response_content)
        except Exception:
            logger.exception("Failed to parse send_response JSON")
            return "(no response)"

        # Extract text parts from artifacts
        texts: List[str] = []
        result_obj = json_content.get("result", {})
        artifacts = result_obj.get("artifacts", []) if isinstance(result_obj, dict) else []
        for artifact in artifacts:
            if not artifact:
                continue
            parts = artifact.get("parts", []) or []
            for p in parts:
                if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                    texts.append(p["text"])

        if not texts:
            print("\n✅ Parsed Artifacts: [] (no text parts)")
            return "(no response)"

        print("\n✅ Parsed Artifacts (first text):")
        print(json.dumps(texts[0], indent=2))
        return "\n".join(texts)

    # ---------- Streaming A2A tool (with fallback) ----------

    async def send_message_streaming(
        self, agent_name: str, task: str, tool_context: ToolContext
    ) -> str:
        """Send task to remote agent via streaming A2A. Falls back to one-shot if nothing arrives."""
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")

        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        state = tool_context.state or {}
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        logger.info(f"🟢 Streaming message to {agent_name}: {task}")

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendStreamingMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )

        stream = None
        responses: List[str] = []

        try:
            stream = await client.send_streaming_message(message_request)

            async for event in stream:
                if isinstance(event, TaskStatusUpdateEvent):
                    logger.debug(f"📡 Status Update: {event.status.state}")
                elif isinstance(event, TaskArtifactUpdateEvent):
                    for artifact in getattr(event, "artifacts", []) or []:
                        parts = artifact.get("parts", []) or []
                        for p in parts:
                            if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                                logger.info(f"💬 {p['text']}")
                                responses.append(p["text"])
                            elif "file_data" in p or "inline_data" in p:
                                logger.info("🎧 Received audio/data part.")
                else:
                    logger.debug(f"Other event: {event}")

        except Exception as e:
            logger.exception("Error during streaming A2A message: %s", e)
        finally:
            if stream is not None:
                try:
                    await stream.aclose()
                except Exception:
                    pass

        if responses:
            final_text = "\n".join(responses)
            logger.info(f"Final streamed response: {final_text}")
            return final_text

        # Fallback
        logger.warning("Falling back to send_message()")
        try:
            return await self.send_message(agent_name=agent_name, task=task, tool_context=tool_context)
        except Exception as e:
            logger.exception("Fallback send_message() failed: %s", e)
            return "(no response)"

    

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """Streams Host Agent’s response to a given query."""
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )

        agen = None
        try:
            agen = self._runner.run_async(
                user_id=self._user_id,
                session_id=session.id,
                new_message=content,
                stream=True,
            )
            async for event in agen:
                if not event.is_final_response():
                    text_parts = [
                        p.text for p in event.content.parts
                        if getattr(p, "text", None)
                    ]
                    if text_parts:
                        yield {"is_task_complete": False, "updates": "".join(text_parts)}
                    else:
                        yield {"is_task_complete": False, "updates": "Thinking..."}
                    continue

                # final
                response = ""
                if event.content and event.content.parts:
                    response = "\n".join(
                        [p.text for p in event.content.parts if getattr(p, "text", None)]
                    )
                yield {"is_task_complete": True, "content": response}
                break
        except Exception as e:
            logger.exception("Error in HostAgent.stream: %s", e)
            yield {"is_task_complete": True, "content": f"Agent error: {e}"}
        finally:
            if agen is not None:
                aclose = getattr(agen, "aclose", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except Exception:
                        pass


# ---------- Synchronous initializer to wire root_agent ----------

def _get_initialized_host_agent_sync() -> Agent:
    """Synchronously creates and initializes the HostAgent and returns ADK Agent instance."""

    async def _async_main() -> Agent:
        agent_urls = [
            "http://localhost:10002"  # Weather Agent URL
        ]

        print("initializing host agent")
        hosting_agent_instance = await HostAgent.create(
            remote_agent_addresses=agent_urls
        )
        print("HostAgent initialized")
        return hosting_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize HostAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing HostAgent within an async function in your application."
            )
            # As a last resort, create a bare agent (no remote cards yet)
            ha = HostAgent()
            return ha.create_agent()
        else:
            raise


root_agent = _get_initialized_host_agent_sync()
