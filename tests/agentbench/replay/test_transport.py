# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from agentinfer.agentbench.replay.config import ReplayBenchConfig
from agentinfer.agentbench.replay.prompt import SyntheticPrompt
from agentinfer.agentbench.replay.transport import ReplayTransport
from agentinfer.agentbench.request_proxy.request_trace import RequestTraceWriter, load_request_facts


def test_replay_transport_disables_keepalive_reuse(tmp_path: Path) -> None:
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1},
            "backend": {"base_url": "http://backend", "endpoint": "/v1/messages"},
            "replay": {"trace_path": "source.jsonl"},
        }
    )

    with patch("agentinfer.agentbench.replay.transport.httpx.AsyncClient", wraps=httpx.AsyncClient) as client:
        transport = ReplayTransport(config, "run", RequestTraceWriter(tmp_path / "requests.jsonl"))

    limits = client.call_args.kwargs["limits"]
    assert limits.max_connections is None
    assert limits.max_keepalive_connections == 0
    asyncio.run(transport.close())


def test_chat_transport_streams_content_usage_and_plan_identity(tmp_path: Path) -> None:
    async def send() -> tuple[object, dict[str, object]]:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {
                    "base_url": "http://backend",
                    "endpoint": "/v1/chat/completions",
                    "prompt_tokens_details_enabled": True,
                    "chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True},
                },
                "replay": {
                    "trace_path": "source.jsonl",
                    "workflow_profile_id": "coding_assistant",
                    "profile_version": "pythia-min-adaptive-v1",
                    "profile_role_source": "request_purpose",
                },
            }
        )
        trace = tmp_path / "requests.jsonl"
        writer = RequestTraceWriter(trace)
        await writer.start()
        transport = ReplayTransport(config, "run", writer)
        await transport.client.aclose()
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            stream = (
                'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

        transport.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        task = SimpleNamespace(runtime_session_id="session")
        node = SimpleNamespace(
            runtime_request_id="request",
            actor_id="lead",
            actor_role="lead",
            prompt_kind="lead_main",
            source_request_purpose="engineer",
            parent_actor_id=None,
            planned_output_tokens=3,
            planned_input_tokens=12,
            backend_sampling_seed=7,
        )
        result = await transport.send(
            task,
            node,
            SyntheticPrompt("system", (), ({"role": "user", "content": "message"},)),
        )
        await transport.close()
        await writer.close()
        return result, captured

    result, body = asyncio.run(send())
    facts = load_request_facts(tmp_path / "requests.jsonl")

    assert result.assistant_content == "hello"
    assert body["min_tokens"] == body["max_tokens"] == 3
    assert body["seed"] == 7
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "clear_thinking": True}
    assert "tools" not in body
    identity = json.loads(body["vllm_xargs"]["agentic_context"])
    assert identity["spawn_reason"] == "engineer"
    assert identity["workflow_profile_id"] == "coding_assistant"
    assert identity["profile_version"] == "pythia-min-adaptive-v1"
    assert facts[0].request_id == "request"
    assert facts[0].request_purpose == "engineer"
    assert facts[0].input_tokens == 12
    assert facts[0].output_tokens == 3
    assert facts[0].cached_tokens == 0
    assert facts[0].schema_version == "2"
    assert facts[0].program_id == "session:lead"
    assert facts[0].agent_id == "lead"
    assert facts[0].expected_resume is True
    assert facts[0].spawn_reason == "engineer"
    assert facts[0].workflow_profile_id == "coding_assistant"


def test_trace_can_mark_final_actor_request_terminal(tmp_path: Path) -> None:
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1},
            "backend": {
                "base_url": "http://backend",
                "endpoint": "/v1/chat/completions",
            },
            "replay": {
                "trace_path": "source.jsonl",
                "infer_expected_resume_from_trace": True,
            },
        }
    )
    transport = ReplayTransport(
        config,
        "run",
        RequestTraceWriter(tmp_path / "requests.jsonl"),
    )

    def node(request_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            runtime_request_id=request_id,
            actor_id="lead",
            actor_role="lead",
            prompt_kind="lead_main",
            source_request_purpose="planner",
            parent_actor_id=None,
            planned_output_tokens=3,
            planned_input_tokens=12,
            backend_sampling_seed=7,
            node_type="request",
        )

    first = node("first")
    final = node("final")
    task = SimpleNamespace(
        runtime_session_id="session",
        requests=(first, final),
    )
    prompt = SyntheticPrompt(
        "system",
        (),
        ({"role": "user", "content": "message"},),
    )

    first_body = transport._body(task, first, prompt)
    final_body = transport._body(task, final, prompt)
    first_identity = json.loads(first_body["vllm_xargs"]["agentic_context"])
    final_identity = json.loads(final_body["vllm_xargs"]["agentic_context"])
    asyncio.run(transport.close())

    assert first_identity["expected_resume"] is True
    assert final_identity["expected_resume"] is False


def test_explicit_trace_identity_overrides_replay_inference(tmp_path: Path) -> None:
    config = ReplayBenchConfig.model_validate(
        {
            "experiment": {"task_num": 1},
            "backend": {
                "base_url": "http://backend",
                "endpoint": "/v1/chat/completions",
            },
            "replay": {
                "trace_path": "source.jsonl",
                "workflow_profile_id": "configured-profile",
                "profile_version": "configured-version",
                "profile_role_source": "request_purpose",
                "infer_expected_resume_from_trace": True,
            },
        }
    )
    transport = ReplayTransport(
        config,
        "run",
        RequestTraceWriter(tmp_path / "requests.jsonl"),
    )
    node = SimpleNamespace(
        runtime_request_id="request",
        actor_id="legacy-actor",
        actor_role="subagent",
        prompt_kind="continuation",
        source_request_purpose="legacy-purpose",
        parent_actor_id="legacy-parent",
        planned_output_tokens=3,
        planned_input_tokens=12,
        backend_sampling_seed=7,
        node_type="request",
        source_program_id="source-task:explorer_1",
        source_agent_id="explorer_1",
        source_parent_program_id="source-task:planner",
        source_blocks_parent=True,
        source_expected_resume=False,
        source_agent_role="subagent",
        source_spawn_reason="explorer",
        source_workflow_profile_id="source-profile",
        source_profile_version="source-version",
    )
    task = SimpleNamespace(runtime_session_id="runtime-session", requests=(node,))
    body = transport._body(
        task,
        node,
        SyntheticPrompt("system", (), ({"role": "user", "content": "message"},)),
    )
    identity = json.loads(body["vllm_xargs"]["agentic_context"])
    asyncio.run(transport.close())

    assert identity == {
        "program_id": "runtime-session:source-task:explorer_1",
        "task_id": "runtime-session",
        "session_id": "runtime-session",
        "agent_id": "explorer_1",
        "parent_program_id": "runtime-session:source-task:planner",
        "blocks_parent": True,
        "expected_resume": False,
        "agent_role": "subagent",
        "spawn_reason": "explorer",
        "workflow_profile_id": "source-profile",
        "profile_version": "source-version",
        "request_id": "request",
    }


def test_messages_transport_keeps_title_prompt_tool_free_and_bridges_sampling(tmp_path: Path) -> None:
    async def build() -> dict[str, object]:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {"base_url": "http://backend", "endpoint": "/v1/messages"},
                "replay": {"trace_path": "source.jsonl"},
            }
        )
        transport = ReplayTransport(config, "run", RequestTraceWriter(tmp_path / "requests.jsonl"))
        task = SimpleNamespace(runtime_session_id="session")
        node = SimpleNamespace(
            runtime_request_id="title",
            actor_id="lead",
            actor_role="lead",
            prompt_kind="lead_title",
            parent_actor_id=None,
            planned_output_tokens=4,
            backend_sampling_seed=9,
        )
        body = transport._body(
            task,
            node,
            SyntheticPrompt("title system", (), ({"role": "user", "content": "title input"},)),
        )
        await transport.close()
        return body

    body = asyncio.run(build())

    assert body["tools"] == []
    sampling = body["metadata"]["_agentinfer_replay_sampling"]
    assert sampling == {"seed": 9, "min_tokens": 4, "ignore_eos": True}


@pytest.mark.parametrize("actual,success", [(5, True), (6, False), (4, False), (None, False)])
def test_inferact_synthetic_checks_wire_usage_and_preserves_both_stream_fields(tmp_path, actual, success):
    """Even HTTP 200 must fail strict calibration on missing or different backend usage."""

    async def send():
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "replay": {
                    "trace_path": "source.json",
                    "trace_type": "inferact_codex_swebenchpro",
                    "interval_mode": "lognormal",
                    "interval_lognormal": {"p50_seconds": 1, "p95_seconds": 2, "p99_seconds": 3},
                    "prompt_calibration_tolerance_tokens": 0,
                },
            }
        )
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        assert config.replay.prompt_calibration_tolerance_tokens == 0
        await writer.start()
        transport = ReplayTransport(config, "run", writer)
        await transport.client.aclose()

        def handler(request):
            data = {"choices": [{"delta": {"content": "answer", "reasoning_content": "thinking"}}]}
            if actual is not None:
                data["usage"] = {"prompt_tokens": actual, "completion_tokens": 2}
            return httpx.Response(200, text=f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n")

        transport.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        node = SimpleNamespace(
            runtime_request_id="request",
            actor_id="lead",
            actor_role="lead",
            parent_actor_id=None,
            prompt_kind="lead_main",
            planned_input_tokens=5,
            planned_output_tokens=2,
            backend_sampling_seed=1,
        )
        try:
            return await transport.send(
                SimpleNamespace(runtime_session_id="session"),
                node,
                SyntheticPrompt("", (), ({"role": "user", "content": "hello"},)),
            )
        finally:
            await transport.close()
            await writer.close()

    result = asyncio.run(send())
    assert result.success == success
    assert result.assistant_content == "answer"
    assert result.assistant_reasoning_content == "thinking"
    assert result.input_tokens == actual
    facts = load_request_facts(tmp_path / "requests.jsonl")
    assert facts[0].status == ("success" if success else "error")
    assert facts[0].input_tokens == actual
    if not success:
        assert "input verification failed" in result.error


def test_exact_token_validation_rejects_successful_response_with_wrong_usage(tmp_path: Path) -> None:
    async def send() -> tuple[object, object]:
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {"base_url": "http://backend", "endpoint": "/v1/chat/completions"},
                "replay": {
                    "trace_type": "tracelab",
                    "trace_path": "rounds.jsonl",
                    "prompt_calibration_tolerance_tokens": 0,
                },
            }
        )
        trace = tmp_path / "tracelab-requests.jsonl"
        writer = RequestTraceWriter(trace)
        await writer.start()
        transport = ReplayTransport(config, "run", writer)
        await transport.client.aclose()

        def handler(request: httpx.Request) -> httpx.Response:
            stream = (
                'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

        transport.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        node = SimpleNamespace(
            runtime_request_id="tracelab-request",
            actor_id="lead",
            actor_role="lead",
            prompt_kind="lead_main",
            parent_actor_id=None,
            planned_input_tokens=12,
            planned_output_tokens=3,
            backend_sampling_seed=7,
            response_validation="exact_tokens",
        )
        result = await transport.send(
            SimpleNamespace(runtime_session_id="tracelab-session"),
            node,
            SyntheticPrompt("", (), ({"role": "user", "content": "message"},)),
        )
        await transport.close()
        await writer.close()
        return result, load_request_facts(trace)[0]

    result, fact = asyncio.run(send())

    assert result.success is False
    assert "input expected=12 observed=11" in str(result.error)
    assert fact.status == "error"
    assert fact.input_tokens == 11


@pytest.mark.parametrize("trace_type", ["agentinfer", "tracelab"])
@pytest.mark.parametrize("output_ids", [None, [90], [90, 91]])
def test_token_continuations_require_actual_output_ids(tmp_path: Path, trace_type: str, output_ids: list[int] | None):
    """AgentX's optional output IDs must not weaken continuation validation."""

    async def send():
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {"base_url": "http://backend"},
                "replay": {"trace_type": trace_type, "prompt_transport": "completion_token_ids"},
            }
        )
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        await writer.start()
        transport = ReplayTransport(config, "run", writer)
        await transport.client.aclose()

        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/completions"
            assert json.loads(request.content)["return_token_ids"] is True
            choice = {"text": "answer"}
            if output_ids is not None:
                choice["token_ids"] = output_ids
            event = {"choices": [choice], "usage": {"prompt_tokens": 12, "completion_tokens": 2}}
            return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

        transport.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        node = SimpleNamespace(
            runtime_request_id="request",
            actor_id="lead",
            actor_role="lead",
            prompt_kind="lead_main",
            parent_actor_id=None,
            planned_input_tokens=12,
            planned_output_tokens=2,
            backend_sampling_seed=7,
        )
        try:
            return await transport.send(
                SimpleNamespace(runtime_session_id="session"),
                node,
                SyntheticPrompt("", (), (), token_ids=tuple(range(12))),
            )
        finally:
            await transport.close()
            await writer.close()

    result = asyncio.run(send())
    assert result.success is (output_ids == [90, 91])
    if result.success:
        assert result.assistant_token_ids == (90, 91)
    else:
        assert "output token_ids missing or incomplete" in result.error


def test_agentx_ignores_malformed_content_chunk_but_requires_valid_usage(tmp_path: Path) -> None:
    async def send():
        config = ReplayBenchConfig.model_validate(
            {
                "experiment": {"task_num": 1},
                "backend": {"base_url": "http://backend", "endpoint": "/v1/completions"},
                "replay": {"trace_type": "agentX"},
            }
        )
        writer = RequestTraceWriter(tmp_path / "requests.jsonl")
        await writer.start()
        transport = ReplayTransport(config, "run", writer)
        await transport.client.aclose()

        def respond(_request: httpx.Request) -> httpx.Response:
            stream = (
                'data: {"choices":[{"text":"unterminated}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=stream)

        transport.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        node = SimpleNamespace(
            runtime_request_id="request",
            actor_id="lead",
            actor_role="lead",
            prompt_kind="lead_main",
            parent_actor_id=None,
            planned_input_tokens=12,
            planned_output_tokens=2,
            backend_sampling_seed=7,
        )
        try:
            return await transport.send(
                SimpleNamespace(runtime_session_id="session"),
                node,
                SyntheticPrompt("", (), (), token_ids=tuple(range(12))),
            )
        finally:
            await transport.close()
            await writer.close()

    result = asyncio.run(send())
    assert result.success is True
    assert result.input_tokens == 12
    assert result.output_tokens == 2
