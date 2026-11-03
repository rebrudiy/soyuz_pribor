import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from application.llm import LLMError
from infrastructure.codex_client import CodexJsonClient

SCHEMA = {"type": "object"}


def test_codex_uses_isolated_structured_request_and_preserves_facts(monkeypatch):
    captured = {}
    process = AsyncMock()
    process.returncode = 0

    async def spawn(*args, **kwargs):
        captured.update(args=args, options=kwargs)
        folder = Path(kwargs["cwd"])
        assert json.loads((folder / "schema.json").read_text()) == SCHEMA
        (folder / "result.json").write_text('{"title":"Манометр"}')
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    result = asyncio.run(
        CodexJsonClient().generate(
            {"specs": {"zero": 0, "missing": None}},
            instructions="Use supplied facts only.",
            schema=SCHEMA,
        )
    )
    assert result == {"title": "Манометр"}
    assert captured["args"][:2] == ("codex", "exec")
    assert "--ignore-user-config" in captured["args"]
    assert "--ephemeral" in captured["args"]
    disabled = {
        captured["args"][index + 1]
        for index, option in enumerate(captured["args"][:-1])
        if option == "--disable"
    }
    assert {"shell_tool", "apps", "multi_agent", "browser_use", "plugins"} <= disabled
    assert 'web_search="disabled"' in captured["args"]
    assert captured["args"][captured["args"].index("--sandbox") + 1] == "read-only"
    prompt = process.communicate.call_args.args[0].decode()
    assert "Use supplied facts only." in prompt
    assert json.dumps({"specs": {"zero": 0, "missing": None}}) in prompt
    assert not Path(captured["options"]["cwd"]).exists()


def test_missing_codex_is_a_safe_application_error(monkeypatch):
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError())
    )
    with pytest.raises(LLMError, match="Codex"):
        asyncio.run(CodexJsonClient().generate({}, instructions="", schema=SCHEMA))


@pytest.mark.parametrize(
    ("returncode", "output"),
    [
        (1, '{"private":"provider details"}'),
        (0, "not JSON"),
        (0, "[]"),
        (0, '{"x":NaN}'),
    ],
)
def test_failed_or_invalid_codex_output_is_not_returned(
    monkeypatch, returncode, output
):
    process = AsyncMock()
    process.returncode = returncode

    async def spawn(*args, **kwargs):
        Path(kwargs["cwd"], "result.json").write_text(output)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(LLMError) as error:
        asyncio.run(CodexJsonClient().generate({}, instructions="", schema=SCHEMA))
    assert "provider details" not in str(error.value)


def test_codex_timeout_kills_and_reaps_process(monkeypatch):
    process = AsyncMock()
    process.returncode = None
    process.kill = Mock()

    async def communicate(*args):
        await asyncio.Event().wait()

    process.communicate.side_effect = communicate
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    with pytest.raises(LLMError, match="timed out"):
        asyncio.run(
            CodexJsonClient(timeout=0.01).generate({}, instructions="", schema=SCHEMA)
        )
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()


def test_codex_cancellation_reaps_process_and_propagates(monkeypatch):
    process = AsyncMock()
    process.returncode = None
    process.kill = Mock()
    started = None

    async def communicate(*args):
        started.set()
        await asyncio.Event().wait()

    process.communicate.side_effect = communicate
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )

    async def run():
        nonlocal started
        started = asyncio.Event()
        task = asyncio.create_task(
            CodexJsonClient().generate({}, instructions="", schema=SCHEMA)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()


def test_invalid_input_does_not_start_codex(monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(LLMError, match="supported JSON"):
        asyncio.run(
            CodexJsonClient().generate(
                {"value": float("nan")}, instructions="", schema=SCHEMA
            )
        )
    spawn.assert_not_called()
