"""Test helpers: LangChain's fake chat model stands in for the LLM, so the graph runs without Ollama or an API key."""
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import main


class FakeLLM(GenericFakeChatModel):
    """GenericFakeChatModel returns the scripted messages in order. main.py calls
    `llm.with_structured_output(Schema)`, which needs tool calling, so `bind_tools` is a no-op here and every
    scripted answer is an AIMessage with a tool call named after the schema (see `claim_data` and `coverage`).
    LangChain then parses that tool call into the Pydantic object, as it would for a real model."""

    def bind_tools(self, tools, **kwargs):
        return self


def _tool_call(schema: type, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": schema.__name__, "args": args, "id": schema.__name__}],
    )


def claim_data(**fields) -> AIMessage:
    """Scripted intake answer; every ClaimData field defaults to None."""
    return _tool_call(
        main.ClaimData, {name: fields.get(name) for name in main.ClaimData.model_fields}
    )


def coverage(
    article: int, covered: bool, reasoning: str = "Scripted coverage answer."
) -> AIMessage:
    """Scripted coverage answer."""
    return _tool_call(
        main.CoverageResult,
        {"reasoning": reasoning, "article": article, "covered": covered},
    )


def unused_answers(fake: FakeLLM) -> list[str]:
    """Schema names of the scripted answers the graph never asked for (consumes the rest of the script)."""
    return [message.tool_calls[0]["name"] for message in fake.messages]


@pytest.fixture
def fake_llm(monkeypatch):
    """Factory: `fake = fake_llm(claim_data(...), coverage(...))` installs a FakeLLM as `main.llm`.
    Answers are given in the order the graph asks for them: intake (once per round), then coverage.
    When the script runs out, the model call raises, which the nodes treat like a model error."""

    def install(*answers: AIMessage):
        fake = FakeLLM(messages=iter(answers))
        monkeypatch.setattr(main, "llm", fake)
        return fake

    return install


@pytest.fixture
def run():
    """`run(message, follow_ups=[], assessor="approve")` runs a claim through the compiled graph."""
    graph = main.build_graph()

    def _run(message, follow_ups=(), assessor="approve"):
        answers = list(follow_ups)

        def respond(payload):
            if payload["type"] == "missing_data":
                return answers.pop(0) if answers else "I don't have that information."
            return {"decision": assessor, "note": "test assessor"}

        return main.run_claim(graph, message, respond)

    return _run
