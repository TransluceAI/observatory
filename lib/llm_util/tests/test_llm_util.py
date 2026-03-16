"""Tests for the llm_util library.

Tests marked with @pytest.mark.api call real LLM APIs and should be skipped in CI.
"""

import json
import tempfile

import pytest
from llm_util.llm_cache import LLMCache
from llm_util.types import (
    AssistantMessage,
    ChatMessage,
    ContentReasoning,
    ContentRedactedReasoning,
    ContentText,
    LLMCompletion,
    LLMCompletionPartial,
    LLMOutput,
    LLMOutputPartial,
    ModelCallParams,
    ResponseFormat,
    SystemMessage,
    ToolCall,
    ToolInfo,
    ToolMessage,
    ToolParam,
    ToolParams,
    Usage,
    UserMessage,
    finalize_llm_output_partial,
    parse_chat_message,
)

# ---------------------------------------------------------------------------
# Type construction tests
# ---------------------------------------------------------------------------


class TestMessageTypes:
    def test_system_message(self):
        msg = SystemMessage(content="You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.text == "You are a helpful assistant."

    def test_user_message(self):
        msg = UserMessage(content="Hello!")
        assert msg.role == "user"
        assert msg.text == "Hello!"

    def test_assistant_message(self):
        msg = AssistantMessage(content="Hi there!")
        assert msg.role == "assistant"
        assert msg.text == "Hi there!"
        assert msg.tool_calls is None

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="tc1", function="get_weather", arguments={"city": "SF"})
        msg = AssistantMessage(content="", tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function == "get_weather"

    def test_tool_message(self):
        msg = ToolMessage(content="72°F", tool_call_id="tc1", function="get_weather")
        assert msg.role == "tool"
        assert msg.text == "72°F"
        assert msg.tool_call_id == "tc1"


class TestContentTypes:
    def test_content_text(self):
        ct = ContentText(text="hello")
        assert ct.type == "text"
        assert ct.text == "hello"

    def test_content_reasoning(self):
        cr = ContentReasoning(reasoning="thinking...", signature="sig123")
        assert cr.type == "reasoning"
        assert cr.reasoning == "thinking..."
        assert cr.signature == "sig123"

    def test_content_redacted_reasoning(self):
        cr = ContentRedactedReasoning(data="abc")
        assert cr.type == "redacted_reasoning"
        assert cr.data == "abc"

    def test_message_with_content_list(self):
        msg = UserMessage(
            content=[
                ContentText(text="Hello"),
                ContentText(text=" world"),
            ]
        )
        assert msg.text == "Hello world"

    def test_discriminated_union_parsing(self):
        """Content types can be parsed from dicts."""
        text_data = {"type": "text", "text": "hello"}
        ct = ContentText.model_validate(text_data)
        assert ct.text == "hello"

        reason_data = {"type": "reasoning", "reasoning": "hmm"}
        cr = ContentReasoning.model_validate(reason_data)
        assert cr.reasoning == "hmm"


class TestToolTypes:
    def test_tool_call(self):
        tc = ToolCall(id="tc1", function="search", arguments={"q": "test"})
        assert tc.id == "tc1"
        assert tc.function == "search"
        assert tc.arguments == {"q": "test"}
        assert tc.type is None

    def test_tool_param(self):
        tp = ToolParam(
            name="query",
            description="Search query",
            input_schema={"type": "string"},
        )
        assert tp.name == "query"

    def test_tool_params(self):
        tp = ToolParams(
            properties={
                "query": ToolParam(
                    name="query",
                    description="Search query",
                    input_schema={"type": "string"},
                )
            },
            required=["query"],
        )
        dumped = tp.model_dump(exclude_none=True)
        assert dumped["type"] == "object"
        assert "query" in dumped["properties"]
        assert dumped["required"] == ["query"]

    def test_tool_info(self):
        ti = ToolInfo(
            name="search",
            description="Search the web",
            parameters=ToolParams(
                properties={
                    "query": ToolParam(
                        name="query",
                        description="Search query",
                        input_schema={"type": "string"},
                    )
                },
                required=["query"],
            ),
        )
        assert ti.name == "search"
        dumped = ti.model_dump(exclude_none=True)
        assert dumped["parameters"]["type"] == "object"

    def test_tool_info_model_dump_for_anthropic(self):
        """parameters.model_dump(exclude_none=True) should produce valid input_schema."""
        ti = ToolInfo(
            name="test",
            description="A test tool",
            parameters=ToolParams(
                properties={
                    "x": ToolParam(name="x", description="An int", input_schema={"type": "integer"})
                },
                required=["x"],
            ),
        )
        schema = ti.parameters.model_dump(exclude_none=True)
        assert schema["type"] == "object"
        assert "x" in schema["properties"]


class TestResponseFormat:
    def test_response_format(self):
        rf = ResponseFormat(
            name="my_schema",
            schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            strict=True,
        )
        assert rf.name == "my_schema"
        assert rf.strict is True
        dumped = rf.model_dump(by_alias=True)
        assert "schema" in dumped
        assert dumped["schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


class TestLLMCompletion:
    def test_basic(self):
        c = LLMCompletion(text="hello", finish_reason="stop")
        assert c.text == "hello"
        assert c.no_text is False
        assert c.reasoning is None

    def test_no_text(self):
        c = LLMCompletion(text=None)
        assert c.no_text is True
        c2 = LLMCompletion(text="")
        assert c2.no_text is True

    def test_reasoning(self):
        c = LLMCompletion(
            text="answer",
            reasoning=ContentReasoning(reasoning="thinking about it..."),
        )
        assert c.reasoning is not None
        assert c.reasoning.reasoning == "thinking about it..."


class TestLLMOutput:
    def test_basic(self):
        out = LLMOutput(
            model="test",
            completions=[LLMCompletion(text="hi")],
        )
        assert out.non_empty is True
        assert out.first_text == "hi"
        assert out.did_error is False
        assert out.from_cache is False
        assert out.duration is None

    def test_empty(self):
        out = LLMOutput(model="test", completions=[])
        assert out.non_empty is False
        assert out.first is None
        assert out.first_text is None

    def test_with_errors(self):
        out = LLMOutput(model="test", completions=[], errors=["rate_limit"])
        assert out.did_error is True

    def test_from_cache_and_duration(self):
        out = LLMOutput(
            model="test",
            completions=[LLMCompletion(text="cached")],
            from_cache=True,
            duration=0.5,
        )
        assert out.from_cache is True
        assert out.duration == 0.5

    def test_serialization_roundtrip(self):
        out = LLMOutput(
            model="test",
            completions=[
                LLMCompletion(
                    text="hi",
                    finish_reason="stop",
                    reasoning=ContentReasoning(reasoning="thought"),
                )
            ],
            usage=Usage(input_tokens=10, output_tokens=5),
            from_cache=True,
        )
        json_str = out.model_dump_json()
        restored = LLMOutput.model_validate_json(json_str)
        assert restored.first_text == "hi"
        assert restored.first is not None
        assert restored.first.reasoning is not None
        assert restored.first.reasoning.reasoning == "thought"
        assert restored.from_cache is True


class TestUsage:
    def test_add(self):
        u1 = Usage(input_tokens=10, output_tokens=5)
        u2 = Usage(input_tokens=20, output_tokens=10)
        total = u1 + u2
        assert total.input_tokens == 30
        assert total.output_tokens == 15

    def test_sum(self):
        usages = [Usage(input_tokens=10), Usage(input_tokens=20)]
        total = sum(usages)
        assert isinstance(total, Usage)
        assert total.input_tokens == 30

    def test_compute_cost(self):
        u = Usage(cache_read_input_tokens=100, input_tokens=1000, output_tokens=500)
        cost = u.compute_cost(cache_cost=0.5, input_cost=3.0, output_cost=15.0)
        assert cost > 0


# ---------------------------------------------------------------------------
# Finalize partial
# ---------------------------------------------------------------------------


class TestFinalizePartial:
    def test_basic(self):
        partial = LLMOutputPartial(
            model="test",
            completions=[
                LLMCompletionPartial(
                    text="hello",
                    finish_reason="stop",
                    reasoning=ContentReasoning(reasoning="hmm"),
                )
            ],
        )
        output = finalize_llm_output_partial(partial)
        assert output.first_text == "hello"
        assert output.first is not None
        assert output.first.reasoning is not None
        assert output.first.reasoning.reasoning == "hmm"


# ---------------------------------------------------------------------------
# Parse chat message
# ---------------------------------------------------------------------------


class TestParseChatMessage:
    def test_system(self):
        msg = parse_chat_message({"role": "system", "content": "sys"})
        assert isinstance(msg, SystemMessage)
        assert msg.text == "sys"

    def test_user(self):
        msg = parse_chat_message({"role": "user", "content": "hi"})
        assert isinstance(msg, UserMessage)

    def test_assistant(self):
        msg = parse_chat_message({"role": "assistant", "content": "hey"})
        assert isinstance(msg, AssistantMessage)

    def test_tool(self):
        msg = parse_chat_message({"role": "tool", "content": "result", "tool_call_id": "tc1"})
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "tc1"

    def test_unknown_role(self):
        with pytest.raises(ValueError, match="Unknown message role"):
            parse_chat_message({"role": "unknown", "content": "?"})


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestLLMCache:
    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(db_path=f"{tmpdir}/test.db")
            messages: list[ChatMessage] = [UserMessage(content="hello")]
            output = LLMOutput(
                model="test",
                completions=[LLMCompletion(text="world", finish_reason="stop")],
            )
            cache.set(messages, "test-model", output)
            result = cache.get(messages, "test-model")
            assert result is not None
            assert result.first_text == "world"
            assert result.from_cache is True

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(db_path=f"{tmpdir}/test.db")
            messages: list[ChatMessage] = [UserMessage(content="hello")]
            result = cache.get(messages, "test-model")
            assert result is None

    def test_response_format_in_key(self):
        """Different response_format should produce different cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(db_path=f"{tmpdir}/test.db")
            messages: list[ChatMessage] = [UserMessage(content="hello")]
            output = LLMOutput(
                model="test",
                completions=[LLMCompletion(text="answer")],
            )
            rf = ResponseFormat(
                name="test",
                schema={"type": "object"},
                strict=True,
            )
            cache.set(messages, "test-model", output, response_format=rf)

            # With response_format -> hit
            result = cache.get(messages, "test-model", response_format=rf)
            assert result is not None

            # Without response_format -> miss
            result = cache.get(messages, "test-model")
            assert result is None

    def test_set_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(db_path=f"{tmpdir}/test.db")
            msgs1: list[ChatMessage] = [UserMessage(content="a")]
            msgs2: list[ChatMessage] = [UserMessage(content="b")]
            out1 = LLMOutput(model="m", completions=[LLMCompletion(text="x")])
            out2 = LLMOutput(model="m", completions=[LLMCompletion(text="y")])
            cache.set_batch([msgs1, msgs2], "m", [out1, out2])

            r1 = cache.get(msgs1, "m")
            r2 = cache.get(msgs2, "m")
            assert r1 is not None and r1.first_text == "x"
            assert r2 is not None and r2.first_text == "y"


# ---------------------------------------------------------------------------
# API tests (require real API keys)
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestCompleteAsync:
    @pytest.fixture
    def anthropic_model(self):
        return ModelCallParams(
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
        )

    @pytest.fixture
    def openai_model(self):
        return ModelCallParams(
            provider="openai",
            model_name="gpt-4o-mini-2024-07-18",
        )

    @pytest.mark.asyncio
    async def test_complete_async_anthropic(self, anthropic_model):
        from llm_util.prod_llms import complete_async

        result = await complete_async(
            messages=[UserMessage(content="Say 'hello' and nothing else.")],
            model=anthropic_model,
            max_new_tokens=32,
            temperature=0.0,
            timeout=30.0,
        )
        assert result.non_empty
        assert result.first_text is not None
        assert "hello" in result.first_text.lower()

    @pytest.mark.asyncio
    async def test_complete_async_openai(self, openai_model):
        from llm_util.prod_llms import complete_async

        result = await complete_async(
            messages=[UserMessage(content="Say 'hello' and nothing else.")],
            model=openai_model,
            max_new_tokens=32,
            temperature=0.0,
            timeout=30.0,
        )
        assert result.non_empty
        assert result.first_text is not None

    @pytest.mark.asyncio
    async def test_reasoning_captured(self):
        """Extended thinking models should capture reasoning as ContentReasoning."""
        from llm_util.prod_llms import complete_async

        model = ModelCallParams(
            provider="anthropic",
            model_name="claude-sonnet-4-6",
            reasoning_effort="high",
        )
        result = await complete_async(
            messages=[
                UserMessage(
                    content=(
                        "A farmer has 17 sheep. All but 9 die. "
                        "How many sheep are left? Think step by step."
                    )
                )
            ],
            model=model,
            max_new_tokens=8192,
            temperature=1.0,
            timeout=120.0,
        )
        assert result.non_empty
        assert result.first is not None
        # With high effort on a reasoning question, the model should think
        assert result.first.reasoning is not None
        assert isinstance(result.first.reasoning, ContentReasoning)
        assert len(result.first.reasoning.reasoning) > 0

    @pytest.mark.asyncio
    async def test_multi_step_tool_calls(self):
        """Multi-turn reasoning with tool calls."""
        from llm_util.prod_llms import complete_async

        model = ModelCallParams(
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
        )
        tools = [
            ToolInfo(
                name="get_weather",
                description="Get the weather for a city",
                parameters=ToolParams(
                    properties={
                        "city": ToolParam(
                            name="city",
                            description="City name",
                            input_schema={"type": "string"},
                        )
                    },
                    required=["city"],
                ),
            )
        ]

        # Turn 1: Send a message with tools
        result = await complete_async(
            messages=[UserMessage(content="What's the weather in San Francisco?")],
            model=model,
            tools=tools,
            tool_choice="auto",
            max_new_tokens=256,
            temperature=0.0,
            timeout=30.0,
        )
        assert result.non_empty
        first = result.first
        assert first is not None
        assert first.tool_calls is not None
        assert len(first.tool_calls) > 0
        assert first.tool_calls[0].function == "get_weather"

        # Turn 2: Send tool result back
        tc = first.tool_calls[0]
        messages: list[ChatMessage] = [
            UserMessage(content="What's the weather in San Francisco?"),
            AssistantMessage(content=first.text or "", tool_calls=first.tool_calls),
            ToolMessage(
                content=json.dumps({"temperature": "72F", "condition": "sunny"}),
                tool_call_id=tc.id,
                function=tc.function,
            ),
        ]
        result2 = await complete_async(
            messages=messages,
            model=model,
            tools=tools,
            max_new_tokens=256,
            temperature=0.0,
            timeout=30.0,
        )
        assert result2.non_empty
        assert result2.first_text is not None

    @pytest.mark.asyncio
    async def test_structured_output_openai(self):
        """ResponseFormat structured output with OpenAI."""
        from llm_util.prod_llms import complete_async

        model = ModelCallParams(
            provider="openai",
            model_name="gpt-4o-mini-2024-07-18",
        )
        rf = ResponseFormat(
            name="answer",
            schema={
                "type": "object",
                "properties": {
                    "result": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                "required": ["result", "explanation"],
                "additionalProperties": False,
            },
            strict=True,
        )
        result = await complete_async(
            messages=[UserMessage(content="What is 2+2? Return as JSON.")],
            model=model,
            response_format=rf,
            max_new_tokens=128,
            temperature=0.0,
            timeout=30.0,
        )
        assert result.non_empty
        text = result.first_text
        assert text is not None
        parsed = json.loads(text)
        assert parsed["result"] == 4


@pytest.mark.api
class TestGetLLMCompletionsAsync:
    @pytest.mark.asyncio
    async def test_batch_completions(self):
        from llm_util.prod_llms import get_llm_completions_async

        model = ModelCallParams(provider="anthropic", model_name="claude-3-haiku-20240307")
        results = await get_llm_completions_async(
            messages_list=[
                [UserMessage(content="Say 'one'")],
                [UserMessage(content="Say 'two'")],
            ],
            model=model,
            max_new_tokens=32,
            temperature=0.0,
            timeout=30.0,
        )
        assert len(results) == 2
        assert all(r.non_empty for r in results)
