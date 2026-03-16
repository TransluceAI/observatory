import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from llm_util.types import ChatMessage, LLMOutput, ResponseFormat, ToolInfo


class LLMCache:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            llm_cache_path = os.getenv("LLM_CACHE_PATH")
            if llm_cache_path is None:
                raise Exception("LLM_CACHE_PATH is not set in .env")

            cache_dir = Path(llm_cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "llm_cache.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    completion TEXT,
                    model_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _create_key(
        self,
        messages: list[ChatMessage],
        model_name: str,
        *,
        tools: list[ToolInfo] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
        reasoning_effort: Literal["low", "medium", "high", "max"] | None = None,
        temperature: float = 1.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        response_format: ResponseFormat | None = None,
    ) -> str:
        """Create a deterministic hash key from messages and model."""
        # Convert messages to a stable string representation
        message_str = json.dumps(
            [msg.model_dump(exclude={"id"}) for msg in messages], sort_keys=True
        )

        # Convert tools to a stable string representation if present
        tools_str = (
            json.dumps([tool.model_dump() for tool in tools], sort_keys=True) if tools else None
        )

        # Convert response_format to a stable string
        response_format_str = (
            json.dumps(response_format.model_dump(by_alias=True), sort_keys=True)
            if response_format
            else None
        )

        # Combine all parameters into a single string
        key_str = (
            f"{message_str}:{model_name}:{tools_str}:{tool_choice}:"
            f"{reasoning_effort}:{temperature}:{response_format_str}"
        )
        if logprobs:
            key_str += f":{top_logprobs}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self,
        messages: list[ChatMessage],
        model_name: str,
        *,
        tools: list[ToolInfo] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
        reasoning_effort: Literal["low", "medium", "high", "max"] | None = None,
        temperature: float = 1.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        response_format: ResponseFormat | None = None,
    ) -> LLMOutput | None:
        """Get cached completion for a conversation if it exists."""
        key = self._create_key(
            messages,
            model_name,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            response_format=response_format,
        )

        with self._get_connection() as conn:
            cursor = conn.execute("SELECT completion FROM llm_cache WHERE key = ?", (key,))
            result = cursor.fetchone()
            if result:
                output = LLMOutput.model_validate_json(result[0])
                output.from_cache = True
                return output
            return None

    def set(
        self,
        messages: list[ChatMessage],
        model_name: str,
        llm_output: LLMOutput,
        *,
        tools: list[ToolInfo] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
        reasoning_effort: Literal["low", "medium", "high", "max"] | None = None,
        temperature: float = 1.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        response_format: ResponseFormat | None = None,
    ) -> None:
        """Cache a completion for a conversation."""
        key = self._create_key(
            messages,
            model_name,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            response_format=response_format,
        )

        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, completion, model_name) VALUES (?, ?, ?)",
                (key, llm_output.model_dump_json(), model_name),
            )
            conn.commit()

    def set_batch(
        self,
        messages_list: list[list[ChatMessage]],
        model_name: str,
        llm_output_list: list[LLMOutput],
        *,
        tools: list[ToolInfo] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
        reasoning_effort: Literal["low", "medium", "high", "max"] | None = None,
        temperature: float = 1.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        response_format: ResponseFormat | None = None,
    ) -> None:
        """Cache a batch of completions."""
        keys: list[str] = []
        for messages in messages_list:
            key = self._create_key(
                messages,
                model_name,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                response_format=response_format,
            )
            keys.append(key)

        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO llm_cache (key, completion, model_name) VALUES (?, ?, ?)",
                [
                    (key, llm_output.model_dump_json(), model_name)
                    for key, llm_output in zip(keys, llm_output_list)
                ],
            )
            conn.commit()

    def clear(self) -> None:
        """Clear all cached completions."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM llm_cache")
            conn.commit()
