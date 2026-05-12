# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for Gemma4 streaming tool-call id propagation.

``Gemma4ToolParser.extract_tool_calls_streaming(...)`` mints a tool-call
id on the first chunk of each tool call (the one that carries the
function name) but historically left ``id`` unset on every subsequent
argument-diff and end-flush chunk for that same tool call.

The OpenAI streaming spec only *requires* ``id`` on the first chunk, so
this is technically permitted, but some strict client validators -- most
notably Vercel's ``@ai-sdk/openai-compatible`` provider, used by
``opencode`` -- reject ``DeltaToolCall`` entries whose ``id`` is
``null``/missing with::

    AI_InvalidResponseDataError: Expected 'id' to be a string

This test pumps a small streaming transcript through the parser one
slice at a time and asserts that *every* emitted ``DeltaToolCall``
carries the same non-empty string id minted on the first chunk.  It
also checks that ``_reset_streaming_state()`` mints a fresh id for the
next request (i.e. no id leakage across requests).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

# Token strings the parser keys off of (mirrors gemma4_tool_parser.py).
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"


@pytest.fixture
def gemma4_tokenizer(default_tokenizer):
    """Augment the default test tokenizer's vocab with the two special
    tokens Gemma4ToolParser looks up at construction time."""
    tokenizer = default_tokenizer
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer.get_vocab = MagicMock()
    tokenizer_vocab.update(
        {
            TOOL_CALL_START: 100001,
            TOOL_CALL_END: 100002,
        }
    )
    tokenizer.get_vocab.return_value = tokenizer_vocab
    return tokenizer


@pytest.fixture
def gemma4_parser(gemma4_tokenizer):
    return Gemma4ToolParser(gemma4_tokenizer)


# A representative gemma4 tool-call transcript: one tool call with one
# string argument and one numeric argument, surrounded by chat text.  The
# ``<|"|>`` sequences are Gemma4's string delimiter (not Python f-string
# interpolation).
_TRANSCRIPT = (
    "Some preamble.\n"
    f"{TOOL_CALL_START}call:get_weather"
    '{city:<|"|>Tokyo<|"|>,days:3}'
    f"{TOOL_CALL_END}"
    "And some trailing text."
)


def _stream_chunks(parser: Gemma4ToolParser, full_text: str, *, chunk_size: int):
    """Feed ``full_text`` through ``parser`` one slice at a time and
    collect every emitted ``DeltaToolCall`` as a tuple."""
    seen: list[tuple[int, str | None, str | None, str | None]] = []
    previous = ""
    for i in range(0, len(full_text), chunk_size):
        delta_text = full_text[i : i + chunk_size]
        current_text = previous + delta_text
        delta_msg = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        previous = current_text
        if delta_msg is None or not getattr(delta_msg, "tool_calls", None):
            continue
        for tc in delta_msg.tool_calls:
            fn = getattr(tc, "function", None) or {}
            if not isinstance(fn, dict):
                fn = fn.model_dump(exclude_none=True)
            seen.append(
                (
                    tc.index,
                    getattr(tc, "id", None),
                    fn.get("name"),
                    fn.get("arguments"),
                )
            )
    return seen


@pytest.mark.parametrize("chunk_size", [1, 4, 8, 16])
def test_every_streaming_chunk_carries_the_same_string_id(
    gemma4_parser: Gemma4ToolParser, chunk_size: int
):
    chunks = _stream_chunks(gemma4_parser, _TRANSCRIPT, chunk_size=chunk_size)

    assert chunks, (
        f"expected at least one DeltaToolCall to be emitted "
        f"(chunk_size={chunk_size})"
    )

    # First chunk (the function-name chunk) must carry a non-empty id.
    first_id = chunks[0][1]
    assert isinstance(first_id, str) and first_id, (
        f"first chunk id must be a non-empty string, got {first_id!r} "
        f"(chunk_size={chunk_size})"
    )

    # Every subsequent chunk for this tool-call index must re-emit the
    # SAME id.  Stock vLLM left these as None, which strict client
    # validators (e.g. @ai-sdk Zod) reject mid-stream.
    for idx, (tc_index, tc_id, name, args) in enumerate(chunks):
        assert isinstance(tc_id, str), (
            f"chunk {idx} (tool_index={tc_index}, name={name!r}, "
            f"args={args!r}, chunk_size={chunk_size}) has non-string id "
            f"{tc_id!r}; strict client validators will reject this"
        )
        assert tc_id, (
            f"chunk {idx} has empty id (chunk_size={chunk_size}); "
            f"expected the persisted first-chunk id {first_id!r}"
        )
        assert tc_id == first_id, (
            f"chunk {idx} id={tc_id!r} differs from first-chunk id "
            f"{first_id!r} (chunk_size={chunk_size}); tool_call_ids "
            f"tracking is broken"
        )


def test_id_is_reset_between_requests(gemma4_parser: Gemma4ToolParser):
    """A fresh request (after ``_reset_streaming_state()``) must mint a
    fresh tool-call id rather than leaking the previous request's id."""
    first = _stream_chunks(gemma4_parser, _TRANSCRIPT, chunk_size=6)
    gemma4_parser._reset_streaming_state()
    second = _stream_chunks(gemma4_parser, _TRANSCRIPT, chunk_size=6)

    assert first and second, "both requests should emit at least one chunk"
    assert first[0][1] != second[0][1], (
        "expected a fresh tool-call id after _reset_streaming_state(); "
        f"got the same id twice: {first[0][1]!r}"
    )
