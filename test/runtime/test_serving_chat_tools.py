"""Tool-shape contract for the Jinja chat-template path.

OpenAI request bodies wrap each tool as ``{"type": "function",
"function": {...}}``. Chat templates split into two camps:

* Wrapped: read ``tool.function.X`` (MiniMax-M2, DeepSeek).
* Flat: read ``tool.X`` directly (Llama-3, Hermes, Qwen, gpt-oss).

``_chat_template_expects_wrapped_tools`` decides which camp the
template falls in by scanning the Jinja AST for ``X.function``
access where ``X`` is bound to a tool element (directly via
``for X in tools``, or transitively through ``set`` rebinds and
macro calls). The serving layer dispatches once on that result —
no retry, no merged-shape dict that would corrupt
``tojson``-ing templates.

Mocks the tokenizer for the dispatch test, so it's a pure unit
test that nonetheless needs CUDA only because the tokenspeed
import graph pulls in ``triton`` at import time. Registered as
``runtime-1gpu`` for that reason; the test itself does not touch
a GPU.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from tokenspeed.runtime.entrypoints.openai.serving_chat import (  # noqa: E402
    OpenAIServingChat,
    _chat_template_expects_wrapped_tools,
    _resolve_chat_template_for_tools,
)

# Mirrors MiniMax-M2.5-Plus's ``chat_template.jinja`` shape: macro indirection
# (the iter var is bound inside ``render_tool_namespace``) plus
# ``tool.function | tojson`` access. Pinning this template guards against AST
# walks that don't follow macro-call argument bindings.
_MINIMAX_LIKE_TEMPLATE = """
{%- macro render_tool_namespace(namespace_name, tool_list) -%}
{%- for tool in tool_list -%}
<tool>{{ tool.function | tojson(ensure_ascii=False) }}</tool>
{% endfor -%}
{%- endmacro -%}
{%- if tools -%}
<tools>
{{- render_tool_namespace("functions", tools) }}
</tools>
{%- endif -%}
"""

# Llama-3.x style: direct ``for tool in tools`` with flat field access.
_LLAMA_LIKE_TEMPLATE = """
{%- if tools -%}
{%- for tool in tools -%}
{{ tool.name }}: {{ tool.description }}
{{ tool.parameters | tojson(indent=4) }}
{%- endfor -%}
{%- endif -%}
"""

# Set-rebind: ``my_tools = tools`` — guards the alias-propagation step.
_SET_REBIND_WRAPPED_TEMPLATE = """
{%- set my_tools = tools -%}
{%- for t in my_tools -%}
{{ t.function.name }}
{%- endfor -%}
"""

# Template that doesn't iterate ``tools`` at all — defaults to flat.
_NO_TOOLS_ITER_TEMPLATE = """
{%- for m in messages -%}{{ m.role }}: {{ m.content }}{% endfor -%}
"""


def _build_request_with_one_tool() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="x",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ],
    )


def _build_serving_chat_with_mock_tokenizer(chat_template):
    """Build a real ``OpenAIServingChat`` with all GPU/engine deps stubbed.

    Uses the real ``__init__`` so the chat-template precompute path is
    exercised end-to-end — important because the wrapped/flat decision
    is locked in at construction time, not per request.
    """
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3])
    tokenizer.encode = MagicMock(return_value=[])
    tokenizer.chat_template = chat_template

    server_args = MagicMock()
    server_args.reasoning_parser = None
    server_args.tool_call_parser = None

    engine_client = MagicMock()
    engine_client.tokenizer = tokenizer
    engine_client.server_args = server_args

    template_manager = MagicMock()
    template_manager.chat_template_name = None
    template_manager.jinja_template_content_format = "string"

    chat = OpenAIServingChat(engine_client, template_manager)
    return chat, tokenizer


class TestChatTemplateShapeDetection(unittest.TestCase):
    """AST-level detection of wrapped vs flat tool-shape conventions."""

    def test_minimax_like_macro_indirection_is_wrapped(self):
        self.assertTrue(_chat_template_expects_wrapped_tools(_MINIMAX_LIKE_TEMPLATE))

    def test_llama_like_flat_access_is_flat(self):
        self.assertFalse(_chat_template_expects_wrapped_tools(_LLAMA_LIKE_TEMPLATE))

    def test_set_rebind_wrapped_is_wrapped(self):
        self.assertTrue(
            _chat_template_expects_wrapped_tools(_SET_REBIND_WRAPPED_TEMPLATE)
        )

    def test_template_without_tools_iteration_defaults_flat(self):
        self.assertFalse(_chat_template_expects_wrapped_tools(_NO_TOOLS_ITER_TEMPLATE))

    def test_empty_or_none_template_defaults_flat(self):
        self.assertFalse(_chat_template_expects_wrapped_tools(None))
        self.assertFalse(_chat_template_expects_wrapped_tools(""))

    def test_unparseable_template_defaults_flat(self):
        # Malformed Jinja — must not raise, must default to flat.
        self.assertFalse(_chat_template_expects_wrapped_tools("{% for x in %}"))


class TestResolveChatTemplateForTools(unittest.TestCase):
    """One resolver covers both the init path (no tokenizer arg) and the
    per-request override path (tokenizer arg drives name lookups)."""

    # --- structural selection (no name lookup) -----------------------------

    def test_string_is_template_body(self):
        self.assertEqual(_resolve_chat_template_for_tools("hello"), "hello")

    def test_dict_prefers_tool_use(self):
        out = _resolve_chat_template_for_tools(
            {"tool_use": _MINIMAX_LIKE_TEMPLATE, "default": _LLAMA_LIKE_TEMPLATE}
        )
        self.assertEqual(out, _MINIMAX_LIKE_TEMPLATE)

    def test_dict_falls_back_to_default(self):
        out = _resolve_chat_template_for_tools({"default": _LLAMA_LIKE_TEMPLATE})
        self.assertEqual(out, _LLAMA_LIKE_TEMPLATE)

    def test_dict_singleton_value(self):
        out = _resolve_chat_template_for_tools({"only": _LLAMA_LIKE_TEMPLATE})
        self.assertEqual(out, _LLAMA_LIKE_TEMPLATE)

    def test_list_of_dicts_prefers_tool_use(self):
        out = _resolve_chat_template_for_tools(
            [
                {"name": "default", "template": _LLAMA_LIKE_TEMPLATE},
                {"name": "tool_use", "template": _MINIMAX_LIKE_TEMPLATE},
            ]
        )
        self.assertEqual(out, _MINIMAX_LIKE_TEMPLATE)

    def test_unsupported_shapes_return_none(self):
        self.assertIsNone(_resolve_chat_template_for_tools(None))
        self.assertIsNone(_resolve_chat_template_for_tools(42))
        self.assertIsNone(_resolve_chat_template_for_tools({}))
        self.assertIsNone(_resolve_chat_template_for_tools([]))
        # Multi-entry dict with no recognized keys mirrors transformers'
        # own ValueError — we return None and the dispatcher falls back
        # to flat.
        self.assertIsNone(
            _resolve_chat_template_for_tools({"odd_a": "x", "odd_b": "y"})
        )

    # --- name lookup against a tokenizer's named-template registry ---------

    def test_named_override_against_dict_tokenizer(self):
        tok = {"tool_use": _MINIMAX_LIKE_TEMPLATE, "default": _LLAMA_LIKE_TEMPLATE}
        self.assertEqual(
            _resolve_chat_template_for_tools("tool_use", tok), _MINIMAX_LIKE_TEMPLATE
        )
        self.assertEqual(
            _resolve_chat_template_for_tools("default", tok), _LLAMA_LIKE_TEMPLATE
        )

    def test_named_override_against_list_tokenizer(self):
        tok = [
            {"name": "tool_use", "template": _MINIMAX_LIKE_TEMPLATE},
            {"name": "default", "template": _LLAMA_LIKE_TEMPLATE},
        ]
        self.assertEqual(
            _resolve_chat_template_for_tools("tool_use", tok), _MINIMAX_LIKE_TEMPLATE
        )

    def test_unmatched_name_falls_back_to_body(self):
        # No name match in the tokenizer dict → treat the string as the
        # template body itself (mirrors transformers).
        self.assertEqual(
            _resolve_chat_template_for_tools(
                _LLAMA_LIKE_TEMPLATE, {"tool_use": _MINIMAX_LIKE_TEMPLATE}
            ),
            _LLAMA_LIKE_TEMPLATE,
        )
        # No tokenizer registry at all: string is always the body.
        self.assertEqual(
            _resolve_chat_template_for_tools(_LLAMA_LIKE_TEMPLATE, None),
            _LLAMA_LIKE_TEMPLATE,
        )

    def test_dict_override_routes_through_structural_selection(self):
        # Even with a tokenizer registry supplied, a dict override is
        # selected structurally — name lookup applies only to strings.
        out = _resolve_chat_template_for_tools(
            {"tool_use": _MINIMAX_LIKE_TEMPLATE, "default": _LLAMA_LIKE_TEMPLATE},
            tokenizer_chat_template={"unrelated": "ignored"},
        )
        self.assertEqual(out, _MINIMAX_LIKE_TEMPLATE)


class TestJinjaToolDispatch(unittest.TestCase):
    """``_process_messages`` picks the shape that matches the template."""

    def test_wrapped_template_receives_wrapped_tools(self):
        chat, tokenizer = _build_serving_chat_with_mock_tokenizer(
            _MINIMAX_LIKE_TEMPLATE
        )
        req = _build_request_with_one_tool()

        chat._process_messages(req, is_multimodal=False)

        self.assertEqual(tokenizer.apply_chat_template.call_count, 1)
        passed = tokenizer.apply_chat_template.call_args.kwargs["tools"]
        expected = [tool.model_dump() for tool in req.tools]
        self.assertEqual(passed, expected)

    def test_flat_template_receives_flat_tools(self):
        chat, tokenizer = _build_serving_chat_with_mock_tokenizer(_LLAMA_LIKE_TEMPLATE)
        req = _build_request_with_one_tool()

        chat._process_messages(req, is_multimodal=False)

        self.assertEqual(tokenizer.apply_chat_template.call_count, 1)
        passed = tokenizer.apply_chat_template.call_args.kwargs["tools"]
        expected = [tool.function.model_dump() for tool in req.tools]
        self.assertEqual(passed, expected)

    def test_dict_template_with_wrapped_tool_use_dispatches_wrapped(self):
        """Dict-shaped ``chat_template`` must score the ``tool_use`` entry,
        not silently fall through and crash on ``Undefined``."""
        chat, tokenizer = _build_serving_chat_with_mock_tokenizer(
            {"tool_use": _MINIMAX_LIKE_TEMPLATE, "default": _LLAMA_LIKE_TEMPLATE}
        )
        req = _build_request_with_one_tool()

        chat._process_messages(req, is_multimodal=False)

        passed = tokenizer.apply_chat_template.call_args.kwargs["tools"]
        expected = [tool.model_dump() for tool in req.tools]
        self.assertEqual(passed, expected)


class TestPerRequestTemplateOverride(unittest.TestCase):
    """``request.chat_template_kwargs={"chat_template": ...}`` overrides the
    tokenizer's default. The precomputed shape must not be used in that case.
    """

    def test_override_flips_flat_tokenizer_to_wrapped_shape(self):
        # Tokenizer's default is flat → precomputed wrapped=False.
        chat, tokenizer = _build_serving_chat_with_mock_tokenizer(_LLAMA_LIKE_TEMPLATE)
        self.assertFalse(chat._tools_use_wrapped_shape)

        req = _build_request_with_one_tool()
        req.chat_template_kwargs = {"chat_template": _MINIMAX_LIKE_TEMPLATE}

        chat._process_messages(req, is_multimodal=False)

        passed = tokenizer.apply_chat_template.call_args.kwargs["tools"]
        expected = [tool.model_dump() for tool in req.tools]
        self.assertEqual(passed, expected)

    def test_override_flips_wrapped_tokenizer_to_flat_shape(self):
        # Tokenizer's default is wrapped → precomputed wrapped=True.
        chat, tokenizer = _build_serving_chat_with_mock_tokenizer(
            _MINIMAX_LIKE_TEMPLATE
        )
        self.assertTrue(chat._tools_use_wrapped_shape)

        req = _build_request_with_one_tool()
        req.chat_template_kwargs = {"chat_template": _LLAMA_LIKE_TEMPLATE}

        chat._process_messages(req, is_multimodal=False)

        passed = tokenizer.apply_chat_template.call_args.kwargs["tools"]
        expected = [tool.function.model_dump() for tool in req.tools]
        self.assertEqual(passed, expected)

    def test_named_override_against_dict_tokenizer_dispatches_correctly(self):
        # Tokenizer is a dict whose ``default`` is flat (so __init__ picks
        # ``tool_use``... but the override below names ``default`` explicitly,
        # which IS flat — so dispatch must flip back to flat regardless of
        # the precomputed value).
        chat, tokenizer = _build_serving_chat_with_mock_tokenizer(
            {"tool_use": _MINIMAX_LIKE_TEMPLATE, "default": _LLAMA_LIKE_TEMPLATE}
        )
        # __init__ scored ``tool_use`` (the tools-path entry) → wrapped.
        self.assertTrue(chat._tools_use_wrapped_shape)

        req = _build_request_with_one_tool()
        req.chat_template_kwargs = {"chat_template": "default"}  # NAME, not body

        chat._process_messages(req, is_multimodal=False)

        # The named override resolves to the flat ``default`` template, so
        # the dispatch must use the flat tool shape.
        passed = tokenizer.apply_chat_template.call_args.kwargs["tools"]
        expected = [tool.function.model_dump() for tool in req.tools]
        self.assertEqual(passed, expected)


class TestServingChatInitPrecomputesShape(unittest.TestCase):
    """The wrapped/flat decision is computed once in ``__init__``."""

    def test_init_sets_tools_use_wrapped_shape(self):
        wrapped_chat, _ = _build_serving_chat_with_mock_tokenizer(
            _MINIMAX_LIKE_TEMPLATE
        )
        self.assertTrue(wrapped_chat._tools_use_wrapped_shape)

        flat_chat, _ = _build_serving_chat_with_mock_tokenizer(_LLAMA_LIKE_TEMPLATE)
        self.assertFalse(flat_chat._tools_use_wrapped_shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
