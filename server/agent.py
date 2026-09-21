"""Agent loop: question in, grounded answer out, via Neo4j graph tools.

No MCP subprocess — tools are plain Python functions called directly.
Same <tool_call> text-fallback parsing for models that don't emit
structured tool_calls (confirmed needed for hf-qwen3-32b-awq).
"""
import inspect
import json
import re
from typing import Callable

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def propose_investigation_plan(client, model: str, question: str, capabilities: list[str], tool_names: list[str]) -> dict | None:
    """Ask for a non-executable JSON investigation proposal with no tool access."""
    prompt = (
        "Return exactly one JSON object with capability, tool_calls, and answer_constraints. "
        f"capability must be one of: {', '.join(capabilities)}. "
        "tool_calls must be a JSON list of {name, arguments} objects, where name is one of: "
        f"{', '.join(tool_names)}. Do not answer the question or issue any tool call. "
        "answer_constraints must include local_only and evidence_only."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(match.group(0) if match else content)
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def propose_capability_classification(client, model: str, question: str, capabilities: list[str]) -> dict | None:
    """Ask the LLM to independently pick which registered capability (if any)
    a question needs, for shadow-mode comparison against the deterministic
    regex planner. Logging only — this result is never used to answer a
    question, only to evaluate whether an LLM router would agree with the
    existing one before ever promoting it to primary (Phase 3's gate)."""
    prompt = (
        "Return exactly one JSON object with fields capability and confidence. "
        f"capability must be one of: {', '.join(capabilities)}, or null if none fit. "
        "confidence is a number from 0 to 1. Do not answer the question."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(match.group(0) if match else content)
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def propose_pr_search_signature(client, model: str, question: str) -> str | None:
    """Ask the LLM to read the WHOLE question -- not score isolated words the way
    server.api_server._pr_search_terms's regex does -- and propose the single best
    search phrase for a PR/Issue evidence search. Pilot fallback for
    _pr_evidence_answer, used only when that regex-based extraction finds nothing
    usable: found live, 2026-09-21, that a pasted error message ("MIOpen Error:
    filesystem error: cannot create directories: Permission denied
    [/tmp/miopen-cache/3.5.2.]") got reduced to scoring "Permission" as an isolated
    capitalized word, when the actual searchable unit is the error message as a
    whole. This never answers the question itself and is never trusted as fact on
    its own -- the caller still runs the real, deterministic find_prs_for_error
    search on whatever this returns and composes the final answer only from its
    real graph results, same non-agentic constrained-JSON pattern as
    propose_capability_classification."""
    prompt = (
        "The user's question may include a pasted error message or log excerpt. "
        "Return exactly one JSON object with field error_signature: the single most "
        "distinctive, searchable phrase from the question (an exact error message, "
        "exception text, module/class name, or file path) that a human GitHub PR or "
        "Issue comment would plausibly quote verbatim -- or null if no such phrase is "
        "present. Never return a generic English word or a sentence fragment made of "
        "ordinary words. Do not answer the question."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        decision = json.loads(match.group(0) if match else content)
        signature = decision.get("error_signature")
        return signature.strip() if isinstance(signature, str) and signature.strip() else None
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def execute_read_only_plan(tool_calls: tuple[tuple[str, dict], ...], tools_registry: dict) -> list[dict]:
    """Execute already-validated tools and retain each structured result as evidence."""
    evidence = []
    for name, arguments in tool_calls:
        tool = tools_registry.get(name)
        if tool is None:
            return []
        try:
            result = tool(**arguments)
        except Exception as exc:
            result = {"error": str(exc)}
        evidence.append({"tool": name, "arguments": arguments, "result": result})
    return evidence


def synthesize_evidence_answer(client, model: str, system_prompt: str, question: str, evidence: list[dict]) -> str | None:
    """Produce a final answer from fixed local evidence without granting tool access."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt + "\nAnswer only from the supplied local evidence."},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nLocal evidence:\n{json.dumps(evidence, default=str)}",
                },
            ],
            temperature=0,
        )
        answer = (response.choices[0].message.content or "").strip()
        return answer or None
    except (AttributeError, TypeError, ValueError):
        return None


def _extract_manual_tool_call(content: str | None) -> dict | None:
    if not content:
        return None
    m = TOOL_CALL_RE.search(content)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        try:
            tool_call, end = json.JSONDecoder().raw_decode(m.group(1))
        except json.JSONDecodeError:
            return None
        trailing = m.group(1)[end:].strip()
        return tool_call if isinstance(tool_call, dict) and trailing.strip("}") == "" else None


def _function_to_openai_schema(name: str, fn) -> dict:
    """Build an OpenAI function-calling tool schema from a Python function's
    signature and docstring."""
    sig = inspect.signature(fn)
    props = {}
    required = []
    for pname, param in sig.parameters.items():
        ptype = "string"
        annotation = param.annotation
        if annotation is int:
            ptype = "integer"
        elif annotation is float:
            ptype = "number"
        elif annotation is bool:
            ptype = "boolean"
        prop: dict = {"type": ptype}
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)
        # handle Optional (str | None)
        if "None" in str(annotation):
            prop["type"] = ptype  # keep the base type, just don't require it
            if pname in required:
                required.remove(pname)
        props[pname] = prop

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (fn.__doc__ or "").strip().split("\n")[0],
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


class KGChatSession:
    def __init__(self, openai_client, model: str, tools_registry: dict,
                 system_prompt: str,
                 history: list[dict] | None = None,
                 on_tool_call: Callable | None = None):
        self.client = openai_client
        self.model = model
        self.tools_registry = tools_registry
        self.openai_tools = [
            _function_to_openai_schema(name, fn)
            for name, fn in tools_registry.items()
        ]
        self.messages = [{"role": "system", "content": system_prompt}] + (history or [])
        self.uses_structured_tools = model != "gpt-5.6-luna"
        if not self.uses_structured_tools:
            tool_descriptions = "\n".join(
                f"- {name}{inspect.signature(fn)}: {(fn.__doc__ or '').strip().split(chr(10))[0]}"
                for name, fn in tools_registry.items()
            )
            self.messages[0]["content"] += (
                "\n\nNative function tools are unavailable for this model. "
                "To call a tool, respond only with "
                "<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>. "
                "Always call a tool before answering a data question, then use the returned "
                "result; do not invent data. For an unspecified 'most clang warnings' or "
                "'top warnings' question, call top_packages_by_warnings with {\"limit\": 10}; "
                "an IB tag is optional.\n\n"
                f"Available tools:\n{tool_descriptions}"
            )
        self.on_tool_call = on_tool_call

    def remember(self, question: str, answer: str) -> None:
        """Keep deterministic turns available for immediate LLM follow-ups."""
        self.messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])

    def _execute_tool(self, name: str, args: dict) -> str:
        fn = self.tools_registry.get(name)
        if not fn:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = fn(**args)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})

        for _ in range(8):
            try:
                request = {
                    "model": self.model,
                    "messages": self.messages,
                }
                if self.uses_structured_tools:
                    request["tools"] = self.openai_tools
                    # "none" was the original value here but the hosted gpt-oss-20b
                    # backend's Harmony response format rejects it outright
                    # ("ValueError : Harmony does not support reasoning_effort='none'",
                    # a real 500 from the litellm/vLLM gateway, not a graph-evidence
                    # decision) -- found live, 2026-09-21: every LLM tool-calling call
                    # this model made (feedback verification, the LLM-fallback chat
                    # path) silently failed and surfaced as "could not reach an
                    # evidence-based decision" / a generic API-error string, never as
                    # an obvious crash. Harmony's valid values are low/medium/high;
                    # "low" keeps closest to the original fast/deterministic intent
                    # (paired with temperature=0 below).
                    request["reasoning_effort"] = "low"
                    request["temperature"] = 0
                response = self.client.chat.completions.create(**request)
            except Exception as e:
                self.messages.append({
                    "role": "assistant",
                    "content": f"(API error: {e})",
                })
                return f"API error — try a narrower question. ({e})"

            msg = response.choices[0].message
            manual = None if msg.tool_calls else _extract_manual_tool_call(msg.content)

            if not msg.tool_calls and manual is None:
                content = (msg.content or "").strip()
                if not content:
                    self.messages.append({"role": "assistant", "content": ""})
                    self.messages.append({
                        "role": "user",
                        "content": "Your last response was empty. Call the relevant tool, then answer the original question.",
                    })
                    continue
                self.messages.append({"role": "assistant", "content": content})
                return content

            if manual is not None:
                name = manual.get("name", "")
                args = manual.get("arguments", {})
                result_text = self._execute_tool(name, args)
                if self.on_tool_call:
                    self.on_tool_call(name, args, result_text)
                self.messages.append({"role": "assistant", "content": msg.content})
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[Tool result for {name}]: {result_text}\n\n"
                        "Answer the original question using this result. "
                        "If you need another tool, emit another <tool_call> block."
                    ),
                })
                continue

            self.messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                content = self._execute_tool(tc.function.name, args)
                if self.on_tool_call:
                    self.on_tool_call(tc.function.name, args, content)
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": content}
                )

        return "(stopped after 8 tool calls)"
