"""Validated request planning for deterministic CMSSDT chat capabilities.

The planner extracts stable entities with deterministic parsing and chooses only
registered capabilities. It never runs graph queries or accepts executable
instructions from an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Iterable


IB_TAG_RE = re.compile(r"\b(CMSSW_[A-Za-z0-9_]+_\d{4}-\d{2}-\d{2}-\d{4})\b")
ARCH_RE = re.compile(
    r"\b(el\d+|fc\d+)[_\s-]+(amd64|aarch64|riscv64)[_\s-]+(gcc\d+)\b", re.IGNORECASE
)
PR_RE = re.compile(r"(?:\b(?:pr|mr|pull request|merge request)\s*#?|#)(\d+)\b", re.IGNORECASE)
WORKFLOW_RE = re.compile(r"\bworkflow\s*#?(\d+(?:\.\d+)?)\b", re.IGNORECASE)
SHIFT_SUMMARY_RE = re.compile(
    r"\bshift\s*summary\b|\bshift\s*report\b|\bvirtual\s*shift(?:er)?\b|"
    r"\bwhat\s+changed\s+(?:since|between)\b|\b(?:since|from)\s+the\s+(?:last|previous)\s+(?:ib|build)\b|"
    # "what are the new failures/errors" is the same request as "what
    # changed since the last IB" -- "new" only means something relative to
    # a previous build, so this is inherently a previous-vs-latest
    # comparison request, not a plain failure-count question. Also covers
    # "new <noun> [are/is] failing" (e.g. "what new workflows are failing")
    # -- found live: the exact-noun form above didn't cover this shape, so
    # it fell through to the raw list of *every* currently-failing workflow
    # instead of the previous-vs-latest delta the question actually asked
    # for ("only those absent in the previous release"). A bounded gap
    # (rather than requiring "new" and the failure noun to be adjacent) also
    # covers a category noun in between -- "new relval failures", "new unit
    # test failures", "new build warnings" -- found live: naming two IBs and
    # asking for one category's new failures ("what are the new relval
    # failures between <A> and <B>") fell through to the two-IB
    # comparison-or-single-build clarification instead of answering, because
    # the old exact-adjacency pattern only matched "new failures" with
    # nothing in between.
    r"\bnew\b[^?!]{0,20}\b(?:failures?|errors?|warnings?|issues?)\b|"
    r"\bnew\b[^?!]{0,30}\bfailing\b|"
    # Reversed order: "relval failures are new [between <A> and <B>]" --
    # found live alongside the gap above, same underlying question, just
    # phrased with the category noun and "new" swapped.
    r"\b(?:failures?|errors?|warnings?|issues?)\b[^?!]{0,10}\b(?:is|are)\s+new\b",
    re.IGNORECASE,
)


class Scope(StrEnum):
    DEFAULT = "default"
    ALL = "all"
    EXPLICIT_ARCHITECTURE = "explicit_architecture"


class PlanKind(StrEnum):
    DETERMINISTIC = "deterministic"
    CLARIFICATION = "clarification"
    LLM = "llm"


@dataclass(frozen=True)
class RequestContext:
    """Stable facts extracted from a user message before routing."""

    question: str
    ib_tags: tuple[str, ...] = ()
    architecture: str | None = None
    pr_number: int | None = None
    workflow_id: str | None = None
    scope: Scope = Scope.DEFAULT
    categories: frozenset[str] = frozenset()
    has_summary_request: bool = False
    has_failure_request: bool = False
    has_comparison_request: bool = False
    has_relationship_request: bool = False
    has_shift_summary_request: bool = False
    is_bare_ib_tag: bool = False

    @property
    def tag(self) -> str | None:
        return self.ib_tags[0] if len(self.ib_tags) == 1 else None

    @classmethod
    def extract(cls, question: str) -> "RequestContext":
        tags = tuple(dict.fromkeys(IB_TAG_RE.findall(question)))
        architecture_match = ARCH_RE.search(question)
        architecture = "_".join(architecture_match.groups()).lower() if architecture_match else None
        lowered = question.lower()
        categories: set[str] = set()
        if re.search(r"\brel[\s-]*vals?\b", lowered) or re.search(r"\b(?:wfs?|work\s*flows?)\b", lowered):
            categories.add("relval")
        if re.search(r"\bunit(?:[\s-]*tests?)?\b", lowered) or re.search(r"\bpackages?\b", lowered):
            categories.add("unittest")
        if re.search(r"\bclang\b", lowered):
            categories.add("clang")
        if re.search(r"\b(?:build|warnings?)\b", lowered):
            categories.add("build")
        if re.search(r"\b(?:pr|mr|pull\s+request|merge\s+request|issues?)\b", lowered):
            categories.add("pr_evidence")
        scope = Scope.EXPLICIT_ARCHITECTURE if architecture else (
            Scope.ALL if re.search(r"\b(?:all|everything)\b", lowered) else Scope.DEFAULT
        )
        pr_match = PR_RE.search(question)
        workflow_match = WORKFLOW_RE.search(question)
        bare_tag = bool(re.fullmatch(r"\s*CMSSW_[A-Za-z0-9_]+_\d{4}-\d{2}-\d{2}-\d{4}\s*[.?!]?\s*", question))
        return cls(
            question=question,
            ib_tags=tags,
            architecture=architecture,
            pr_number=int(pr_match.group(1)) if pr_match else None,
            workflow_id=workflow_match.group(1) if workflow_match else None,
            scope=scope,
            categories=frozenset(categories),
            has_summary_request=bool(re.search(r"\b(?:summary|summarize|results?|show|provide)\b", lowered)),
            has_failure_request=bool(re.search(r"\b(?:fail(?:ing|ed|ures?)|errors?)\b", lowered)),
            has_comparison_request=bool(re.search(r"\b(?:compare|comparison|delta|difference)\b", lowered)),
            has_relationship_request=bool(re.search(r"\b(?:why|parent|sub[\s-]*ib|variant)\b", lowered)),
            has_shift_summary_request=bool(SHIFT_SUMMARY_RE.search(question)),
            is_bare_ib_tag=bare_tag,
        )


@dataclass(frozen=True)
class Capability:
    """Declarative description of one approved deterministic answer type."""

    name: str
    required_categories: frozenset[str] = frozenset()
    required_tag_count: int | None = None
    requires_summary: bool = False
    requires_failures: bool = False
    requires_comparison: bool = False
    requires_relationship: bool = False
    requires_shift_summary: bool = False
    bare_tag_only: bool = False
    llm_only: bool = False
    priority: int = 0

    def score(self, context: RequestContext) -> int | None:
        if self.llm_only:
            return None
        if self.required_categories and not self.required_categories.issubset(context.categories):
            return None
        if self.required_tag_count is not None and len(context.ib_tags) != self.required_tag_count:
            return None
        if self.requires_summary and not context.has_summary_request:
            return None
        if self.requires_failures and not context.has_failure_request:
            return None
        if self.requires_comparison and not context.has_comparison_request:
            return None
        if self.requires_relationship and not context.has_relationship_request:
            return None
        if self.requires_shift_summary and not context.has_shift_summary_request:
            return None
        if self.bare_tag_only and not context.is_bare_ib_tag:
            return None
        score = self.priority
        score += 20 * len(self.required_categories)
        score += 12 if self.required_tag_count is not None else 0
        score += 8 if self.requires_summary else 0
        score += 8 if self.requires_failures else 0
        score += 8 if self.requires_comparison else 0
        score += 8 if self.requires_relationship else 0
        score += 8 if self.requires_shift_summary else 0
        score += 16 if self.bare_tag_only else 0
        score += 4 if context.architecture else 0
        return score


@dataclass(frozen=True)
class Clarification:
    missing_field: str
    question: str
    context: dict[str, str]
    choices: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Plan:
    kind: PlanKind
    capability: str | None = None
    confidence: float = 0.0
    clarification: Clarification | None = None
    context: RequestContext | None = None


@dataclass(frozen=True)
class LLMPlan:
    """A validated, non-executable LLM proposal for an open-ended investigation."""

    capability: str
    tool_calls: tuple[tuple[str, dict], ...]
    answer_constraints: tuple[str, ...]


def validate_llm_plan(raw: object, capabilities: Iterable[Capability], allowed_tools: set[str]) -> LLMPlan | None:
    """Accept only a schema-valid proposal that references registered capabilities/tools."""
    if not isinstance(raw, dict):
        return None
    capability_names = {capability.name for capability in capabilities}
    capability = raw.get("capability")
    tool_calls = raw.get("tool_calls")
    constraints = raw.get("answer_constraints")
    if capability not in capability_names or not isinstance(tool_calls, list) or not isinstance(constraints, list):
        return None
    validated_calls: list[tuple[str, dict]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            return None
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        if name not in allowed_tools or not isinstance(arguments, dict):
            return None
        validated_calls.append((name, arguments))
    if not all(isinstance(item, str) for item in constraints):
        return None
    return LLMPlan(capability, tuple(validated_calls), tuple(constraints))


@dataclass
class RequestPlanner:
    """Select capabilities by declared evidence, never source-code ordering."""

    capabilities: tuple[Capability, ...]
    pending: dict[str, Clarification] = field(default_factory=dict)

    def plan(self, context: RequestContext, session_id: str | None = None) -> Plan:
        resolved = self._resolve_pending(context, session_id)
        if resolved is not None:
            context = resolved
        clarification = self._clarification(context)
        if clarification is not None:
            if session_id:
                self.pending[session_id] = clarification
            return Plan(PlanKind.CLARIFICATION, confidence=1.0, clarification=clarification, context=context)
        candidates = [
            (score, capability.name)
            for capability in self.capabilities
            if (score := capability.score(context)) is not None
        ]
        if not candidates:
            return Plan(PlanKind.LLM, confidence=0.0, context=context)
        score, capability = max(candidates, key=lambda item: (item[0], item[1]))
        return Plan(
            PlanKind.DETERMINISTIC,
            capability=capability,
            confidence=min(score / 100, 0.99),
            context=context,
        )

    def _resolve_pending(self, context: RequestContext, session_id: str | None) -> RequestContext | None:
        if not session_id or session_id not in self.pending:
            return None
        pending = self.pending[session_id]
        if pending.missing_field != "category" or not context.categories:
            return None
        self.pending.pop(session_id, None)
        return RequestContext.extract(f"{context.question} {pending.context['tag']}")

    @staticmethod
    def _clarification(context: RequestContext) -> Clarification | None:
        if (
            context.tag and context.has_failure_request
            and not context.categories and not context.has_shift_summary_request
        ):
            return Clarification(
                missing_field="category",
                question="Do you want build warnings, failed RelVals, unit-test results, or a combined IB summary?",
                context={"tag": context.tag},
                choices=(
                    ("Build warnings", f"build warnings for {context.tag}"),
                    ("RelVal failures", f"failed RelVals for {context.tag}"),
                    ("Unit-test results", f"unit tests for {context.tag}"),
                    ("Combined summary", f"summary for {context.tag}"),
                ),
            )
        if (
            len(context.ib_tags) > 1 and not context.has_comparison_request
            and not context.has_shift_summary_request
        ):
            # Mirrors the single-tag guard above: "what's new failing between
            # <A> and <B>" is already an unambiguous shift-summary request
            # (has_shift_summary_request), just like "shift summary for <tag>"
            # is unambiguous for the single-tag case -- neither needs the
            # generic "comparison or single build?" clarification.
            return Clarification(
                missing_field="intent",
                question="Do you want a comparison of these Integration Builds, or results for one specific build?",
                context={"tags": ", ".join(context.ib_tags)},
            )
        return None
