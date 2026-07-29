"""Frozen, natural-language surfaces for the hybrid user simulator.

The response model remains responsible for selecting an option.  This module
only renders that already-selected option as a two-turn conversation.  It
never receives latent preferences, susceptibilities, utilities, or numeric
feature vectors.

Conversation templates are generated outside the experiment (for example by
an LLM through OpenRouter), reviewed, and stored in a small JSON bank.  Runtime
rendering is therefore offline and deterministic: every evaluated updater sees
the same wording for the same visible context and selected option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import json
import re

from .schemas import InteractionContext, PolicyProvenance


SCHEMA_VERSION = 1
PRESENTATIONS = (
    "balanced",
    "restricted",
    "default",
    "suggested",
    "ranking",
)

_CORE_PRESENTATION_FIELDS = frozenset(
    {
        "prompt",
        "option_1_name",
        "option_1_description",
        "option_2_name",
        "option_2_description",
    }
)
_OPTIONAL_PRESENTATION_FIELDS = frozenset(
    {"default_name", "suggested_name"}
)
_CHOICE_FIELDS = frozenset({"selected_name"})
_MECHANISM_TO_PRESENTATION = {
    "balanced": "balanced",
    "restriction": "restricted",
    "restricted": "restricted",
    "default": "default",
    "suggestion": "suggested",
    "suggested": "suggested",
    "ranking": "ranking",
    # These mechanisms select a question but do not add a visible default,
    # suggestion, or restriction to the resulting choice surface.
    "none": "balanced",
    "target_selection": "balanced",
}
_GENERAL_PREFERENCE = re.compile(
    r"\b(?:i|we)\s+(?:generally\s+|always\s+|usually\s+|normally\s+)?"
    r"(?:prefer|like|love|hate|want)\b",
    re.IGNORECASE,
)
_EXPLANATION = re.compile(
    r"\b(?:because|since|the reason|which is why|so that)\b",
    re.IGNORECASE,
)
_LOCAL_CHOICE = re.compile(
    r"\b(?:choose|select|take|use|keep|pick|go\s+with|works?\s+for\s+me)\b",
    re.IGNORECASE,
)
_CLAUSE_JOINER = re.compile(r"(?:[;:]|\s+(?:and|but|although|while)\s+)", re.IGNORECASE)
_LOCAL_CHOICE_SHAPE = re.compile(
    r"^(?:"
    r"i(?:(?:'ll)|(?:\s+(?:would|will)))?\s+"
    r"(?:choose|select|pick|take|use)\s+\[selected option\]"
    r"|i(?:(?:'ll)|(?:\s+(?:would|will)))?\s+"
    r"go\s+with\s+\[selected option\]"
    r"|let(?:'s|\s+us)\s+(?:choose|select|pick|take|use)\s+"
    r"\[selected option\]"
    r"|(?:please\s+)?(?:choose|select|pick|take|use|keep)\s+"
    r"\[selected option\]"
    r"|\[selected option\]\s+works?\s+for\s+me"
    r"|\[selected option\]\s*,?\s+please"
    r")\s*[.!]?$",
    re.IGNORECASE,
)


def _contains_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _text(
    value: Any,
    name: str,
    *,
    maximum: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    if _contains_control(value):
        raise ValueError(f"{name} cannot contain control characters")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> None:
    allowed = expected | (optional or set())
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - allowed)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError(f"{name} fields must be exact ({'; '.join(details)})")


def _freeze_string_mapping(
    value: Mapping[str, str],
    name: str,
    *,
    key_maximum: int = 300,
    value_maximum: int = 300,
) -> Mapping[str, str]:
    frozen: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, f"{name} key", maximum=key_maximum)
        item = _text(raw_value, f"{name}[{key!r}]", maximum=value_maximum)
        frozen[key] = item
    return MappingProxyType(frozen)


def _template_fields(template: str, name: str) -> tuple[str, ...]:
    fields: list[str] = []
    try:
        parsed = tuple(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"{name} has invalid placeholder syntax: {exc}") from exc
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not field_name:
            raise ValueError(f"{name} cannot contain positional placeholders")
        if format_spec or conversion:
            raise ValueError(
                f"{name} placeholders cannot use conversions or format specs"
            )
        fields.append(field_name)
    return tuple(fields)


def _validate_presentation_template(
    presentation: str,
    template: str,
) -> str:
    name = f"presentation_templates[{presentation!r}]"
    value = _text(template, name, maximum=2_000)
    fields = _template_fields(value, name)
    observed = set(fields)
    allowed = _CORE_PRESENTATION_FIELDS | _OPTIONAL_PRESENTATION_FIELDS
    unknown = sorted(observed - allowed)
    if unknown:
        raise ValueError(
            f"{name} contains unknown placeholders: {', '.join(unknown)}"
        )
    missing = sorted(_CORE_PRESENTATION_FIELDS - observed)
    if missing:
        raise ValueError(
            f"{name} must preserve placeholders: {', '.join(missing)}"
        )
    if presentation == "default":
        if "default_name" not in observed:
            raise ValueError(f"{name} must preserve {{default_name}}")
        if "suggested_name" in observed:
            raise ValueError(f"{name} cannot contain {{suggested_name}}")
    elif presentation == "suggested":
        if "suggested_name" not in observed:
            raise ValueError(f"{name} must preserve {{suggested_name}}")
        if "default_name" in observed:
            raise ValueError(f"{name} cannot contain {{default_name}}")
    elif observed & _OPTIONAL_PRESENTATION_FIELDS:
        raise ValueError(
            f"{name} cannot contain default or suggested placeholders"
        )
    return value


def _validate_choice_template(template: str) -> str:
    value = _text(template, "choice_template", maximum=160)
    fields = _template_fields(value, "choice_template")
    observed = set(fields)
    unknown = sorted(observed - _CHOICE_FIELDS)
    if unknown:
        raise ValueError(
            "choice_template contains unknown placeholders: "
            + ", ".join(unknown)
        )
    if observed != _CHOICE_FIELDS:
        raise ValueError("choice_template must preserve {selected_name}")
    probe = value.format(selected_name="Option A")
    _validate_local_choice(probe, selected_name="Option A")
    return value


def _validate_local_choice(value: str, *, selected_name: str) -> None:
    _text(value, "rendered user message", maximum=160)
    if selected_name.casefold() not in value.casefold():
        raise ValueError("rendered user message does not identify the selected option")
    semantic_text = re.sub(
        re.escape(selected_name),
        "[selected option]",
        value,
        flags=re.IGNORECASE,
    )
    if _GENERAL_PREFERENCE.search(semantic_text):
        raise ValueError(
            "rendered user message invents a persistent preference claim"
        )
    if _EXPLANATION.search(semantic_text):
        raise ValueError("rendered user message invents a reason for the choice")
    if _CLAUSE_JOINER.search(semantic_text):
        raise ValueError("rendered user message must contain only one local choice")
    if not _LOCAL_CHOICE.search(semantic_text):
        raise ValueError(
            "rendered user message must explicitly make a local choice"
        )
    if _LOCAL_CHOICE_SHAPE.fullmatch(semantic_text.strip()) is None:
        raise ValueError(
            "rendered user message must contain only the selected local choice"
        )


def _strict_json(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} must be UTF-8: {exc}") from exc

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    return _mapping(parsed, name)


@dataclass(frozen=True, slots=True)
class RenderedConversation:
    """One natural assistant turn followed by one constrained user choice."""

    surface_id: str
    assistant_message: str
    user_message: str
    display_names: Mapping[str, str]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface_id",
            _text(self.surface_id, "surface_id", maximum=1_000),
        )
        object.__setattr__(
            self,
            "assistant_message",
            _text(
                self.assistant_message,
                "assistant_message",
                maximum=4_000,
            ),
        )
        object.__setattr__(
            self,
            "user_message",
            _text(self.user_message, "user_message", maximum=160),
        )
        names = _freeze_string_mapping(
            _mapping(self.display_names, "display_names"),  # type: ignore[arg-type]
            "display_names",
        )
        if len(names) != 2:
            raise ValueError(
                "a rendered conversation must name exactly two displayed options"
            )
        if len(set(names.values())) != len(names):
            raise ValueError("rendered display names must be distinct")
        object.__setattr__(self, "display_names", names)
        object.__setattr__(
            self,
            "source",
            _text(self.source, "source", maximum=500),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "assistant_message": self.assistant_message,
            "user_message": self.user_message,
            "display_names": dict(self.display_names),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ScenarioConversationTemplate:
    """One reviewed LLM-authored conversation template for one scenario."""

    scenario_id: str
    display_names: Mapping[str, str]
    presentation_templates: Mapping[str, str]
    choice_template: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_id",
            _text(self.scenario_id, "scenario_id", maximum=300),
        )
        names = _freeze_string_mapping(
            _mapping(self.display_names, "display_names"),  # type: ignore[arg-type]
            "display_names",
            value_maximum=100,
        )
        if len(names) != 4:
            raise ValueError(
                "scenario display_names must cover exactly four catalog options"
            )
        normalized_names = {
            " ".join(name.casefold().split()) for name in names.values()
        }
        if len(normalized_names) != len(names):
            raise ValueError("scenario display names must be distinct")
        object.__setattr__(self, "display_names", names)

        raw_presentations = _mapping(
            self.presentation_templates,
            "presentation_templates",
        )
        if set(raw_presentations) != set(PRESENTATIONS):
            missing = sorted(set(PRESENTATIONS) - set(raw_presentations))
            unknown = sorted(set(raw_presentations) - set(PRESENTATIONS))
            details: list[str] = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unknown:
                details.append("unknown=" + ",".join(unknown))
            raise ValueError(
                "presentation_templates must contain exactly "
                f"{PRESENTATIONS} ({'; '.join(details)})"
            )
        prepared = {
            presentation: _validate_presentation_template(
                presentation,
                raw_presentations[presentation],
            )
            for presentation in PRESENTATIONS
        }
        object.__setattr__(
            self,
            "presentation_templates",
            MappingProxyType(prepared),
        )
        object.__setattr__(
            self,
            "choice_template",
            _validate_choice_template(self.choice_template),
        )
        object.__setattr__(
            self,
            "source",
            _text(self.source, "source", maximum=500),
        )

    def _presentation(
        self,
        context: InteractionContext,
        provenance: PolicyProvenance,
    ) -> str:
        try:
            presentation = _MECHANISM_TO_PRESENTATION[
                provenance.presentation_mechanism
            ]
        except KeyError as exc:
            raise ValueError(
                "unsupported presentation mechanism for conversation rendering: "
                f"{provenance.presentation_mechanism!r}"
            ) from exc

        if context.default_option_id is not None:
            if presentation != "default":
                raise ValueError(
                    "visible default and provenance mechanism disagree"
                )
        elif presentation == "default":
            raise ValueError(
                "default provenance requires a visible default option"
            )
        if context.suggested_option_id is not None:
            if presentation != "suggested":
                raise ValueError(
                    "visible suggestion and provenance mechanism disagree"
                )
        elif presentation == "suggested":
            raise ValueError(
                "suggestion provenance requires a visible suggested option"
            )
        if (
            presentation not in {"default", "suggested"}
            and (
                context.default_option_id is not None
                or context.suggested_option_id is not None
            )
        ):
            raise ValueError(
                "non-treatment conversation contains a default or suggestion"
            )
        return presentation

    def render(
        self,
        context: InteractionContext,
        provenance: PolicyProvenance,
        selected_option_id: str,
    ) -> RenderedConversation:
        """Render a context after a mathematical simulator selects an option."""

        if not isinstance(context, InteractionContext):
            raise TypeError("context must be an InteractionContext")
        if not isinstance(provenance, PolicyProvenance):
            raise TypeError("provenance must be a PolicyProvenance")
        selected = _text(
            selected_option_id,
            "selected_option_id",
            maximum=300,
        )
        if context.scenario_id != self.scenario_id:
            raise ValueError(
                "conversation template scenario does not match the context"
            )
        if len(context.options) != 2:
            raise ValueError(
                "conversation rendering currently requires exactly two options"
            )
        if selected not in context.option_ids:
            raise ValueError("selected option is not displayed")
        if context.prompt is None:
            raise ValueError(
                "catalog-backed conversation context requires a natural prompt"
            )

        ordered_options = tuple(
            context.option(option_id) for option_id in context.ranking
        )
        selected_name = self.display_names.get(selected)
        if selected_name is None:
            raise ValueError(
                "conversation template does not name the selected option"
            )
        try:
            ordered_names = tuple(
                self.display_names[option.option_id]
                for option in ordered_options
            )
        except KeyError as exc:
            raise ValueError(
                "conversation template does not name every displayed option"
            ) from exc
        descriptions = tuple(
            _text(
                option.label,
                f"description for {option.option_id}",
                maximum=500,
            )
            for option in ordered_options
        )
        presentation = self._presentation(context, provenance)
        values = {
            "prompt": context.prompt,
            "option_1_name": ordered_names[0],
            "option_1_description": descriptions[0],
            "option_2_name": ordered_names[1],
            "option_2_description": descriptions[1],
            "default_name": (
                ""
                if context.default_option_id is None
                else self.display_names[context.default_option_id]
            ),
            "suggested_name": (
                ""
                if context.suggested_option_id is None
                else self.display_names[context.suggested_option_id]
            ),
        }
        assistant = self.presentation_templates[presentation].format_map(values)
        assistant = _text(
            assistant,
            "rendered assistant message",
            maximum=4_000,
        )
        user = self.choice_template.format(selected_name=selected_name)
        _validate_local_choice(user, selected_name=selected_name)
        displayed_names = {
            option.option_id: name
            for option, name in zip(ordered_options, ordered_names)
        }
        surface_id = (
            f"{self.scenario_id}:{presentation}:"
            f"{context.ranking[0]}>{context.ranking[1]}:{selected}"
        )
        return RenderedConversation(
            surface_id=surface_id,
            assistant_message=assistant,
            user_message=user,
            display_names=displayed_names,
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "display_names": dict(self.display_names),
            "presentation_templates": dict(self.presentation_templates),
            "choice_template": self.choice_template,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ConversationTemplateBank:
    """Complete frozen collection of one conversation template per scenario."""

    bank_id: str
    templates: tuple[ScenarioConversationTemplate, ...]
    source: str
    schema_version: int = SCHEMA_VERSION
    _by_scenario: Mapping[str, ScenarioConversationTemplate] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"conversation bank schema_version must be {SCHEMA_VERSION}"
            )
        object.__setattr__(
            self,
            "bank_id",
            _text(self.bank_id, "bank_id", maximum=300),
        )
        object.__setattr__(
            self,
            "source",
            _text(self.source, "source", maximum=500),
        )
        templates = tuple(self.templates)
        if not templates:
            raise ValueError("conversation template bank cannot be empty")
        if not all(
            isinstance(template, ScenarioConversationTemplate)
            for template in templates
        ):
            raise TypeError(
                "templates must contain ScenarioConversationTemplate objects"
            )
        by_scenario = {
            template.scenario_id: template for template in templates
        }
        if len(by_scenario) != len(templates):
            raise ValueError(
                "conversation template bank contains duplicate scenario IDs"
            )
        object.__setattr__(
            self,
            "templates",
            tuple(sorted(templates, key=lambda item: item.scenario_id)),
        )
        object.__setattr__(
            self,
            "_by_scenario",
            MappingProxyType(by_scenario),
        )

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(template.scenario_id for template in self.templates)

    def template(self, scenario_id: str) -> ScenarioConversationTemplate:
        try:
            return self._by_scenario[scenario_id]
        except KeyError as exc:
            raise KeyError(
                f"no conversation template for scenario {scenario_id!r}"
            ) from exc

    def render(
        self,
        context: InteractionContext,
        provenance: PolicyProvenance,
        selected_option_id: str,
    ) -> RenderedConversation:
        """Select the scenario template and render one two-turn conversation."""

        return self.template(context.scenario_id).render(
            context,
            provenance,
            selected_option_id,
        )

    def validate_catalog(self, catalog: Any) -> None:
        """Require exact scenario and four-option coverage of a scenario catalog."""

        scenarios = getattr(catalog, "scenarios", None)
        if not isinstance(scenarios, tuple) and not isinstance(scenarios, list):
            raise TypeError("catalog.scenarios must be a sequence")
        scenario_by_id: dict[str, Any] = {}
        for scenario in scenarios:
            scenario_id = getattr(scenario, "scenario_id", None)
            if not isinstance(scenario_id, str) or not scenario_id:
                raise TypeError("catalog scenarios must expose scenario_id")
            if scenario_id in scenario_by_id:
                raise ValueError(
                    f"catalog contains duplicate scenario ID {scenario_id!r}"
                )
            scenario_by_id[scenario_id] = scenario
        missing = sorted(set(scenario_by_id) - set(self._by_scenario))
        unknown = sorted(set(self._by_scenario) - set(scenario_by_id))
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unknown:
                details.append("unknown=" + ",".join(unknown))
            raise ValueError(
                "conversation bank scenario coverage is not exact "
                f"({'; '.join(details)})"
            )
        for scenario_id, scenario in scenario_by_id.items():
            options = getattr(scenario, "options", None)
            if not isinstance(options, tuple) and not isinstance(options, list):
                raise TypeError(
                    f"catalog scenario {scenario_id!r} must expose options"
                )
            option_ids = {
                getattr(option, "option_id", None) for option in options
            }
            if None in option_ids or not all(
                isinstance(option_id, str) and option_id
                for option_id in option_ids
            ):
                raise TypeError(
                    f"catalog scenario {scenario_id!r} has invalid option IDs"
                )
            template_ids = set(
                self._by_scenario[scenario_id].display_names
            )
            if template_ids != option_ids:
                missing_options = sorted(option_ids - template_ids)
                unknown_options = sorted(template_ids - option_ids)
                details = []
                if missing_options:
                    details.append(
                        "missing=" + ",".join(missing_options)
                    )
                if unknown_options:
                    details.append(
                        "unknown=" + ",".join(unknown_options)
                    )
                raise ValueError(
                    f"conversation template {scenario_id!r} option coverage "
                    f"is not exact ({'; '.join(details)})"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bank_id": self.bank_id,
            "source": self.source,
            "templates": [
                template.to_dict() for template in self.templates
            ],
        }


def load_conversation_bank(path: str | Path) -> ConversationTemplateBank:
    """Load and strictly validate a frozen JSON conversation template bank."""

    source_path = Path(path)
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read conversation template bank {source_path}: {exc}"
        ) from exc
    payload = _strict_json(
        raw,
        name=f"conversation template bank {source_path}",
    )
    _exact_keys(
        payload,
        {"schema_version", "bank_id", "source", "templates"},
        "conversation template bank",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"conversation bank schema_version must be {SCHEMA_VERSION}"
        )
    bank_source = _text(payload["source"], "source", maximum=500)
    raw_templates = payload["templates"]
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("templates must be a non-empty array")
    templates: list[ScenarioConversationTemplate] = []
    for index, raw_template in enumerate(raw_templates):
        name = f"templates[{index}]"
        item = _mapping(raw_template, name)
        _exact_keys(
            item,
            {
                "scenario_id",
                "display_names",
                "presentation_templates",
                "choice_template",
            },
            name,
            optional={"source"},
        )
        display_names = _mapping(
            item["display_names"],
            f"{name}.display_names",
        )
        presentations = _mapping(
            item["presentation_templates"],
            f"{name}.presentation_templates",
        )
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in display_names.items()
        ):
            raise TypeError(f"{name}.display_names must map strings to strings")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in presentations.items()
        ):
            raise TypeError(
                f"{name}.presentation_templates must map strings to strings"
            )
        templates.append(
            ScenarioConversationTemplate(
                scenario_id=item["scenario_id"],
                display_names=display_names,  # type: ignore[arg-type]
                presentation_templates=presentations,  # type: ignore[arg-type]
                choice_template=item["choice_template"],
                source=item.get("source", bank_source),
            )
        )
    return ConversationTemplateBank(
        bank_id=payload["bank_id"],
        templates=tuple(templates),
        source=bank_source,
        schema_version=payload["schema_version"],
    )


__all__ = [
    "ConversationTemplateBank",
    "PRESENTATIONS",
    "RenderedConversation",
    "ScenarioConversationTemplate",
    "load_conversation_bank",
]
