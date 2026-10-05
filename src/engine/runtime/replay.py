"""Record and replay the structured decisions made by runtime agents.

The tape deliberately sits at the agent boundary: replay mode returns the same
typed objects the rest of the engine already consumes, without constructing a
fake pydantic-ai result or making a model request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from src.agents.character.tools import CharacterTool
from src.agents.dm.agent import DMResult
from src.agents.dm.director import DirectorResult
from src.agents.dm.tools import Create, Modify
from src.engine.state import WorldState


ReplayKind = Literal["character", "action_resolution", "dm", "director", "chronicle"]
_CHARACTER_ADAPTER = TypeAdapter(CharacterTool)


class ReplayError(RuntimeError):
    """The replay tape is exhausted, malformed, or does not match this run."""


@dataclass(frozen=True)
class _Entry:
    kind: ReplayKind
    key: str
    output: Any


class ReplayTape:
    """A sequential JSONL tape of typed agent outputs.

    Use :meth:`recording` for a live run and :meth:`playback` for an offline
    deterministic run. A tape is intentionally single-use and sequential so a
    scheduler change fails loudly instead of silently applying the wrong result.
    """

    def __init__(self, path: Path, mode: Literal["record", "replay"]):
        self.path = Path(path)
        self.mode = mode
        self._position = 0
        self._entries: list[_Entry] = []
        self._scenario: str | None = None
        self._character_id: str | None = None
        if mode == "record":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
        else:
            self._entries = self._load()

    def resolve_context(self, scenario: str | None, character_id: str | None) -> tuple[str | None, str | None]:
        """Supply recorded run identity during playback and reject conflicts."""
        if not self.is_playback:
            return scenario, character_id
        if self._scenario is None or self._character_id is None:
            raise ReplayError(f"Replay tape {self.path} has no run context metadata")
        if scenario is not None and scenario != self._scenario:
            raise ReplayError(f"Replay scenario is {self._scenario!r}, not requested {scenario!r}")
        if character_id is not None and character_id != self._character_id:
            raise ReplayError(f"Replay character is {self._character_id!r}, not requested {character_id!r}")
        return self._scenario, self._character_id

    def bind_context(self, scenario: str, character_id: str) -> None:
        """Write run identity before outputs, or validate it during playback."""
        if self.is_playback:
            self.resolve_context(scenario, character_id)
            return
        if self._scenario is not None:
            if (scenario, character_id) != (self._scenario, self._character_id):
                raise ReplayError("A recording tape cannot be rebound to a different run")
            return
        self._scenario, self._character_id = scenario, character_id
        entry = {"kind": "context", "scenario": scenario, "character_id": character_id}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @classmethod
    def recording(cls, path: str | Path) -> ReplayTape:
        return cls(Path(path), "record")

    @classmethod
    def playback(cls, path: str | Path) -> ReplayTape:
        return cls(Path(path), "replay")

    @property
    def is_playback(self) -> bool:
        return self.mode == "replay"

    @property
    def remaining(self) -> int:
        return len(self._entries) - self._position if self.is_playback else 0

    def character(self, actor_id: str, output: CharacterTool | None = None) -> CharacterTool | None:
        payload = self._exchange("character", actor_id, None if output is None else output.model_dump(mode="json"))
        return None if payload is None else _CHARACTER_ADAPTER.validate_python(payload)

    def dm(self, output: DMResult | None = None) -> DMResult:
        payload = self._exchange("dm", "dm", None if output is None else {
            "creates": [item.model_dump(mode="json") for item in output.creates],
            "modifies": [item.model_dump(mode="json") for item in output.modifies],
            "minutes": output.minutes,
        })
        return DMResult(
            creates=[Create.model_validate(item) for item in payload["creates"]],
            modifies=[Modify.model_validate(item) for item in payload["modifies"]],
            minutes=[int(value) for value in payload["minutes"]],
        )

    def chronicle(self, output: str | None = None) -> str:
        return str(self._exchange("chronicle", "chronicle", output))

    def action_resolution(self, actor_id: str, state: WorldState, result: str | None = None) -> str:
        """Record or restore the stateful result of a free-form Action.

        Action resolution may issue several state-mutating model tools. A state
        snapshot at this boundary is both smaller and more stable than replaying
        pydantic-ai's internal message/tool protocol.
        """
        payload = self._exchange("action_resolution", actor_id, {
            "result": result,
            "state": state.model_dump(mode="json"),
        } if self.mode == "record" else None)
        if self.is_playback:
            restored = WorldState.model_validate(payload["state"])
            for field_name in WorldState.model_fields:
                setattr(state, field_name, getattr(restored, field_name))
        return str(payload.get("result") or "")

    def director(self, output: DirectorResult | None = None) -> DirectorResult | None:
        payload = self._exchange("director", "director", None if output is None else {
            "event": output.event,
            "create": output.create.model_dump(mode="json") if output.create else None,
            "quest_id": output.quest_id,
        })
        if payload is None:
            return None
        return DirectorResult(
            event=str(payload["event"]),
            create=Create.model_validate(payload["create"]) if payload.get("create") else None,
            quest_id=payload.get("quest_id"),
        )

    def assert_consumed(self) -> None:
        if self.is_playback and self.remaining:
            raise ReplayError(f"Replay finished with {self.remaining} unconsumed output(s) in {self.path}")

    def _exchange(self, kind: ReplayKind, key: str, output: Any) -> Any:
        if self.mode == "record":
            entry = {"kind": kind, "key": key, "output": output}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return output

        if self._position >= len(self._entries):
            raise ReplayError(f"Replay exhausted while requesting {kind}:{key} at position {self._position}")
        entry = self._entries[self._position]
        self._position += 1
        if (entry.kind, entry.key) != (kind, key):
            raise ReplayError(
                f"Replay mismatch at position {self._position - 1}: "
                f"expected {kind}:{key}, found {entry.kind}:{entry.key}"
            )
        return entry.output

    def _load(self) -> list[_Entry]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ReplayError(f"Cannot read replay tape {self.path}: {exc}") from exc
        entries: list[_Entry] = []
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                kind = raw["kind"]
                if kind == "context":
                    if self._scenario is not None or entries:
                        raise ValueError("context must be the first and only context entry")
                    self._scenario = str(raw["scenario"])
                    self._character_id = str(raw["character_id"])
                    continue
                if kind not in {"character", "action_resolution", "dm", "director", "chronicle"}:
                    raise ValueError(f"unknown kind {kind!r}")
                entries.append(_Entry(kind=kind, key=str(raw["key"]), output=raw["output"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ReplayError(f"Invalid replay entry on line {index} of {self.path}: {exc}") from exc
        return entries
