"""Shared base for state-operation classes: state binding + history logging."""

from src.engine.state.models import WorldState, HistoryEvent


class _OpsBase:
    def __init__(self, state: WorldState):
        self.state = state

    def _log(
        self, text: str, location: str, characters: list[str] | None = None
    ) -> None:
        self.state.history.append(
            HistoryEvent(text=text, location=location, characters=characters or [])
        )
