"""Deterministic faction-agenda and progress-clock operations."""

from src.engine.state.operations._base import _OpsBase


_TERMINAL_QUEST_STATUSES = {"completed", "failed"}


class FactionOps(_OpsBase):
    def advance_faction_clock(self, faction_id: str, clock_id: str, amount: int = 1) -> str:
        """Advance one clock, clamping at its segment count and firing its consequence once."""
        if amount < 0:
            return "Cannot advance a faction clock by a negative amount."

        faction = self.state.factions.get(faction_id)
        if not faction:
            return f"Cannot advance faction clock — faction '{faction_id}' not found."

        clock = next((candidate for candidate in faction.clocks if candidate.id == clock_id), None)
        if not clock:
            return f"Cannot advance faction clock — clock '{clock_id}' not found in faction '{faction_id}'."

        linked_quest = self.state.quests.get(clock.fail_quest_id) if clock.fail_quest_id else None
        if linked_quest and linked_quest.status.lower() in _TERMINAL_QUEST_STATUSES:
            return (
                f"Clock '{clock_id}' no longer advances — linked quest "
                f"'{linked_quest.id}' is already {linked_quest.status.lower()}."
            )

        clock.progress = min(clock.segments, clock.progress + amount)
        if clock.progress == clock.segments and not clock.consequence_triggered:
            clock.consequence_triggered = True
            event_text = f"{faction.name} completes '{clock.name}'. {clock.consequence}"
            if linked_quest:
                linked_quest.status = "failed"
                self.state.last_quest_advance_time = self.state.time
                event_text += f" Quest '{linked_quest.title}' failed."
            self._log(event_text, "", list(self.state.characters) if linked_quest else None)
            result = f"Clock '{clock_id}' completed. Consequence triggered: {clock.consequence}"
            if linked_quest:
                result += f" Quest '{linked_quest.id}' failed."
            return result
        return f"Clock '{clock_id}' advanced to {clock.progress}/{clock.segments}."

    def advance_faction_clocks(self, amount: int = 1) -> list[str]:
        """Advance every incomplete faction clock in stable faction/clock order."""
        results: list[str] = []
        for faction_id in sorted(self.state.factions):
            faction = self.state.factions[faction_id]
            for clock in sorted(faction.clocks, key=lambda candidate: candidate.id):
                linked_quest = self.state.quests.get(clock.fail_quest_id) if clock.fail_quest_id else None
                linked_quest_resolved = (
                    linked_quest is not None
                    and linked_quest.status.lower() in _TERMINAL_QUEST_STATUSES
                )
                if not clock.consequence_triggered and not linked_quest_resolved:
                    results.append(self.advance_faction_clock(faction_id, clock.id, amount))
        return results

    def advance_faction_clocks_hourly(self, pre_minutes: int, post_minutes: int) -> list[str]:
        """Advance every clock once per full in-world hour crossed between the two timestamps."""
        hours = post_minutes // 60 - pre_minutes // 60
        return self.advance_faction_clocks(hours) if hours > 0 else []
