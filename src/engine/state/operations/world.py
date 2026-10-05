"""
World-scoped operations: quests, characters (spawn/delete), items (world-level), locations, time.
"""

from typing import Protocol, cast

from src.engine.state.models import Character, Location, Quest
from src.engine.state.operations._base import _OpsBase


class _XpAwarder(Protocol):
    def award_xp(self, character_id: str, amount: int, reason: str = "") -> str: ...


_TERMINAL_QUEST_STATUSES = {"completed", "failed"}


class WorldOps(_OpsBase):
    # ============ QUESTS ============
    def advance_quest(self, quest_id: str, new_status: str | None = None, step: str | None = None, advance: bool = False) -> str:
        quest = self.state.quests.get(quest_id)
        if not quest: return f"Cannot advance quest — '{quest_id}' not found."
        if quest.status.lower() in _TERMINAL_QUEST_STATUSES:
            return f"Cannot advance quest — '{quest_id}' is already {quest.status.lower()}."
        normalized_status = new_status.strip().casefold() if new_status is not None else None
        if advance and quest.plan and quest.current_step >= len(quest.plan):
            return f"Cannot advance quest — '{quest_id}' has no remaining objectives."

        # A planned quest can only complete by advancing its final objective.
        # DM status updates alone may reflect a lead or a discovered location,
        # neither of which is evidence that the resolution objective was met.
        completing_final_objective = advance and quest.current_step == len(quest.plan) - 1
        if normalized_status == "completed" and quest.plan and not completing_final_objective:
            return f"Cannot complete quest — '{quest_id}' has not achieved its final objective."

        # `advance` accomplishes the CURRENT plan objective: log it, move the pointer, maybe auto-complete.
        # `step` alone (no advance) is a plain log note — used alongside explicit new_status.
        awarded_step = False
        if advance:
            objective = quest.plan[quest.current_step] if quest.current_step < len(quest.plan) else None
            note = f"{objective} — {step}" if objective and step else (objective or step or "objective accomplished")
            quest.steps.append(note)
            quest.current_step += 1
            awarded_step = True
            if quest.plan and quest.current_step >= len(quest.plan) and not normalized_status:
                normalized_status = "completed"
        elif step:
            quest.steps.append(step)

        if normalized_status: quest.status = normalized_status
        if advance or normalized_status: self.state.last_quest_advance_time = self.state.time  # stall detection

        # XP award to owner — step/advance=10, completion=50. Silent if owner isn't a known character.
        award_suffix = ""
        if quest.owner and quest.owner in self.state.characters:
            award_xp = cast(_XpAwarder, self).award_xp
            if awarded_step:
                award_xp(quest.owner, 10, reason=f"quest '{quest_id}' progress")
                award_suffix = f" (+10 XP to {quest.owner})"
            if normalized_status == "completed":
                award_xp(quest.owner, 50, reason=f"quest '{quest_id}' completed")
                award_suffix = f" (+50 XP to {quest.owner})"
        return f"Quest '{quest_id}' updated.{award_suffix}"

    def add_quest(self, quest_id: str, title: str, description: str = "", owner: str | None = None, plan: list[str] | None = None) -> str:
        if quest_id in self.state.quests: return f"Quest '{quest_id}' already exists."

        owner_id = owner or ""
        self.state.quests[quest_id] = Quest(id=quest_id, title=title, description=description, owner=owner_id, plan=plan or [])
        owner_loc = self.state.characters[owner_id].location if owner_id in self.state.characters else ""
        self._log(f"New quest: '{title}'" + (f" (owner: {owner_id})" if owner_id else ""), owner_loc, [owner_id] if owner_id else None)
        return f"Quest '{quest_id}' added."

    # ============ CHARACTERS ============
    def spawn_character(self, character_id: str, role: str, location_id: str, backstory: str = "", goal: str = "", personality: str = "") -> str:
        if character_id in self.state.characters: return f"Cannot spawn '{character_id}' — character already exists."
        if location_id not in self.state.locations: return f"Cannot spawn '{character_id}' — location '{location_id}' not found."

        self.state.characters[character_id] = Character(id=character_id, role=role, location=location_id, backstory=backstory, goal=goal, personality=personality)
        result = f"{character_id} appears at '{location_id}'."
        self._log(result, location_id, [character_id])
        return result

    def delete_npc(self, npc_id: str, reason: str = "") -> str:
        char = self.state.characters.get(npc_id)
        if not char: return f"Cannot delete '{npc_id}' — character not found."

        location = char.location
        del self.state.characters[npc_id]
        result = f"{npc_id} is gone. {reason}".strip()
        self._log(result, location)
        return result

    # ============ ITEMS ============
    def create_item(self, item: str, location_id: str) -> str:
        loc = self.state.locations.get(location_id)
        if not loc: return f"Cannot create item — location '{location_id}' not found."
        if item in loc.items: return f"'{item}' already exists at '{location_id}'."

        loc.items.append(item)
        result = f"'{item}' appears at '{location_id}'."
        self._log(result, location_id)
        return result

    # ============ LOCATIONS ============
    def add_location(self, location_id: str, description: str = "", connections: list[str] | None = None) -> str:
        if location_id in self.state.locations: return f"Location '{location_id}' already exists."

        validated = [c for c in (connections or []) if c in self.state.locations]
        self.state.locations[location_id] = Location(id=location_id, description=description, connections=validated)

        for c in validated:
            if location_id not in self.state.locations[c].connections:
                self.state.locations[c].connections.append(location_id)

        return f"Location '{location_id}' added."

    def modify_location(self, location_id: str, description: str | None = None, add_feature: str | None = None, remove_feature: str | None = None) -> str:
        loc = self.state.locations.get(location_id)
        if not loc: return f"Cannot modify — location '{location_id}' not found."

        if description: loc.description = description
        if add_feature and add_feature not in loc.features: loc.features.append(add_feature)
        if remove_feature and remove_feature in loc.features: loc.features.remove(remove_feature)
        return f"Location '{location_id}' updated."

    # ============ EVENTS ============
    def world_event(self, text: str, location_id: str) -> str:
        """log a narrative event witnessed by everyone present at the location."""
        witnesses = [cid for cid, c in self.state.characters.items() if c.location == location_id]
        self._log(text, location_id, witnesses)
        return text
