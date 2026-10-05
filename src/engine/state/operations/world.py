"""
World-scoped operations: quests, characters (spawn/delete), items (world-level), locations, time.
"""

from typing import Protocol, cast

from src.engine.state.models import Character, Location, Quest
from src.engine.state.operations._base import _OpsBase
from src.engine.state.queries import (
    character_ids_match,
    resolve_character,
    resolve_location_id,
    slugify,
)


class _XpAwarder(Protocol):
    def award_xp(self, character_id: str, amount: int, reason: str = "") -> str: ...


_TERMINAL_QUEST_STATUSES = {"completed", "failed"}
_QUEST_STATUSES = {"active", *_TERMINAL_QUEST_STATUSES}
_IDENTITY_ARTICLES = {"a", "an", "the"}
_GENERIC_QUEST_WORDS = {
    "a",
    "about",
    "after",
    "an",
    "and",
    "ask",
    "at",
    "before",
    "check",
    "clear",
    "clue",
    "clues",
    "confront",
    "determine",
    "defeat",
    "discover",
    "find",
    "for",
    "from",
    "in",
    "investigate",
    "learn",
    "locate",
    "of",
    "on",
    "or",
    "search",
    "scout",
    "the",
    "to",
    "track",
    "with",
}


def _quest_topic_words(title: str, description: str, plan: list[str] | None) -> set[str]:
    text = " ".join([title, description, *(plan or [])]).replace("'s", "").replace("’s", "")
    return {
        word
        for word in slugify(text).split("-")
        if len(word) > 2 and word not in _GENERIC_QUEST_WORDS
    }


def _overlapping_active_quest(
    quests: dict[str, Quest],
    owner_id: str,
    title: str,
    description: str,
    plan: list[str] | None,
) -> Quest | None:
    if not owner_id:
        return None
    proposed_words = _quest_topic_words(title, description, plan)
    if len(proposed_words) < 2:
        return None
    return next(
        (
            quest
            for quest in quests.values()
            if quest.owner == owner_id
            and quest.status.casefold() not in _TERMINAL_QUEST_STATUSES
            and len(
                proposed_words
                & _quest_topic_words(quest.title, quest.description, quest.plan)
            )
            >= 2
        ),
        None,
    )


def _identity_words(value: str) -> set[str]:
    return {
        word
        for word in slugify(value).split("-")
        if word and word not in _IDENTITY_ARTICLES
    }


def _role_qualified_alias_matches(alias: str, character: Character) -> bool:
    """Match a name/title alias against a character's name and role."""
    alias_words = _identity_words(alias)
    id_words = _identity_words(character.id)
    role_words = _identity_words(character.role)
    return bool(
        alias_words
        and alias_words <= id_words | role_words
        and alias_words & id_words
        and alias_words & role_words
    )


def _canonical_reveal_id(
    characters: dict[str, Character], proposed_id: str
) -> str:
    """Reuse one unresolved relationship alias that uniquely extends a short name."""
    proposed_words = _identity_words(proposed_id)
    if not proposed_words:
        return proposed_id

    deferred_aliases: set[str] = set()
    for character in characters.values():
        for related_id in character.relationships:
            alias_id = slugify(related_id)
            if not alias_id or any(
                character_ids_match(alias_id, candidate_id)
                or _role_qualified_alias_matches(alias_id, candidate)
                for candidate_id, candidate in characters.items()
            ):
                continue
            if proposed_words < _identity_words(alias_id):
                deferred_aliases.add(alias_id)

    if len(deferred_aliases) == 1:
        return deferred_aliases.pop()
    return proposed_id


def _reconcile_relationship_aliases(
    characters: dict[str, Character], canonical_id: str
) -> None:
    """Bind a unique deferred name/title alias when its character is revealed."""
    for character in characters.values():
        for related_id, relation in list(character.relationships.items()):
            if related_id in characters:
                continue
            candidates = [
                candidate_id
                for candidate_id, candidate in characters.items()
                if character_ids_match(related_id, candidate_id)
                or _role_qualified_alias_matches(related_id, candidate)
            ]
            if candidates != [canonical_id]:
                continue
            character.relationships.pop(related_id)
            character.relationships.setdefault(canonical_id, relation)


class WorldOps(_OpsBase):
    # ============ QUESTS ============
    def advance_quest(self, quest_id: str, new_status: str | None = None, step: str | None = None, advance: bool = False) -> str:
        quest_key = next(
            (candidate for candidate in self.state.quests if slugify(candidate) == slugify(quest_id)),
            None,
        )
        if quest_key is None:
            title_matches = [
                candidate
                for candidate, quest in self.state.quests.items()
                if slugify(quest.title) == slugify(quest_id)
            ]
            quest_key = title_matches[0] if len(title_matches) == 1 else quest_id
        quest = self.state.quests.get(quest_key)
        if not quest: return f"Cannot advance quest — '{quest_id}' not found."
        quest_id = quest_key
        if quest.status.lower() in _TERMINAL_QUEST_STATUSES:
            return f"Cannot advance quest — '{quest_id}' is already {quest.status.lower()}."
        normalized_status = new_status.strip().casefold() if new_status is not None else None
        if normalized_status is not None and normalized_status not in _QUEST_STATUSES:
            return f"Cannot advance quest — unsupported status '{new_status}'."
        if advance and normalized_status == "failed":
            return (
                f"Cannot advance quest — '{quest_id}' cannot advance and fail "
                "in the same update."
            )
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
            if quest.plan and quest.current_step >= len(quest.plan) and normalized_status in {None, "active"}:
                normalized_status = "completed"
        elif step:
            quest.steps.append(step)

        status_changed = normalized_status is not None and normalized_status != quest.status.casefold()
        if normalized_status: quest.status = normalized_status
        if advance or status_changed:
            self.state.last_quest_advance_time = self.state.time  # stall detection

        # XP award to owner — step/advance=10, completion=50. Silent if owner isn't a known character.
        xp_awarded = 0
        if quest.owner and quest.owner in self.state.characters:
            award_xp = cast(_XpAwarder, self).award_xp
            if awarded_step:
                award_xp(quest.owner, 10, reason=f"quest '{quest_id}' progress")
                xp_awarded += 10
            if normalized_status == "completed":
                award_xp(quest.owner, 50, reason=f"quest '{quest_id}' completed")
                xp_awarded += 50
        award_suffix = f" (+{xp_awarded} XP to {quest.owner})" if xp_awarded else ""
        return f"Quest '{quest_id}' updated.{award_suffix}"

    def add_quest(self, quest_id: str, title: str, description: str = "", owner: str | None = None, plan: list[str] | None = None) -> str:
        if not slugify(quest_id): return "Cannot add quest — quest ID must contain a letter or number."
        if any(slugify(existing_id) == slugify(quest_id) for existing_id in self.state.quests):
            return f"Quest '{quest_id}' already exists."
        if plan and any(not objective.strip() for objective in plan):
            return "Cannot add quest — plan objectives cannot be blank."

        owner_id = owner or ""
        if owner_id:
            resolved_owner = resolve_character(self.state, owner_id)
            if not resolved_owner:
                return f"Cannot add quest — owner '{owner}' not found."
            owner_id = resolved_owner.id
        overlapping = _overlapping_active_quest(
            self.state.quests,
            owner_id,
            title,
            description,
            plan,
        )
        if overlapping:
            return (
                f"Cannot add quest — overlaps active quest '{overlapping.id}' "
                f"for owner '{owner_id}'."
            )
        self.state.quests[quest_id] = Quest(id=quest_id, title=title, description=description, owner=owner_id, plan=plan or [])
        owner_loc = self.state.characters[owner_id].location if owner_id in self.state.characters else ""
        self._log(f"New quest: '{title}'" + (f" (owner: {owner_id})" if owner_id else ""), owner_loc, [owner_id] if owner_id else None)
        return f"Quest '{quest_id}' added."

    # ============ CHARACTERS ============
    def spawn_character(self, character_id: str, role: str, location_id: str, backstory: str = "", goal: str = "", personality: str = "") -> str:
        if not slugify(character_id): return "Cannot spawn character — character ID must contain a letter or number."
        if any(slugify(existing_id) == slugify(character_id) for existing_id in self.state.characters):
            return f"Cannot spawn '{character_id}' — character already exists."
        resolved_location_id = resolve_location_id(self.state, location_id)
        if not resolved_location_id: return f"Cannot spawn '{character_id}' — location '{location_id}' not found."

        self.state.characters[character_id] = Character(id=character_id, role=role, location=resolved_location_id, backstory=backstory, goal=goal, personality=personality)
        _reconcile_relationship_aliases(self.state.characters, character_id)
        result = f"{character_id} appears at '{resolved_location_id}'."
        self._log(result, resolved_location_id, [character_id])
        return result

    def reveal_character(self, character_id: str, role: str, location_id: str, backstory: str = "", goal: str = "", personality: str = "") -> str:
        """Reveal an NPC, reusing a uniquely matching known identity when possible."""
        resolved_location_id = resolve_location_id(self.state, location_id)
        if not resolved_location_id:
            return f"Cannot reveal '{character_id}' — location '{location_id}' not found."

        existing = resolve_character(self.state, character_id)
        if not existing:
            character_id = _canonical_reveal_id(self.state.characters, character_id)
            return self.spawn_character(
                character_id,
                role,
                resolved_location_id,
                backstory=backstory,
                goal=goal,
                personality=personality,
            )
        if existing.location == resolved_location_id:
            return f"{existing.id} is already at '{resolved_location_id}'."

        existing.location = resolved_location_id
        result = f"{existing.id} appears at '{resolved_location_id}'."
        self._log(result, resolved_location_id, [existing.id])
        return result

    def delete_npc(self, npc_id: str, reason: str = "") -> str:
        char = self.state.characters.get(npc_id)
        if not char: return f"Cannot delete '{npc_id}' — character not found."

        active_quest = next(
            (
                quest
                for quest in self.state.quests.values()
                if quest.owner == npc_id and quest.status.lower() not in _TERMINAL_QUEST_STATUSES
            ),
            None,
        )
        if active_quest:
            return (
                f"Cannot delete '{npc_id}' — character owns active quest "
                f"'{active_quest.id}'."
            )

        location = char.location
        for other_character in self.state.characters.values():
            other_character.relationships.pop(npc_id, None)
        del self.state.characters[npc_id]
        result = f"{npc_id} is gone. {reason}".strip()
        self._log(result, location)
        return result

    # ============ ITEMS ============
    def create_item(self, item: str, location_id: str) -> str:
        loc = self.state.locations.get(location_id)
        if not loc: return f"Cannot create item — location '{location_id}' not found."

        normalized_item = slugify(item)
        if not normalized_item:
            return "Cannot create item — item name must contain a letter or number."
        for existing_location in self.state.locations.values():
            if any(slugify(existing_item) == normalized_item for existing_item in existing_location.items):
                return f"Cannot create item — '{item}' already exists at '{existing_location.id}'."
        for character in self.state.characters.values():
            if any(slugify(existing_item) == normalized_item for existing_item in character.inventory):
                return f"Cannot create item — '{item}' already belongs to '{character.id}'."

        loc.items.append(item)
        result = f"'{item}' appears at '{location_id}'."
        self._log(result, location_id)
        return result

    # ============ LOCATIONS ============
    def add_location(self, location_id: str, description: str = "", connections: list[str] | None = None) -> str:
        if not slugify(location_id): return "Cannot add location — location ID must contain a letter or number."
        if any(slugify(existing_id) == slugify(location_id) for existing_id in self.state.locations):
            return f"Location '{location_id}' already exists."

        requested_connections = connections or []
        unknown_connection = next(
            (connection for connection in requested_connections if connection not in self.state.locations),
            None,
        )
        if unknown_connection is not None:
            return (
                f"Cannot add location '{location_id}' — connected location "
                f"'{unknown_connection}' not found."
            )

        self.state.locations[location_id] = Location(
            id=location_id,
            description=description,
            connections=requested_connections,
        )

        for c in requested_connections:
            if location_id not in self.state.locations[c].connections:
                self.state.locations[c].connections.append(location_id)

        return f"Location '{location_id}' added."

    def modify_location(self, location_id: str, description: str | None = None, add_feature: str | None = None, remove_feature: str | None = None) -> str:
        loc = self.state.locations.get(location_id)
        if not loc:
            return f"Cannot modify — location '{location_id}' not found."

        before = loc.model_dump()
        if description:
            loc.description = description
        if add_feature and add_feature not in loc.features:
            loc.features.append(add_feature)
        if remove_feature and remove_feature in loc.features:
            loc.features.remove(remove_feature)
        if loc.model_dump() == before:
            return f"Location '{location_id}' unchanged."
        return f"Location '{location_id}' updated."

    # ============ EVENTS ============
    def world_event(self, text: str, location_id: str) -> str:
        """log a narrative event witnessed by everyone present at the location."""
        witnesses = [cid for cid, c in self.state.characters.items() if c.location == location_id]
        self._log(text, location_id, witnesses)
        return text
