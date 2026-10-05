"""
Data Models - Pydantic models for game state.
    - CharacterStats: HP, level, etc.
    - Location: id, description, connections, features, items.
    - Character: id, role, backstory, personality, goal, location, relationships, inventory, knowledge, stats.
    - HistoryEvent: text, location, characters involved.
    - Quest: id, title, description, status, owner, plan, current_step, steps.
    - WorldState: time, locations, characters, quests, history, chronicle.

See tests/engine/state/test_models.py for coverage.
"""

from typing import Dict, List

from pydantic import BaseModel, Field, model_validator

from src.engine.state.identifiers import slugify


class CharacterStats(BaseModel):
    hp: int = Field(default=5, ge=0)
    max_hp: int = Field(default=5, ge=0)
    level: int = Field(default=1, ge=1)
    xp: int = Field(default=0, ge=0)
    gold: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def hp_does_not_exceed_maximum(self) -> "CharacterStats":
        if self.hp > self.max_hp:
            raise ValueError("hp cannot exceed max_hp")
        return self


class Location(BaseModel):
    id: str
    description: str = ""
    connections: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    items: List[str] = Field(default_factory=list)


class Character(BaseModel):
    id: str
    role: str = "" 
    backstory: str = ""
    personality: str = "" 
    goal: str = ""
    location: str = ""
    relationships: Dict[str, str] = Field(default_factory=dict)
    inventory: List[str] = Field(default_factory=list)
    knowledge: List[str] = Field(default_factory=list)
    stats: CharacterStats = Field(default_factory=CharacterStats)


class HistoryEvent(BaseModel):
    text: str
    location: str
    characters: List[str] = Field(default_factory=list)
    minutes_elapsed: int = Field(default=0, ge=0)


class Quest(BaseModel):
    id: str
    title: str
    description: str = ""
    status: str = "active"
    owner: str = ""
    plan: List[str] = Field(default_factory=list)  # ordered, concrete objectives set at creation
    current_step: int = Field(default=0, ge=0)  # index into plan of the current objective
    steps: List[str] = Field(default_factory=list)  # progress log — accomplished objectives + notes

    @model_validator(mode="after")
    def status_is_canonical(self) -> "Quest":
        status = self.status.strip()
        normalized_status = status.casefold()
        if normalized_status in {"active", "completed", "failed"}:
            self.status = normalized_status
            return self

        # Older agents sometimes wrote a progress description into `status`.
        # Keep that information as a note while restoring lifecycle semantics.
        if status and not any(status.casefold() in step.casefold() for step in self.steps):
            self.steps.append(status)
        self.status = "active"
        return self

    @model_validator(mode="after")
    def planned_progress_is_consistent(self) -> "Quest":
        if any(not objective.strip() for objective in self.plan):
            raise ValueError("quest plan objectives cannot be blank")
        if not self.plan:
            return self
        if self.current_step > len(self.plan):
            raise ValueError("current_step cannot exceed plan length")
        if self.status == "active" and self.current_step == len(self.plan):
            raise ValueError("active quest cannot have an exhausted plan")
        if self.status == "completed" and self.current_step != len(self.plan):
            raise ValueError("completed quest must exhaust its plan")
        return self


class ProgressClock(BaseModel):
    """A finite, deterministic countdown toward a faction consequence."""

    id: str
    name: str
    consequence: str = Field(min_length=1)
    progress: int = Field(default=0, ge=0)
    segments: int = Field(ge=1)
    event_acceleration: bool = True
    consequence_triggered: bool = False
    fail_quest_id: str | None = None

    @model_validator(mode="after")
    def progress_does_not_exceed_segments(self) -> "ProgressClock":
        if self.progress > self.segments:
            raise ValueError("progress cannot exceed segments")
        if self.progress == self.segments and not self.consequence_triggered:
            raise ValueError("completed clock must trigger its consequence")
        if self.consequence_triggered and self.progress < self.segments:
            raise ValueError("consequence cannot be triggered before clock completion")
        return self


class Faction(BaseModel):
    """An off-screen actor pursuing a goal through one or more progress clocks."""

    id: str
    name: str
    goal: str
    clocks: List[ProgressClock] = Field(min_length=1)

    @model_validator(mode="after")
    def clock_ids_are_unique(self) -> "Faction":
        clock_ids = [slugify(clock.id) for clock in self.clocks]
        if any(not clock_id for clock_id in clock_ids):
            raise ValueError("clock ids must contain a letter or number")
        if len(clock_ids) != len(set(clock_ids)):
            raise ValueError("clock ids must be unique after normalization within a faction")
        return self


class WorldState(BaseModel):
    time: int = Field(default=0, ge=0)
    minutes_elapsed: int = Field(default=0, ge=0)
    last_quest_advance_time: int = Field(default=0, ge=0)  # tick of the most recent quest advancement — stall detection
    director_interventions: Dict[str, int] = Field(default_factory=dict)  # quest id (or "world") → director beat count
    locations: Dict[str, Location] = Field(default_factory=dict)
    characters: Dict[str, Character] = Field(default_factory=dict)
    quests: Dict[str, Quest] = Field(default_factory=dict)
    factions: Dict[str, Faction] = Field(default_factory=dict)
    history: List[HistoryEvent] = Field(default_factory=list)
    chronicle: List[str] = Field(default_factory=list)  # compact era summaries of history archived by compaction

    @model_validator(mode="after")
    def quest_advance_time_is_not_in_the_future(self) -> "WorldState":
        if self.last_quest_advance_time > self.time:
            raise ValueError("last_quest_advance_time cannot exceed time")
        return self

    @model_validator(mode="after")
    def faction_clock_quest_links_exist(self) -> "WorldState":
        entity_maps = {
            "location": self.locations,
            "character": self.characters,
            "quest": self.quests,
            "faction": self.factions,
        }
        for entity_name, entities in entity_maps.items():
            for entity_id, entity in entities.items():
                if entity.id != entity_id:
                    raise ValueError(
                        f"{entity_name} map key '{entity_id}' does not match "
                        f"entity id '{entity.id}'"
                    )
            normalized_ids = [slugify(entity_id) for entity_id in entities]
            if any(not entity_id for entity_id in normalized_ids):
                raise ValueError(f"{entity_name} ids must contain a letter or number")
            if len(normalized_ids) != len(set(normalized_ids)):
                raise ValueError(f"{entity_name} ids must be unique after normalization")
        for character in self.characters.values():
            if character.location and character.location not in self.locations:
                raise ValueError(
                    f"character '{character.id}' location '{character.location}' "
                    "is not a known location"
                )
        for quest in self.quests.values():
            if (
                quest.owner
                and quest.owner not in self.characters
                and quest.status.casefold() not in {"completed", "failed"}
            ):
                raise ValueError(
                    f"active quest '{quest.id}' owner '{quest.owner}' is not a known character"
                )
        for faction in self.factions.values():
            for clock in faction.clocks:
                if not clock.fail_quest_id:
                    continue
                linked_quest = self.quests.get(clock.fail_quest_id)
                if linked_quest is None:
                    raise ValueError(
                        f"faction clock '{faction.id}/{clock.id}' links unknown quest "
                        f"'{clock.fail_quest_id}'"
                    )
                if clock.consequence_triggered and linked_quest.status != "failed":
                    raise ValueError(
                        f"triggered faction clock '{faction.id}/{clock.id}' requires linked quest "
                        f"'{linked_quest.id}' to be failed"
                    )
        return self
