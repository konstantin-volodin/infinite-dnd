"""Typed tool-call schema for the DM agent and the resolver that executes it."""

from typing import Literal

from pydantic import BaseModel


class Create(BaseModel):
    kind: Literal["create"] = "create"
    type: Literal["location", "item", "npc", "quest"]
    name: str
    description: str = ""
    location: str | None = None
    role: str | None = None
    goal: str | None = None
    owner: str | None = None
    plan: list[str] | None = None


class Modify(BaseModel):
    kind: Literal["modify"] = "modify"
    action: Literal["update_quest", "remove_npc", "update_location", "update_relationship", "advance_faction_clock"]
    target_id: str
    status: str | None = None
    step: str | None = None
    advance: bool = False
    reason: str | None = None
    other_id: str | None = None  # update_relationship: the opinion's subject; advance_faction_clock: the clock id
