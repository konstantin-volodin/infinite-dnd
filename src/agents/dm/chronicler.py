"""Chronicler agent: writes each archived batch of events up as a prose chapter, and
folds old chapter summaries into a bounded "saga so far".

Output is plain text, not a tool call: long prose inside JSON is fragile on small
local models, and the layout below is trivial to parse.
"""

import re

from pydantic_ai import Agent

from src.agents.utils import create_model
from src.engine.state import WorldState
from src.engine.state.models import HistoryEvent

# The narrator may use its own model (LLM_NARRATOR_MODEL): the best storyteller that
# fits a small GPU is not always the best at the game's tool calls.
agent: Agent[None, str] = Agent(
    model=create_model("LLM_NARRATOR_MODEL"), output_type=str
)

CHAPTER_INSTRUCTIONS = """\
You are the narrator of a fantasy serial, writing one chapter at a time from the game \
log of a D&D campaign. Write the next chapter from the events you are given.

Rules:
- Faithful: every outcome in your chapter must come from the events. Never invent \
fights, deaths, discoveries, or decisions that are not in them, and never give a \
character an item, wound, or trait the events do not mention. You may add weather, \
sensory detail, mood, and the characters' inner thoughts.
- No game mechanics: show dice rolls, checks, and damage numbers as what they look \
like in the story (a clean blow, a stumble, a shallow cut), never as numbers.
- Write in past tense, third person, following the hero closely.
- Turn the log into a story: merge repetitive actions, skip idle moments, keep the \
important lines of dialogue as quoted speech. Quote only words the events give a \
character; do not write new speeches.
- Lines about goals, XP, quest progress, what someone "learns", or a place being \
"updated" are the game's bookkeeping: use them to understand what characters want \
and know, not as scenes.
- Pick up where the previous chapter ended, and do not retell it.
- End on the last thing that happens, not on a reflection or a moral.
- Plain prose paragraphs only: no headings, lists, or bold text.

Reply in exactly this layout:
TITLE: <a short evocative chapter title>
PROSE:
<the chapter, several paragraphs>
SUMMARY: <two or three plain sentences of fact for the campaign record: who did \
what, where, what changed, and what is still unresolved>
"""

SAGA_INSTRUCTIONS = """\
You keep the long-term memory of a D&D campaign. Merge the passages below, oldest \
first, into one "saga so far" of at most 120 words. Keep every important name, what \
each did, and the causes behind what is still unresolved. Drop scenery and dialogue. \
Reply with the merged text only.
"""

_FIELD = re.compile(r"^\W*(TITLE|SUMMARY|PROSE)\W*:\s*(.*)$", re.IGNORECASE)


def parse_chapter(text: str) -> tuple[str, str, str]:
    """Split a chapter reply into (title, summary, prose); missing parts come back empty."""
    fields: dict[str, list[str]] = {"title": [], "summary": [], "prose": []}
    current: list[str] | None = None
    for line in text.splitlines():
        if match := _FIELD.match(line):
            current = fields[match.group(1).lower()]
            line = match.group(2)
        if current is not None:
            current.append(line)
        if current is fields["title"]:  # one line; anything after it is prose
            current = fields["prose"]
    title = " ".join(fields["title"]).strip().strip("*#\"' ")
    summary = " ".join(" ".join(fields["summary"]).split())
    paragraphs = (" ".join(line.split()) for line in fields["prose"])
    return title, summary, "\n\n".join(p for p in paragraphs if p)


def readable(text: str, state: WorldState) -> str:
    """Drop dice brackets and turn slugs into names: 'mira-thorne' -> 'Mira Thorne'."""
    text = re.sub(r" ?\[[^\]\n]*\]", "", text)
    names = {cid: cid.replace("-", " ").title() for cid in state.characters}
    names |= {lid: lid.replace("-", " ") for lid in state.locations}
    for slug in sorted(names, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(slug)}(?![\w-])", names[slug], text)
    return text


def chapter_prompt(
    state: WorldState,
    events: list[HistoryEvent],
    *,
    number: int,
    hero: str,
    start_minutes: int,
    previous_ending: str,
) -> str:
    """The bounded context a chapter needs: hero, saga, last ending, open quests, events."""
    pc = state.characters.get(hero)
    lines = [
        f"## the hero\n{hero}"
        + (f", {pc.role}. {pc.backstory} {pc.personality}" if pc else "")
    ]
    if state.chronicle:
        lines.append("## the story so far\n" + "\n".join(state.chronicle))
    if previous_ending:
        lines.append(f"## how chapter {number - 1} ended\n{previous_ending}")
    quests = [q for q in state.quests.values() if q.status == "active"]
    if quests:
        lines.append(
            "## open quests\n"
            + "\n".join(
                f"- {q.title}"
                + (
                    f": {q.plan[q.current_step]}"
                    if q.current_step < len(q.plan)
                    else ""
                )
                for q in quests
            )
        )
    log, clock, place = [], start_minutes, None
    for event in events:
        if event.location != place:
            place = event.location
            log.append(f"\n### day {clock // 1440 + 1}, {place}")
        log.append(f"- {event.text}")
        clock += event.minutes_elapsed
    lines.append(f"## events of chapter {number}, in order" + "\n".join(log))
    opening = " This is the first chapter: open the story." if number == 1 else ""
    lines.append(
        f"Write chapter {number}.{opening} Keep to what the events say, and leave the "
        "numbers of the game log (damage, HP, rolls) out of the prose."
    )
    return readable("\n\n".join(lines), state)
