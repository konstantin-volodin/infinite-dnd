import argparse
import os
from contextlib import nullcontext

from src.agents.server import LlamaServer
from src.engine.runtime import ReplayTape, run_game
from src.engine.state import StateManager, slugify
from src.interface.player import console_pc_controller
from src.interface.web_player import make_web_pc_controller


def _prompt_new_character(locations: list[str], existing_ids: set[str]) -> dict:
    print("\n=== Create Your Character ===")
    name = input("Name: ").strip()
    character_id = slugify(name)
    while not character_id or character_id in existing_ids:
        prompt = "Name cannot be empty. Name: " if not character_id else f"'{character_id}' is already taken. Choose a different name: "
        name = input(prompt).strip()
        character_id = slugify(name)

    role = input("Role: ").strip()
    backstory = input("Backstory: ").strip()
    personality = input("Personality: ").strip()
    goal = input("Goal: ").strip()

    location = input(f"Starting location ({', '.join(locations)}): ").strip()
    while location not in locations:
        location = input(f"Not a valid location. Choose from {', '.join(locations)}: ").strip()

    return {
        "character_id": character_id,
        "role": role,
        "location_id": location,
        "backstory": backstory,
        "personality": personality,
        "goal": goal,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Infinite DnD session.")
    parser.add_argument("--scenario", default=None, help="Scenario id under src/world/ (default: random).")
    parser.add_argument("--character", default=None, help="PC id to play as (default: scenario's manifest pc).")
    parser.add_argument("--turns", "--turn", dest="turns", type=int, default=50, help="Max turns to run.")
    parser.add_argument("--new-character", action="store_true", help="Create a new PC interactively instead of playing a scenario's preset character.")
    parser.add_argument("--interactive", action="store_true", help="Play the PC's turns yourself from the console instead of the character agent.")
    parser.add_argument("--web", action="store_true", help="Play PC turns in the World dashboard (start infinite-dnd-state first).")
    parser.add_argument("--web-url", default="http://127.0.0.1:8765", help="World dashboard URL used with --web.")
    replay_group = parser.add_mutually_exclusive_group()
    replay_group.add_argument("--record-replay", metavar="PATH", help="Record structured agent outputs to a JSONL replay tape.")
    replay_group.add_argument("--replay", metavar="PATH", help="Replay structured agent outputs from a JSONL tape without character, DM, or director calls.")
    args = parser.parse_args(argv)
    if args.interactive and args.web:
        parser.error("--interactive and --web cannot be used together")
    return args


def main() -> None:
    args = _parse_args()

    scenario = args.scenario
    new_character = None
    if args.new_character:
        manager = StateManager(scenario=scenario)
        scenario = manager.scenario  # lock in the resolved scenario so the engine doesn't re-roll a random one
        state = manager.init_state()
        new_character = _prompt_new_character(sorted(state.locations), set(state.characters))

    # LlamaServer manages the local llama.cpp process; skip it when routing to a hosted provider.
    if args.replay:
        tape = ReplayTape.playback(args.replay)
    elif args.record_replay:
        tape = ReplayTape.recording(args.record_replay)
    else:
        tape = None
    server = LlamaServer() if not args.replay and os.getenv("LLM_PROVIDER", "local") == "local" else nullcontext()
    with server:
        succeeded = run_game(
            character_id=args.character,
            max_turns=args.turns,
            scenario=scenario,
            new_character=new_character,
            replay=tape,
            pc_controller=(
                console_pc_controller if args.interactive else make_web_pc_controller(args.web_url) if args.web else None
            ),
        )
    if not succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
