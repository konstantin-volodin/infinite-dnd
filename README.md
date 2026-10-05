# Infinite DnD

AI-driven D&D engine where LLM agents run tabletop RPG sessions autonomously.

### 🚀 Features
- Autonomous campaign management
- NPC interaction & dialogue generation
- World generation & exploration
- Character creation & tracking
- Gold/item economy & trading
- Combat, death & kill XP
- Deterministic d20 checks and contested actions
- Off-screen faction agendas with progress clocks
- Human play from either the console or World dashboard

### 📋 Prerequisites
**python environment**
- `uv sync --extra dev`

**LLM (CUDA):**
- download the LLAMA CUDA binary from the [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases/latest)
- extract to desired path (`C:\Users\{USER}\.local\bin\llama-cuda`)
- add the folder to your `PATH`.

### ⚡ Running stuff
- `infinite-dnd` - runs the game
- `infinite-dnd --new-character` - runs the game with an interactively-created PC
- `infinite-dnd --interactive` - play the PC's turns yourself from the console; everything else stays autonomous
- `infinite-dnd --web` - play the PC from the World dashboard (start `infinite-dnd-state` first)
- `infinite-dnd --record-replay run.jsonl` - records structured agent decisions for deterministic playback
- `infinite-dnd --replay run.jsonl` - replays a recorded run without an LLM provider
- `infinite-dnd-state` - runs the app (World + Logs tabs)
- `infinite-dnd-scorecard` - compares campaign runs
- `infinite-dnd-recap` - exports a run's saved state as a markdown story recap
- `infinite-dnd-dump-context` - dumps each agent's exact prompt/tools/context for inspection

### 🌐 Deploy
Runs on a home server via `docker-compose.yml` + `Caddyfile`, behind a `.env` with
`LLM_PROVIDER=openai` (no GPU needed for that path — see `.env.example`).
- `docker compose up -d app caddy` - starts the dashboard, reachable at `http://dnd.voxlab.home` (point that host at the server's LAN IP in your local DNS first)
- `docker compose run --rm game` - starts one game session against the running dashboard; exits when the session ends or hits `--turns`
- override scenario/turns with `docker compose run --rm game infinite-dnd --web --web-url http://app:8765 --turns 30`
- only one game can be actively waiting on player input at a time (the dashboard's play broker is single-slot) - start a new one after the previous session ends

### 🤖 Code style                                                   
- Every change must leave the code easier to read than before.
- Prefer deletion over addition. If a refactor grows a file, question the approach.
- Remove dead code, redundant comments, and unused abstractions immediately.
- Simpler is correct. A smaller diff is usually a better diff.
- Before adding a function, check if the file's primary purpose is still singular. If not, it's time for a new module.

### 🧪 Tests
- Tests live under `tests/`, mirroring `src/`'s layout — one `test_*.py` per source module.
- `uv run pytest` runs the unit suite: deterministic, fast, no live LLM calls.
- `uv run pytest -m integration` runs the live-agent suite (`tests/integration/`); needs a reachable LLM provider and is skipped otherwise.

### 📂 Structure

```
src/                        - source code
├── agents/                 - LLM agent implementations
│   ├── action_resolver/        - sole writer; resolves intents against state
│   ├── character/              - PC/NPC turn: emits a CharacterTool
│   └── dm/                     - directing, narration, quest progress, and chronicle compaction
├── engine/                 - game engine components
│   ├── rules.py                - game rules and derived stats
│   ├── state/                  - world models and state persistence
│   └── runtime/                - game loop, messages, replay, and chronicle orchestration
├── interface/              - user interface
│   ├── templates/              - World and Logs page structure
│   └── static/                 - page scripts and shared/view-specific styles
├── tests/                  - agent prompt/tool/context inspection utilities
└── world/                  - bundled scenario data (one subdirectory per scenario)
tests/                      - pytest suite, mirroring src/ where practical
└── integration/                - live-agent tests, needs a reachable LLM provider
```
