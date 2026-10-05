import pytest

from src.cli import _parse_args, main


@pytest.mark.parametrize("flag", ["--turn", "--turns"])
def test_turn_flag_aliases_set_max_turns(flag):
    args = _parse_args([flag, "10"])

    assert args.turns == 10


def test_main_exits_unsuccessfully_when_campaign_run_fails(monkeypatch):
    monkeypatch.setattr("sys.argv", ["infinite-dnd", "--turn", "1"])
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    monkeypatch.setattr("src.cli.run_game", lambda **_kwargs: False)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1


def test_main_passes_resume_to_engine(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["infinite-dnd", "--scenario", "smuggler-cove", "--resume"]
    )
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    calls = []
    monkeypatch.setattr(
        "src.cli.run_game", lambda **kwargs: calls.append(kwargs) or True
    )
    main()
    assert calls[0]["resume"] is True
    assert calls[0]["scenario"] == "smuggler-cove"


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--scenario", "smuggler-cove", "--new-character"],
        ["--scenario", "smuggler-cove", "--replay", "run.jsonl"],
    ],
)
def test_resume_rejects_ambiguous_or_conflicting_options(flags):
    with pytest.raises(SystemExit):
        _parse_args(["--resume", *flags])
