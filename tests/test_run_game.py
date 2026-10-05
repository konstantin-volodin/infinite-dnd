import pytest

from run_game import _parse_args, main


@pytest.mark.parametrize("flag", ["--turn", "--turns"])
def test_turn_flag_aliases_set_max_turns(flag):
    args = _parse_args([flag, "10"])

    assert args.turns == 10


def test_main_exits_unsuccessfully_when_campaign_run_fails(monkeypatch):
    monkeypatch.setattr("sys.argv", ["infinite-dnd", "--turn", "1"])
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    monkeypatch.setattr("run_game.run_game", lambda **_kwargs: False)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
