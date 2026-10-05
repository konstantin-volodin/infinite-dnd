from src.agents.character.tools import Action, Attack, Check, Speak, Travel, Wait


def test_speak():
    s = Speak(actor="alice", message="hi", target="bob")
    assert s.kind == "speak" and s.actor == "alice" and s.target == "bob"


def test_travel():
    t = Travel(actor="alice", destination="forest")
    assert t.kind == "travel" and t.destination == "forest"


def test_wait():
    w = Wait(actor="alice")
    assert w.kind == "wait"


def test_action():
    a = Action(actor="alice", description="pick the lock")
    assert a.kind == "action" and a.target is None


def test_attack():
    k = Attack(actor="alice", target="bob")
    assert k.kind == "attack" and k.target == "bob"


def test_check():
    check = Check(actor="alice", ability="dexterity", description="pick the lock", difficulty=14)
    assert check.kind == "check" and check.difficulty == 14


def test_roundtrip_serialization():
    s = Speak(actor="alice", message="hi", target="bob")
    t = Travel(actor="alice", destination="forest")
    w = Wait(actor="alice")
    a = Action(actor="alice", description="pick the lock")
    k = Attack(actor="alice", target="bob")
    check = Check(actor="alice", ability="wisdom", description="spot the ambush", difficulty=12)
    for tool in (s, t, w, a, k, check):
        clone = type(tool).model_validate_json(tool.model_dump_json())
        assert clone == tool


def test_remember_and_new_goal_optional_defaults():
    for tool in (Speak(actor="alice", message="hi"), Travel(actor="alice", destination="forest"), Wait(actor="alice"), Action(actor="alice", description="look around"), Attack(actor="alice", target="bob")):
        assert tool.remember is None
        assert tool.new_goal is None


def test_remember_and_new_goal_settable_on_any_tool():
    w = Wait(actor="alice", remember="the door was unlocked", new_goal="find the missing key")
    assert w.remember == "the door was unlocked"
    assert w.new_goal == "find the missing key"
    clone = Wait.model_validate_json(w.model_dump_json())
    assert clone == w
