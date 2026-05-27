from app import db


def test_init_creates_tables():
    db.init_db()
    runs = db.list_scenario_runs()
    assert runs == []


def test_save_and_load_scenario_run():
    db.save_scenario_run(
        run_id="run_a",
        seed="seed",
        scenario_mode="base_case",
        scenario_title="A title",
        full_json={"foo": "bar"},
    )
    runs = db.list_scenario_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_a"

    payload = db.load_scenario_run("run_a")
    assert payload == {"foo": "bar"}

    missing = db.load_scenario_run("nope")
    assert missing is None


def test_cache_set_and_get():
    db.cache_set("k1", "model", "agent", {"text": "hello"})
    assert db.cache_get("k1") == {"text": "hello"}
    assert db.cache_get("missing") is None
