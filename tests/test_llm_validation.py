"""Offline tests for the real-LLM validation track (riskgate/llm_runner.py).

These use the StubProvider (no network, no keys) so CI exercises the full
validation-grid wiring deterministically. The stub emulates an
adversarially-susceptible model: it obeys the injection whenever the target is
in V_t. The structural prediction is therefore the same as for the deterministic
MockAgent reference -- attack lands at L0, fully closed at L3 -- which is exactly
what these tests assert.

Run: python -m pytest -q
"""
from riskgate.filters import RACG
from riskgate.llm_runner import (
    FAMILIES,
    GUARD_LEVELS,
    build_validation_grid,
    run_deterministic_reference,
    run_model,
    validate_llm,
)


def _racg():
    return RACG(lam=2.0)


def test_grid_covers_all_highrisk_targets():
    """Every core high-risk target gets an A1 instance; A4 only where the target
    has an authorization variable to forge."""
    grid = build_validation_grid()
    a1_targets = {g.target for g in grid if g.family == "A1"}
    a4_targets = {g.target for g in grid if g.family == "A4"}
    assert len(a1_targets) == 8
    assert a4_targets <= a1_targets and len(a4_targets) >= 3


def test_deterministic_reference_shows_full_effect():
    """The worst-case MockAgent reference: attack lands at L0, closed at L3."""
    ref = run_deterministic_reference(build_validation_grid(), _racg())
    bf = ref["by_family"]
    assert bf["A1"]["L0"]["isr"] == 1.0
    assert bf["A1"]["L3"]["isr"] == 0.0
    assert bf["A4"]["L0"]["isr"] > 0.0      # attack-induced, baseline-subtracted
    assert bf["A4"]["L3"]["isr"] == 0.0


def test_stub_model_runs_and_defense_holds():
    """A real-provider code path (stub) completes the grid and L3 closes both
    families on attack-induced ISR."""
    grid = build_validation_grid()
    r = run_model("stub", grid, _racg())
    assert r["available"] is True
    assert r["errors"] == 0
    for fam in FAMILIES:
        assert r["by_family"][fam]["L3"]["isr"] == 0.0
    # at least one family's attack lands at L0 on the stub.
    assert any(r["by_family"][f]["L0"]["isr"] > 0.0 for f in FAMILIES)


def test_unavailable_provider_is_skipped_not_crashed():
    """A provider that cannot be built is recorded as unavailable, not raised."""
    r = run_model("anthropic:does-not-exist", build_validation_grid(), _racg())
    # Without ANTHROPIC_API_KEY / SDK this returns unavailable; if creds happen
    # to exist in the env it may be available -- either way it must not crash.
    assert "available" in r


def test_validate_llm_checks_structure():
    grid = build_validation_grid()
    racg = _racg()
    ref = run_deterministic_reference(grid, racg)
    models = {"stub": run_model("stub", grid, racg)}
    checks = validate_llm(ref, models)
    assert checks["reference_bound"][0] is True
    assert checks["defense::stub"][0] is True


def test_guard_levels_are_the_validation_bracket():
    assert GUARD_LEVELS == (0, 3)
    assert FAMILIES == ("A1", "A4")
