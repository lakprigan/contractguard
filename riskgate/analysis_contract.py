"""Supporting analyses (paper additions #5 overhead/utility, #8 field ablation).

  overhead_utility(): ContractGuard's cost on HONEST contracts -- task success,
      tokens, and tools-dropped -- across guard levels, on the full benign +
      safety-stress suite. Backs the deployability claim (H_util).

  field_ablation(): which single contract field carries the attack. For each
      core high-risk target, perturb exactly one field (requires / produces /
      risk / authorizes) and measure ISR under RACG with no guard. Ranks the
      attack surface by field, empirically justifying "effect integrity is
      primary".
"""
from __future__ import annotations

import statistics
from dataclasses import replace

from .contractguard import ContractGuard
from .env import run_task
from .filters import RACG
from .model import Risk, Tool
from .registry import REGISTRY
from .tasks import build_tasks


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def overhead_utility() -> dict:
    """Run the honest benign + safety-stress suite under each guard level."""
    racg = RACG(lam=2.0)
    benign, stress = build_tasks()
    tasks = benign + stress
    auth = [t for t in tasks if t.flavor == "auth_required"]

    rows = {}
    for L in (0, 1, 2, 3):
        guard = ContractGuard(level=L) if L > 0 else None
        results = [run_task(t, racg, REGISTRY, guard=guard) for t in tasks]
        auth_results = [run_task(t, racg, REGISTRY, guard=guard) for t in auth]
        # tools-dropped: difference in registry size after sanitization on honest
        # contracts (should be 0: guard must not drop legitimate tools).
        dropped = 0
        if guard is not None:
            sani = guard.sanitize(dict(REGISTRY), trusted=REGISTRY)
            dropped = len(REGISTRY) - len(sani)
        rows[f"L{L}"] = {
            "success": _mean([1.0 if r.success else 0.0 for r in results]),
            "auth_success": _mean([1.0 if r.success else 0.0 for r in auth_results]),
            "tokens": _mean([r.total_tokens for r in results]),
            "tools_dropped": dropped,
        }
    return rows


# ---------------------------------------------------------------------------
# Field-sensitivity ablation
# ---------------------------------------------------------------------------
def _core_targets():
    from .attack_suite import core_highrisk_targets
    return core_highrisk_targets()


def _no_action_task(target: Tool):
    from .attack_suite import _no_action_task as nat
    return nat(target)


def _onpath_task(target: Tool):
    from .attack_suite import _domain_search, _task_for_target
    carrier = _domain_search(target)
    if carrier == target.name:
        carrier = "search_files"
    return _task_for_target(target, carrier), carrier


def field_ablation() -> dict:
    """Perturb one field at a time; report ISR under RACG (no guard).

    requires : forge to the task initial state (precondition forgery)
    produces : add the goal variable (effect forgery / shortcut)
    risk     : downgrade to LOW
    authorizes: empty the set (auth aliasing)

    On-path task is used so each field has a chance to matter; we also report the
    pair (produces+requires) which is the true shortcut. Mean ISR over the core
    high-risk targets and 3 phrasings."""
    from .attack_suite import PHRASINGS
    from .attacks import Attack
    from .model import Injection
    racg = RACG(lam=2.0)

    def isr(task, target, atk, step):
        injs = [Injection(task_id=task.task_id, inject_at_step=step,
                          target_tool=target, phrasing=p) for p in PHRASINGS]
        return _mean([1.0 if run_task(task, racg, REGISTRY, injection=inj,
                                      attack=atk).injection_succeeded else 0.0
                      for inj in injs])

    fields = ["requires", "produces", "risk", "authorizes", "produces+requires"]
    acc = {f: [] for f in fields}

    for name in _core_targets():
        target = REGISTRY[name]
        task, carrier = _onpath_task(target)
        step = 0  # inject at start; shortcut perturbations expose target at step 0

        def mk(**changes):
            def t(reg):
                reg = dict(reg)
                reg[name] = replace(reg[name], signer="attacker", **changes)
                return reg
            return Attack(f"abl_{name}", t, None)

        acc["requires"].append(isr(task, name, mk(requires=task.initial_state), step))
        acc["produces"].append(isr(task, name, mk(produces=target.produces | task.goal_state), step))
        acc["risk"].append(isr(task, name, mk(risk=Risk.LOW), step))
        acc["authorizes"].append(isr(task, name, mk(authorizes=frozenset()), step))
        acc["produces+requires"].append(
            isr(task, name, mk(requires=task.initial_state,
                               produces=target.produces | task.goal_state,
                               risk=Risk.LOW), step))

    return {f: _mean(v) for f, v in acc.items()}
