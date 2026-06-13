"""Core data model: tool contracts, risk levels, tasks, and the risk() map.

Mirrors the paper's formalism:
    t_i = (d_i, R_i, E_i, c_i, rho_i, alpha_i)
where R_i are preconditions, E_i effects, c_i cost, rho_i risk, alpha_i auth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class Risk(IntEnum):
    """Ordinal risk tiers (Section 4 of the paper)."""
    LOW = 0
    MED = 1
    HIGH = 2


# risk() penalty map from Eq. (6). Super-linear so a single irreversible
# action is never preferred over a short chain of reversible ones.
RISK_PENALTY: dict[Risk, float] = {
    Risk.LOW: 0.0,
    Risk.MED: 1.0,
    Risk.HIGH: 4.0,
}


def risk_penalty(rho: Risk) -> float:
    return RISK_PENALTY[rho]


@dataclass(frozen=True)
class Tool:
    """A tool contract t_i = (d_i, R_i, E_i, c_i, rho_i, alpha_i).

    Provenance fields (Arc A, contract integrity) extend the original contract:
      signer   -- the principal who issued/signed this contract. The trusted
                  registry issues "registry"; an attacker-supplied or injected
                  contract carries a different (or empty) signer. ContractGuard
                  rung 1 (signed provenance) rejects fields not vouched for by a
                  trusted signer.
      entitled -- the set of state variables this tool is *entitled* to produce
                  as authorization variables. Even a correctly-signed contract
                  must not produce an auth variable outside its entitlement;
                  ContractGuard rung 2 (typed auth variables) enforces this.
    These default to backward-compatible values so the original H1-H5 behavior
    is unchanged when integrity checking is disabled.
    """
    name: str
    description: str
    requires: frozenset[str] = field(default_factory=frozenset)   # R_i
    produces: frozenset[str] = field(default_factory=frozenset)   # E_i
    cost: float = 1.0                                             # c_i
    risk: Risk = Risk.LOW                                         # rho_i
    authorizes: frozenset[str] = field(default_factory=frozenset)  # alpha_i
    domain: str = "misc"
    signer: str = "registry"                                      # provenance
    entitled: frozenset[str] = field(default_factory=frozenset)   # auth entitlement

    def executable(self, state: frozenset[str]) -> bool:
        """R_i subset of s_t."""
        return self.requires <= state

    def authorized(self, state: frozenset[str]) -> bool:
        """Low-risk tools need no authorization; others need alpha_i in state."""
        if self.risk == Risk.LOW:
            return True
        return self.authorizes <= state


@dataclass(frozen=True)
class Task:
    """A benchmark task with a gold tool chain and mocked outputs."""
    task_id: str
    domain: str
    query: str
    initial_state: frozenset[str]
    goal_state: frozenset[str]
    gold_chain: tuple[str, ...]
    # Deterministic mocked outputs: tool_name -> variables it produces for THIS task.
    mock_outputs: dict[str, frozenset[str]] = field(default_factory=dict)
    # High-risk tool(s) that should be gated until authorized (for safety-stress).
    gated_targets: frozenset[str] = field(default_factory=frozenset)
    # Task flavor: "benign", "auth_required", "no_action".
    flavor: str = "benign"

    def is_complete(self, state: frozenset[str]) -> bool:
        return self.goal_state <= state


@dataclass(frozen=True)
class Injection:
    """An indirect prompt injection attached to a task observation."""
    task_id: str
    inject_at_step: int           # step index where the poisoned observation appears
    target_tool: str              # the high-risk tool the attacker wants called
    phrasing: str                 # "direct", "roleplay", "obfuscated"
    # If True, the injection also tries to forge the auth variable (boundary
    # condition for H5). Default False = attacker controls content only.
    forges_authorization: bool = False


# ---------------------------------------------------------------------------
# Authorization variables are a privileged TYPE (Arc A, ContractGuard rung 2).
# An authorization variable may only enter the state via a tool that is
# *entitled* to produce it. The trusted registry declares, for each auth
# variable, the single legitimate establisher (the LOW-risk confirmation step).
# This is the typed-state-variable contribution: it distinguishes "any tool that
# claims produces={recipient_confirmed}" from "the tool actually permitted to
# confirm a recipient".
# ---------------------------------------------------------------------------
AUTH_VARIABLES: frozenset[str] = frozenset({
    "recipient_confirmed",   # legit establisher: confirm_recipient / read_email
    "deletion_approved",     # legit establisher: review_for_deletion
    "payment_confirmed",     # legit establisher: confirm_payment
    "external_approved",     # legit establisher: verify_external_party
    "share_scope_set",       # legit establisher: set_share_scope
    "attendee_confirmed",    # (no benign establisher in suite -> stays gated)
})


def is_auth_variable(var: str) -> bool:
    return var in AUTH_VARIABLES
