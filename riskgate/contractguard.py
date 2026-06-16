"""ContractGuard: a verifier between the registry and the gate (Arc A defense).

ContractGuard makes the contract layer's trust explicit and checkable. It sits
between the (possibly poisoned) registry and the filter, and returns a sanitized
registry the gate can safely read. Three mechanisms compose as a STRICT
necessity ladder (each necessary, the prior insufficient):

  rung 1  signed provenance       : reject any contract field not vouched for by
                                     a trusted signer. Falls back to the trusted
                                     contract for any tool whose signer is not in
                                     `trusted_signers`. Defeats declaration-time
                                     forgery where the attacker re-signs (A1-A3).
  rung 2  typed contract attestation : for a tool present in the trusted
                                     attestation that claims a trusted signer,
                                     verify its integrity-critical, gate-relevant
                                     fields (requires, produces, risk, authorizes)
                                     against the trusted reference and restore any
                                     that diverge. Subsumes authorization-variable
                                     typing (an over-claimed auth variable in
                                     `produces` is removed because `produces` is
                                     restored to the attested value) and also
                                     catches same-signer tampering with requires/
                                     risk/authorizes. Defeats a signed-but-
                                     overscoped contract that rung 1 waves through.
  rung 3  runtime effect checking : at execution, compare a tool's realized
                                     output against its declared `produces`; an
                                     undeclared effect is dropped (and flagged).
                                     Defeats execution-time divergence (A4) that
                                     both static rungs miss.

Enabling rungs cumulatively (rung<=1, <=2, <=3) gives the ablation that backs
H7 < H8 < H9.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .model import AUTH_VARIABLES, Tool


@dataclass
class ContractGuard:
    """A cumulative contract verifier. `level` selects how many rungs are on.

    level 0 -> no defense (passthrough; used to reproduce the vulnerability)
    level 1 -> signed provenance
    level 2 -> + typed contract attestation
    level 3 -> + runtime effect verification
    """
    level: int = 3
    trusted_signers: frozenset[str] = frozenset({"registry"})

    def __post_init__(self):
        self.name = f"contractguard_L{self.level}"
        # populated during a run for diagnostics / tests
        self.dropped_effects: dict[str, set[str]] = {}

    # -- static sanitization (rungs 1 & 2), applied before the filter ---------
    def sanitize(self, registry: dict[str, Tool],
                 trusted: dict[str, Tool]) -> dict[str, Tool]:
        """Return a sanitized copy of `registry`.

        `trusted` is the ground-truth registry the verifier compares against
        (models a signed contract store / attestation the attacker cannot forge).
        """
        if self.level <= 0:
            return registry
        out: dict[str, Tool] = {}
        for name, tool in registry.items():
            ref = trusted.get(name)
            fixed = tool

            # rung 1: signed PROVENANCE (origin only). Reject contracts whose
            # signer is not trusted; drop tools absent from the attestation. This
            # stops an attacker who introduces a NEW tool or RE-SIGNS a contract
            # under their own identity (e.g. the shortcut-forgery attack, which
            # re-signs the target as "attacker"). It does NOT, by itself, verify
            # the field values of a contract that claims a trusted signer -- that
            # is rung 2's job. This separation is what makes the ladder strict:
            # provenance alone is necessary but insufficient.
            if self.level >= 1:
                if tool.signer not in self.trusted_signers:
                    if ref is None:
                        continue  # unknown, untrusted contract: refuse it
                    fixed = ref   # restore the attested contract for a known tool

            # rung 2: TYPED FIELD INTEGRITY. For a tool that passed provenance
            # (claims a trusted signer) but is present in the attestation, verify
            # its integrity-critical, gate-relevant fields against the trusted
            # reference and restore any that diverge: preconditions (requires),
            # produced effects (produces), risk tier, and authorization set. This
            # subsumes authorization-variable typing (an over-claimed auth
            # variable in produces is removed because produces is restored to the
            # attested value) and additionally catches same-signer tampering with
            # requires/risk/authorizes that the adaptive attacker exploits. A tool
            # absent from the attestation keeps its declared fields (it has no
            # reference to compare against; rung 1 already refused untrusted ones).
            if self.level >= 2 and ref is not None:
                fixed = replace(
                    fixed,
                    requires=ref.requires,
                    produces=ref.produces,
                    risk=ref.risk,
                    authorizes=ref.authorizes,
                    entitled=ref.entitled,
                )

            # rung 2: typed auth variables. Even a trusted-signed contract may
            # only produce auth variables it is entitled to. Strip the rest.
            if self.level >= 2:
                entitled = (ref.entitled if ref is not None else fixed.entitled)
                illegal = (fixed.produces & AUTH_VARIABLES) - entitled
                if illegal:
                    fixed = replace(fixed, produces=fixed.produces - illegal)

            out[name] = fixed
        return out

    # -- dynamic verification (rung 3), applied to realized effects -----------
    def verify_effect(self, tool: Tool, realized: frozenset[str]) -> frozenset[str]:
        """Drop any realized effect the tool did not DECLARE in `produces`.

        Below level 3 this is a passthrough. At level 3 it enforces that runtime
        output cannot exceed the (sanitized, declared) contract -- catching A4
        effect-divergence where a benign-looking tool emits an undeclared auth
        variable at execution time.
        """
        if self.level < 3:
            return realized
        undeclared = realized - tool.produces
        if undeclared:
            self.dropped_effects.setdefault(tool.name, set()).update(undeclared)
        return realized & tool.produces


def guard_ladder() -> list[ContractGuard]:
    """The cumulative ablation: no-defense, then rungs 1, 2, 3."""
    return [ContractGuard(level=l) for l in (0, 1, 2, 3)]
