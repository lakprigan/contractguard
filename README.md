# ContractGuard (Arc A)

Companion artifact to *Capability Minimization as a Safety Primitive: Risk-Aware
Causal Gating (RACG)*. This is **Paper 1** of a two-paper program:

> **The Gate Is Only as Honest as Its Contracts** — threat-modeling the contract
> layer of RACG, and defending it with **ContractGuard** (signed provenance,
> typed contract attestation, runtime effect verification).

Arc B (toolchain confusion + FlowGate) is seeded as future work in the paper.

## Thesis

RACG's structural injection defense ("if the dangerous tool isn't in the visible
set, a compliant agent can't call it") does not remove the trust assumption — it
**relocates** it into the integrity of the tool *contracts* the gate reads. An
attacker who poisons a contract makes the gate mis-decide **without persuading
the agent**. We attack the contract layer and defend it with ContractGuard.

Key finding: RACG has **two gates** (causal + admissibility) and the causal gate
dominates, so risk-relabeling / auth-aliasing alone *fail*. The load-bearing
trust is **effect integrity**, and the working attacks forge effects.

## Layout

```
main.tex                 # the paper (IEEEtran, compiles with references.bib)
references.bib           # shared bibliography (citation keys verified present)
results_contract.json    # measured results the paper's tables/figures come from
make_contract_figures.py # regenerates figures/ from results_contract.json
figures/
  ladder_isr.png         # strict necessity ladder (H6-H9)
  lambda_negative.png    # risk-knob negative result (H14a)
run_contract.py          # entry point -> prints tables, writes results_contract.json
run_llm.py               # real-LLM validation track -> writes results_llm.json
requirements.txt
riskgate/
  model.py               # tool contracts + provenance fields + auth-variable type
  registry.py            # 100-tool registry (with entitlements)
  filters.py             # RACG and baselines
  env.py                 # agent loop; MockAgent + LLMAgent; applies attack + ContractGuard
  attacks.py             # A1 forgery, A2 risk-downgrade, A3 auth-alias, A4 divergence
  contractguard.py       # 3-rung verifier (provenance / typed attestation / runtime)
  attack_scenarios.py    # S0-S3 scenarios isolating each rung
  runner_contract.py     # aggregates ISR, validates H6-H9 + H14a
  llm.py                 # provider abstraction (Anthropic / Bedrock / OpenAI-compat / stub)
  llm_runner.py          # validation grid: models x targets x phrasings x {L0,L3} x {A1,A4}
  tasks.py, runner.py    # carried over from the base benchmark
tests/
  test_contract_integrity.py   # structural tests for attacks + rungs
  test_contract_strength.py    # distributional / adaptive / overhead / field
  test_llm_validation.py       # offline (stub) tests for the validation track
```

## Reproduce

`results_contract.json` and `figures/` are generated artifacts (not tracked in
git); the commands below regenerate them locally from source.

```bash
pip install -r requirements.txt
python run_contract.py            # prints the ladder + H6-H9/H14a PASS/FAIL, writes JSON
python -m pytest -q tests         # structural unit tests
python make_contract_figures.py   # regenerate figures/ from results_contract.json
```

## Real-LLM validation track

The deterministic track above uses a worst-case adversarially-compliant
`MockAgent` and *upper-bounds* injection success. The validation track drives
*real* models through the identical env/filter/registry/attack/guard stack to
confirm the structural prediction on real agents: a contract attack re-exposes a
high-risk tool (attack-induced ISR > 0 at L0), and ContractGuard L3 removes that
exposure so the tool leaves V_t and **no** model can call it (attack-induced
ISR = 0 at L3), independent of injection phrasing.

Grid (a strict subset of the scripted suite): **models × 8 high-risk targets ×
3 phrasings × {L0, L3} × {A1, A4}**. ISR is *attack-induced* (baseline-subtracted,
matching the adaptive search), so a legitimately-authorizable target such as
`transfer_funds` is not miscounted as an attack.

```bash
# offline smoke test (no keys/network): a stub model emulating a susceptible agent
python run_llm.py --models stub

# real models (set provider creds first; any subset works)
python run_llm.py --models \
  bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0,\
anthropic:claude-3-5-haiku-latest,\
openai_compat:gpt-4o-mini
```

Providers that cannot be constructed (missing SDK/key) are recorded as
*skipped*, never crashing the sweep; the deterministic `MockAgent` reference is
always printed beside the model columns. Results are written to `results_llm.json`.

## Build the paper

```bash
pdflatex main.tex && bibtex main && \
pdflatex main.tex && pdflatex main.tex
```

## Results (all hypotheses PASS)

**Core ladder (hand-built scenarios):**

| Scenario | L0 | L1 | L2 | L3 | Defended by |
|----------|----|----|----|----|-------------|
| honest             | 0.00 | 0.00 | 0.00 | 0.00 | — |
| shortcut forgery   | 1.00 | **0.00** | 0.00 | 0.00 | provenance (rung 1) |
| signed over-scope  | 1.00 | 1.00 | **0.00** | 0.00 | typed attestation (rung 2) |
| effect divergence  | 1.00 | 1.00 | 1.00 | **0.00** | runtime check (rung 3) |

**Strengthened claims (10 hypotheses total):**
- **H6′ generality** — shortcut forgery: ISR=1.00 across all 8 core high-risk
  targets (n=24); negative controls (A2/A3 alone): 0.00 across all targets.
- **H_compound** — A1+A4 compound attack closed only by the full stack (L3).
- **H_adapt** — exhaustive white-box adaptive attacker: L0–L2 defeated, **L3
  worst-case attack-induced ISR = 0.00**. (This search also found and fixed a gap
  in a signer-only rung-2 design.)
- **H_field** — effect integrity is primary: only the produces+requires
  (causal-routing) perturbation reaches ISR=1.00; risk/auth alone reach 0.25.
- **H_util** — ContractGuard is free on honest contracts: success, auth-success,
  tokens, and tools-dropped unchanged at every level.
- **H14a** — no λ∈[0,100] removes a contract-forgery attack.
- **Soundness** — Propositions 1–3 (provenance / field-integrity / runtime) with
  proof sketches, conditional on a trusted attestation.

## Modules

```
riskgate/
  attacks.py            # A1/A2/A3/A4 poisoned-contract transforms
  contractguard.py      # 3-rung verifier (provenance / typed attestation / runtime)
  attack_scenarios.py   # hand-built S0-S3 (core ladder)
  attack_suite.py       # programmatic suite across all high-risk targets (#1,#4)
  adaptive.py           # exhaustive white-box adaptive attacker (#2)
  analysis_contract.py  # overhead/utility (#5) + field ablation (#8)
  runner_contract.py    # aggregates + validates all 10 hypotheses
  llm.py, llm_runner.py # real-LLM validation track (Sec. 7.1)
tests/
  test_contract_integrity.py   # core ladder + negatives
  test_contract_strength.py    # distributional / adaptive / overhead / field
  test_llm_validation.py       # offline (stub) validation-track tests
```
