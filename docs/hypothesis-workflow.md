# Hypothesis Workflow

Every experiment in this lab follows a strict hypothesis-driven workflow. This is not optional.

## The Three Files

Each experiment folder contains exactly three artifacts:

```
experiments/exp_NNN_<name>/
├── hypothesis.md   ← Write BEFORE running
├── run.py          ← The experiment script
└── results.md      ← Fill AFTER running
```

## Step 1 — Write the Hypothesis

Open `hypothesis.md` and fill in all sections:

```markdown
# Hypothesis — EXP 003

**Date:** 2026-06-16
**Author:** your-name

## What I expect to happen
Ring entanglement will converge faster than chain entanglement because
cyclic dependencies create richer qubit correlations.

## Why I expect this
Ring topology connects the last qubit back to the first, potentially
allowing information to flow in both directions.

## What would prove me wrong
If ring entanglement accuracy is >5% lower than chain after 50 epochs,
or if training time is >2x longer.

## Metrics I will measure
- [ ] Final accuracy per entanglement type
- [ ] Convergence epoch
- [ ] Training time
```

**The Cursor agent will refuse to help implement `run.py` until `hypothesis.md` has real content (not placeholders).**

## Step 2 — Run the Experiment

```bash
python experiments/exp_003_entanglement_effect/run.py
```

Results are automatically appended to `logs/experiments.jsonl`.

## Step 3 — Analyze and Document

1. Open the dashboard: `streamlit run dashboard/app.py`
2. Compare learning curves and final metrics
3. Fill in `results.md`:

```markdown
# Results — EXP 003

## What happened
Chain entanglement reached 92% accuracy; ring reached 89%; none reached 85%.

## Comparison with hypothesis
Partially correct — chain beat ring, contrary to my expectation.
Ring may add gradient noise without proportional representational gain.

## Unexpected finding
No-entanglement model was only 7% behind chain, suggesting
entanglement matters less than expected for this 2D task.

## Suggested next experiment
EXP 006 — test if barren plateau explains ring's slower convergence.
```

## Working with Cursor

### Before an experiment
> "I'm running EXP 003. My hypothesis is that ring entanglement converges faster.
> What might I be ignoring? What additional metric should I track?"

### During implementation
> "Is this quantum circuit correct? Is there barren plateau risk with 6 qubits and 4 layers?"

### After results
> "quantum_6q_3l converged slower than classical_32. Here are the logs: [paste].
> Why might this have happened? What should I test next?"

## Log Discipline

- `logs/experiments.jsonl` is **append-only**
- Never delete or overwrite log entries
- Each line is a self-contained JSON record with full training history
- The dashboard reads this file directly — no database needed
