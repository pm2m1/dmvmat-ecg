# Phase3B locked evaluation — BLOCKED outcome-decoding failure

This is the authoritative status notice for this evaluation. It supersedes any
suggestion of successful scientific completion from the automatic `COMPLETE.json`
marker or `LOCKED_EVALUATION_COMPLETE` process log. The process finished, but the
requested evaluation did **not** produce interpretable performance results.
The generated `PHASE3B_LOCKED_EVALUATION_RESULTS.md`, its NA tables and figures
are retained unchanged for audit; they are not evidence about any method's efficacy.

## Observed execution

- All 295 frozen artifacts, the lock, memberships, mapping, base/real code identities
  and authorized commit passed the pre-access and post-access checks.
- All 138 registered procedures were predicted, using 121 physical inference runs.
  No model fitting, normalization fitting, calibration, threshold selection or
  model/mask/checkpoint change occurred.
- O: 3,860 records, zero short-record exclusions, 2,436 first-10-second truncations.
- Complete prediction seal:
  `b7c7e795b9641979858c5050bd6f7db9c2f1340a1f466b4828a0e4088eec5cde`.
- The seal preceded the single outcome-access marker. All prediction hashes still match.
- **Exactly one O outcome access was consumed.** Re-entry is rejected before the
  outcome loader runs. O headers/outcomes have not been reread and no retry occurred.
- The decoded observation masks contained zero observed labels for both labels:
  3,860 unknowns each. Thus all requested AP, AUROC, Brier, log-loss, ECE and
  threshold metrics are undefined. All 10,000 primary bootstrap replicates are
  undefined; there is no primary CI or substantive superiority/inconclusive-effect finding.
  This is an execution failure, not a scientific negative result.

## Verified parser defect; remaining uncertainty

`src/dmvmat_ecg/phase3b_eval/signals.py:73` requires a raw line to start with
`#dx:` after trimming only its outer whitespace. A fictional regression probe
proved that `#Dx: 1,2` parses while the valid comment form `# Dx: 1,2` becomes
unknown. The existing development adapter instead uses WFDB's decoded comments
(`src/dmvmat_ecg/phase3/signals.py:54` onward), which strips the comment prefix.
The new raw-header decoder failed to preserve that behavior.

This defect plausibly explains the all-unknown result. The actual O header
format has **not** been reread to confirm the cause, as that would exceed the
single-use authorization. Unknowns were not converted to negatives. No correction
to the decoder, masks, outcomes, metrics, predictions or evaluator code has been
applied after access. The evaluator also failed to stop on wholly unobserved
outcomes and emitted a process-completion marker; that marker is not a validity check.

## Tests and integrity

Before inference: `pytest -q`: 310 passed, seven warnings, 52.41s.
After inference: 310 passed, seven warnings, 54.56s.
`ruff check src tests scripts`: passed.
`ruff format --check src tests scripts`: 117 files already formatted.
The passing suite did not cover whitespace after the WFDB comment marker and
therefore did not establish real outcome-decoder correctness. The separate
fictional probe exposed the gap; tests must be expanded before any corrected run.

The entire attempt and alias tables still exactly match the locked snapshot:
121/144 actual attempts, 138 logical slots; no reserve use or new training.
The original development SQLite evaluation row remains frozen and unmodified;
this separately versioned evaluator records its one-time access in canonical
`O_EVAL_BEGIN_V1.json` / `O_OUTCOME_ACCESS_V1.json` parent markers and the private
`evaluation_v1/` mirror receipts. These markers remain consumed, not reset.
Frozen evaluator hash:
`0b6f06c989478fd8866658bd9c056347f60195b324e2fd5f5d8f3ae9650db358`.
No Phase2 artifacts, PTB-XL fold10 or CPSC FEAS data were accessed by this run.

## Repository and files

Branch `phase3-cross-hospital`, HEAD
`7e93ed3a7a00b603d34d190843bcfbfc1753db1e`.
No tracked-file edits, commits, pushes, merges or PRs.
New evaluator files: `src/dmvmat_ecg/phase3b_eval/{__init__,access,signals,provider,
statistics,analysis,report,run}.py`; synthetic tests:
`tests/phase3b_eval/test_evaluation.py`. New outputs include the requested report,
this blocker notice, aggregate-only `results/phase3b_eval/` records/figures and
private authorization, prediction, seal, access, decoded-outcome and completion
artifacts. Private raw/prediction arrays remain ignored. Pre-existing untracked
Phase3B implementation/results/reports were preserved. The exact new-file
inventory and Git status are recorded in `results/phase3b_eval/FINAL_VERIFICATION_RECEIPT.json`.

## Required next human decision

No automatic retry is permitted. A separately authorized corrective audit would
need to approve synthetic decoder fixes, whitespace/annotation-mask regression
tests and an explicit decision about any further outcome access using the same
immutable prediction seal. The current cached decoded outcomes are all unknown
and cannot recover the diagnosis information. Models must not be regenerated,
selected or changed. No such correction or further access is undertaken here.

**BLOCKED — outcome decoding produced no observed labels; one-time access consumed.**
