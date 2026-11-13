# Phase 2C decision and immutable evidence boundary

2026-11-13. The primary question remains selective adaptation versus conventional
full fine-tuning with matched total target-label access. No architecture search,
PTB-XL score maximization or performance-based domain selection occurred.

## A. Scientific pilot feasibility

**READY FOR HUMAN PROTOCOL REVIEW.**

The original within-domain greedy allocation failure is diagnosed and reproduced
from metadata. True source/target age-domain differences remain:161 versus585
joint positives. Canonical population16,415 and age boundary60 are unchanged.
One prospective joint-pattern quota candidate, seed44301, gives source
train/stop joint positives137/24 and target A293, removing the structural empty
reservoir without oversampling or changing domains. Both G draws and capped
source analyses meet A marginal and B four-cell floors. No seed search or
label-informed G enrichment was performed.

This warrants review of the **same33-fit pilot with B retained**. It does not
prove valid representation geometry, a useful mask or a predictive improvement.
B can still abstain from weak geometry or finite bootstrap evidence. For a random
G256 from the candidate reservoir, joint-cell support risk alone is about1.872%;
the actual fixed draws passed with17/21 joint positives. No model was used to
select the split. Primary A remains meaningful if B later abstains; no A-only
budget change is proposed or silently implemented.

Reviewable limitations:625 patients have differing serial CD/STTC annotations,
10 cross the age boundary across serial eligible records; canonical-record
selection handles grouping, not clinical adjudication. Unknown clinical absence
is not magically verified by recorded tag zeros. This is a two-label annotation
benchmark, not external clinical validation. The replacement O was informed by
metadata feasibility and dataset reuse, so it cannot be described as independent
confirmation or historically untouched.

## B. Execution readiness

**READY WITH CONDITIONS.**

The metadata generator, versioned lineage, aggregate-only auditor/worker boundary,
immutable candidate/context, deny-by-default accesses and zero-attempt checks
pass synthetic verification. The new context is **DRAFT,0 attempts**, no training
or evaluation authority. The old real Phase2B ledger stays **draft,0/36**, with its
original blockers and no accesses. Its old runner deliberately cannot consume
the new split without reviewed, version-specific integration. That refusal is a
safeguard, not permission to substitute the candidate into the old namespace.

Before execution:

1. Human review/adoption of [amendment](SPLIT_AMENDMENT_PROPOSAL.md), exact seed/
   grouping/endpoint, changed O history, all support tables and unchanged33+3
   budget. Explicitly acknowledge that the underlying source/target task is an
   internal recorded-annotation age-shift study.
2. Review the outstanding Phase2B draft clarifications: normalized objective
   history (now superseded by quotas),20 post-update stopping looks plus eligible
   epoch zero, and calibration-failure threshold convention. No draft was adopted
   by writing these reports or passing tests.
3. Prepare a new real-execution namespace/ledger and role-specific loader bound
   to this exact approved version, with source-from-scratch and normalization
   lineage, update/label budgets, durable attempts and one-time O lock. Rehearse
   that authorized integration synthetically; do not clear/reuse the old ledger.
4. Separately authorize real source training/adaptation/calibration. Any real
   smoke/failure/retry consumes the shared36 ceiling. Full-cost gets F∪G plus
   E/C, matching A's640/1,024 target-label unions. No favorable reruns or extra
   architectures/source seeds are authorized by reserve availability.
5. Only after all development/checkpoint/calibration artifacts are locked,
   separately authorize one new internal O evaluation. Never reopen fold10.

External clinical transfer remains blocked by its separate annotation/person/
overlap evidence gates. Synthetic engineering, future internal pilot findings
and independent external validation are three different evidence levels.

## Files and verification

Created this sprint (13 Git-visible files, all new):

- `src/dmvmat_ecg/phase2c/__init__.py`
- `src/dmvmat_ecg/phase2c/splits.py`
- `src/dmvmat_ecg/phase2c/auditor.py`
- `src/dmvmat_ecg/phase2c/diagnostics.py`
- `tests/phase2c/test_split_amendment.py`
- `configs/phase2c/joint_pattern_v1.json`
- `scripts/phase2c_preflight.py`
- `results/phase2c/METADATA_PREFLIGHT.json`
- [Root cause](SPLIT_FAILURE_ROOT_CAUSE.md)
- [Joint-pattern feasibility](JOINT_PATTERN_FEASIBILITY.md)
- [Amendment proposal](SPLIT_AMENDMENT_PROPOSAL.md)
- [Revised preflight](REVISED_PILOT_PREFLIGHT.md)
- This decision report.

Private/ignored: one prospective generation receipt, full auditor manifest,
zero-attempt draft context, source/config integrity inventory and an aggregate-only
worker summary in `private/phase2c_joint_pattern_v1/`. They contain no real
learned models/representations/predictions. Only auditor manifest contains
record/patient assignments and outcomes; none is published. Temporary synthetic
artifacts live under pytest namespaces; the old Phase2B DB is not used for them.

Tests: **174 passed**, including32 new Phase2C cases,0 failed/skipped,
7 unchanged warnings. Full Ruff/format checks pass. See
[revised preflight](REVISED_PILOT_PREFLIGHT.md) for exact commands, initial test
correction and limitations. Source/config/dependency/role/candidate hashes are
in the [metadata receipt](../../results/phase2c/METADATA_PREFLIGHT.json); full bindings are private.

Final preservation check: all131 files in the starting public/source inventory
remain byte-identical, including existing uncommitted Phase2A/B work; all48
historical lock artifacts and16 source best/final checkpoints match stored
hashes. Original Phase2B candidate/role vault/context/SQLite bytes remain unchanged;
old pilot ledger has0 attempts/accesses. No previous file was overwritten or
removed. Other repositories were not touched.

Git: branch `main`, HEAD `1f0ff9031e9e1e2adc97962922fd97705570baf2` unchanged;
tracked diff empty; prior untracked work and13 new paths remain untracked and
visible (not ignored). No dependencies/old selectors/models/results/checkpoints
were changed. No real predictor, real preprocessing, inference, calibration or
holdout evaluation was executed; no authorization activated, no commit/push/
merge/PR. Stop for review and separate authorization.
