# Phase 2A selector implementation and synthetic verification

**2026-11-12. Engineering core READY within synthetic-only scope. No real transfer
fit, source pretraining, ECG inference or final evaluation was performed.**

Baseline: branch `main`, commit `1f0ff9031e9e1e2adc97962922fd97705570baf2`.
The six preceding plans and completed ECG reports/code were read before changes.
Initial user work was the six untracked Phase 2 planning documents. No tracked
baseline source, configuration, dependency lock, result, checkpoint or report
was edited. New implementation lives under a separate namespace.

## 1. What was implemented

| File | Implemented behavior |
|---|---|
| [contracts.py](../../src/dmvmat_ecg/phase2/contracts.py) | Explicit observed-label masks; ordered label schemas; immutable copied synthetic inputs; role/group isolation; membership/content hashes; per-operation/role access accounting; source/target label unions; source-history leakage rejection |
| [evidence.py](../../src/dmvmat_ecg/phase2/evidence.py) | A marginal signed retention; B joint-label-standardized retention; shared patient bootstrap; source anchor/class caps; Gaussian multikernel MMD with permutation evidence; Holm adjustment; seven SHARE/ADAPT votes with abstention |
| [adaptation.py](../../src/dmvmat_ecg/phase2/adaptation.py) | Source cloning; exact owner partition validation; persistent frozen BN/parameters; bounded CPU synthetic adaptation; masked BCE; exact macro-AP checkpoint selection with epoch zero; random-mask matching; hash-checked weight serialization; role-bound comparator interfaces |
| [synthetic_selector.json](../../configs/phase2/synthetic_selector.json) | Exact method defaults and explicit `real_execution_authorized=false`; CPU, no alignment loss |

`__init__.py` identifies the versioned namespace. No imports discover/load real
datasets or old checkpoints; no real-data CLI is provided. In-memory synthetic
namespaces are a programming contract, **not authentication or proof of origin**
against arbitrary malicious Python callers. Future real execution needs a trusted
data adapter and durable provenance/authorization at the file-access boundary.

## 2. Scientific provenance and numerical contract

The original authors' released implementation uses Jensen–Shannon divergence.
Our MMD/permutation/signed-geometry logical selector is later re-engineering, not
an equivalent reproduction. See the pinned author/re-engineered references in
[the method](MULTILABEL_DMV_MAT_METHOD.md). No unlicensed author code was copied.
The unchanged legacy ECG prototype is still
`src/dmvmat_ecg/transfer/interface.py`; the new implementation does not change it.

Implemented defaults match the draft: four Gaussian bandwidths .5/1/2/4 after
label-blind pooled scalar RMS; biased MMD² including diagonals; distinct-patient
hash caps256/domain;999 permutations with plus-one p-value; seven-test Holm
adjustment with invalid tests treated as p=1 but still flagged invalid. Zero
radius (≤1e-8), too few patients or a numerically degenerate null (std≤1e-12)
is not positive evidence. No MMD term enters the BCE training loss.

A uses equal-class mean Euclidean scatter and source-direction signed retention.
B uses equal weights across the other label's two strata and is deliberately
restricted to **two labels**. Source analysis caps each label/class at256;
bootstrap uses their patient union with each label's own analysis mask, restoring
the other label's verified status for B. This avoids silently exceeding class
caps or interpreting an independently subsampled other-label mask as missing
clinical information. Counts and source anchors are included in receipts.

Source and target patients are independently bootstrapped, with the **same
within-domain multiplicities shared across all labels/modules**, preserving
co-occurrence.4,999 draws, min10 observed patients/class or B joint cell,
source D≥0.1, fixed directions/scatters for conditional class-mean uncertainty.
Invalid source denominators ≤0.02 or empty required cells in >5% of draws abstain.
Upper quantile is `1−.05/(7L)`; fixed historical thresholds U<.75 or corroborated
q≤.05/R<.8/U<1. Any valid label can vote ADAPT; unsupported labels are logged.
All-invalid/no vote defaults to SHARE. These are approximate conditional evidence
bounds, not proved finite-sample error control or clinically calibrated decisions.

Unknown labels may contain NaN/placeholders but never contribute as negatives.
Observed nonbinary/nonfinite labels fail validation. An all-unknown training
batch fails explicitly; the engine does not quietly invent targets or update
on an empty loss. Unsupported checkpoint labels also fail instead of changing
the macro-AP label set. B does not invent missing joint cells. Thresholds were
not tuned using any real scores, including the opened fold10.

## 3. Seven owners, freezing and parameters

Existing five-view signal architecture is unchanged; synthetic tests verified
outputs64/64/64/64/64/128/64 and the total212,261 with its original five-label
head. Replacing only the head for a fictional two-label fixture gives212,066:

| Owner | Parameter count |
|---|---:|
| Limb `view_1` |32,576|
| V1–V3 `view_2` |31,904|
| V4–V6 `view_3` |31,904|
| Temporal `view_4` |31,904|
| Frequency `view_5` |33,920|
| `fusion_1` |41,344|
| `fusion_2` |8,384|
| Two-output head |130|

Each parameter belongs to exactly one owner; cross-owner shared buffers are
rejected. `SelectiveModel` deep-clones the source, clears inherited gradients,
sets mask ownership and overrides `train()` so enclosing-wrapper recursion
cannot reactivate SHARE BatchNorm/dropout. Frozen BN affine parameters have no
gradient; frozen running statistics/counters and weights are snapshot-verified.
ADAPT BN runs in training mode on fitting batches and its counters update; E
scoring uses eval mode. Source-only freezes the head too and has no optimizer.
All other adaptation methods always update the head. Output equivalence before
updates, actual ECG-architecture masked backward paths and source hash preservation
were tested. A direct illicit buffer mutation is detected by the verifier.

Optimizer includes exactly `requires_grad` parameters: AdamW lr.0003/wd.01/
betas(.9,.99). Synthetic executor is CPU only; no new MPS parity claim is made.
Historical common-CPU fallback remains applicable. Checkpoint states include
weights/buffers; they are **not exact optimizer/RNG continuation checkpoints**.
Payload hashes are checked before loading; candidate copies are validated before
changing the destination. Corruption and incompatible state shapes fail.

## 4. Comparators, roles and label-cost fairness

Synthetic-tested methods: source-only; head-only F; head-cost F∪G; full F;
**full-cost F∪G**; closest-parameter random mask; evidence-guided selection.
Random mask enumerates128 possibilities including the reference; exact-cost
head-only/full degeneracy is recorded, not evaded with a different-size control.

Bound data roles permit F fit, G evidence (and cost-matched gradient use), E stop,
C calibration access, S selection access. O is **always refused** in this
synthetic executor; no outer-evaluation routine is implemented. Calibration and
selection role accounting are tested, but new Platt fitting or real configuration
search is not claimed. Source history includes both training and model-selection
patients; overlapping target roles, undeclared history, absent history and
checkpoint/label-schema mismatches fail. Lineage here is a synthetic declaration;
real source provenance must be bound to a newly executed pretraining ledger.

Every access verifies the actual bound membership/content hash. Access receipts
count distinct records, patients and observed record-label cells, with positive,
negative and unknown counts per role. Repeated use does not double-count the
union. `target_all` separates source anchors from target supervision; E/C/S are
not hidden. Evidence masks bind both the source checkpoint and the actual G data
identity. No caller can authorize O by changing only an operation string.

Epoch zero is eligible for pretrained adaptation. Stopping maximizes exact
scikit-learn AP over the explicit observed label schema, tie lower BCE then
earlier look. The bounded fixture engine reports **selected_update**, not an
invented real epoch. Deterministic CPU synthetic fits use isolated RNG state;
only retained weights are loaded, without claiming mid-training continuation.

## 5. Tests actually executed

Final commands from the project workspace:

```bash
.venv/bin/python -m pytest tests/phase2 -q
.venv/bin/ruff format src/dmvmat_ecg/phase2 tests/phase2
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/python -m pytest -q
```

Targeted suite: **83 passed**. Final full suite: **109 passed, zero failed/skipped,
7 warnings, 5.82 seconds** on this run. Full lint passed; formatting check reported
39 files already formatted. An interim formatting check found three new files
needing formatting after edits; those were formatted and the final check passed.
No historical file was reformatted. Earlier smaller passing runs were superseded
by this final verification; they are not added together as independent tests.

Warnings are five upstream WFDB/pandas deprecations, one deliberate no-positive
legacy metric fixture warning and one scalar-loss conversion warning in the
unchanged legacy synthetic training integration. They do not indicate real-data
execution. Existing tests create fake WFDB/cache files only under pytest temporary
directories. No real `Store`, dataset preprocessing, fold10 loader or old training
command was invoked against the research workspace.

Coverage includes signed reversal/scale/translation controls, co-occurrence
mixture changes, missing/insufficient verified classes, constant/weak embeddings,
uncertain bootstrap denominators, exact permutation p-values, Holm boundaries,
seven masks, both formulations, default999/4,999 draw execution on small fictional
samples, label-order mismatches, source-history overlap, roles/access hashes,
full-cost label union, epoch-zero retention, reproducibility, actual five-view
gradients/frozen buffers/serialization and corrupted checkpoints. Reduced Monte
Carlo counts in most fast fixtures are explicit overrides, not changed defaults.
Synthetic AP values are not ECG research findings and are not reported as such.

## 6. Remaining engineering and scientific boundaries

This delivers the synthetic-tested core, **not a real-study runner**. Before a
real internal pilot: approve its new source/target protocol; implement audited
folds1–9-only preparation, source-train-only normalization, durable source/role
lineage, real attempt reservations, calibration and a new single-use evaluation
lock. None can reuse/reset the completed study. MPS and real signal throughput
would need a separately approved validation stage. No transfer superiority is
established; actual module evidence may be invalid or select only the head.

## 7. Integrity and complete inventory

Final read-only checks: all95 tracked baseline files unchanged;48 old locked
artifacts and16 existing best/final checkpoints match stored SHA-256. Original
SQLite remains `evaluated`,11 successful attempts, zero failed, one consumed
access receipt. No new experimental predictor fit was reserved. Old results and
checkpoints were not deserialized, rewritten or used by the new synthetic tests.

The legacy `source_hash()` hashes the whole current src/config tree. Additive
Phase2 files legitimately change that **current-tree** hash; the historical
stored lock/source hash was not recomputed or rewritten. Reproduce the old
source at its frozen commit, not by pretending the expanded working tree equals
the old source snapshot. No old evaluation lease can be reopened.

Created: four source files (`phase2/__init__.py`, `contracts.py`, `evidence.py`,
`adaptation.py`); two test files (`tests/phase2/test_evidence.py`,
`test_adaptation.py`); one config (`configs/phase2/synthetic_selector.json`);
four reports (`EXTERNAL_EVIDENCE_RESOLUTION.md`, this report,
`INTERNAL_TRANSFER_PILOT_PROPOSAL.md`, `PHASE2A_DECISION.md`).
Updated by append/status note: existing external audit, diagnostic crosswalk and
multi-label methodology document. The other three existing plans are preserved.
Ignored pytest/bytecode caches are incidental. No dependencies changed; no raw
records, per-person outputs or new checkpoint files were added to Git.

Git: `main`, unchanged HEAD `1f0ff9031e9e1e2adc97962922fd97705570baf2`;
baseline tracked diff empty, intended new namespace/config/tests and Phase2
documents untracked. No commit, push, merge, PR, external waveform archive download,
real source/target training or final evaluation occurred.
