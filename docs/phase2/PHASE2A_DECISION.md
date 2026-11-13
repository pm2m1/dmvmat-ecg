# Phase 2A decision and next authorization boundary

**2026-11-12. No real-data predictive execution authorized or performed.**

| Independent decision | Status | Meaning |
|---|---|---|
| **EXTERNAL CLINICAL TRANSFER** | **BLOCKED** | Georgia clinical negatives, public patient linkage and source-overlap exclusion remain unresolved; neither the broad five-superclass nor narrower clinical task is ready to execute |
| **SELECTIVE-TRANSFER ENGINEERING** | **READY — synthetic-tested core only** | Both geometry formulations, seven-module evidence/masks, freezing, epoch-zero and fair comparator interfaces pass synthetic verification; this is not real-study or clinical readiness |

The primary objective is unchanged: determine whether DMV-MAT-inspired selective
adaptation improves target-population ECG classification over conventional full
fine-tuning under matched **total** labeled-data budgets. No architecture search,
PTB-XL score chase or transfer superiority claim replaces that objective.

## Workstream A: evidence did not clear the external gate

[External evidence resolution](EXTERNAL_EVIDENCE_RESOLUTION.md) distinguishes
recorded-code positives from clinically confirmed diagnoses and verified negatives.
IAVB270492004 and IRBBB713426002 are compatible named positive concepts with the
corresponding PTB-XL statements. Absence is still unknown. The competition
evaluator's zero initialization is a scoring convention, not proof of complete
clinical exclusion. No negative population or final eligible count was invented.

The authors report repeated patients in the full Georgia collection and original
split isolation; the required public file-to-patient linkage for our new roles
was not established. Separate institution names, record numbers and exact-waveform
hashes cannot prove no common patient with different ECGs. The entire Challenge
is not an independent target because it includes PTB-XL and other repeated
distribution components. See the updated audit/crosswalk for provenance.

Still needed: curator-supported person mapping, release-specific comprehensive
phenotype ascertainment or blinded adjudication, overlap/duplicate audit and actual
eligible patient/label counts. A reduced endpoint remains a possibility, not an
approved clinical target. No full external waveform archive was downloaded and
no curator message was sent.

## Workstream B: completed synthetic implementation

New code is in [src/dmvmat_ecg/phase2](../../src/dmvmat_ecg/phase2/), preserving
the original five-view architecture and legacy prototype. A: per-label signed
source-relative geometry. B: co-occurrence-standardized two-label geometry.
Both use joint patient-vector bootstrap, explicit observation masks, low-support
abstention and the specified MMD/permutation/Holm corroboration. MMD is evidence,
not an alignment loss. Original-author JS divergence and our later MMD
re-engineering remain explicitly distinguished.

Reliable module operations clone source state, identify all seven owners, keep
the head trainable except source-only, enforce frozen BatchNorm affine/buffers
across enclosing train calls, verify immutable owners, and select pretrained
checkpoints with epoch zero eligible. Exact AP stopping and hash-checked
serialization are tested. Full-cost may train on F∪G, so selectors do not get
undisclosed evidence labels while the comparator only sees F. Access accounting
includes fitting, evidence, stopping, calibration and selection, with distinct
target/source unions. History checks reject sources that saw target participants.

[Implementation report](SELECTOR_IMPLEMENTATION_REPORT.md) gives parameters,
coverage, commands and limitations: **83 new Phase2 tests**, **109 full-suite
tests passed**, seven existing warnings; Ruff and formatting checks pass.
No real datasets/checkpoints were used by tests. The real study remains at11
successful predictor attempts and one consumed historical evaluation lease.

Engineering readiness is limited to the tested core. Missing real-run infrastructure
is not hidden: a trusted role-bound real data adapter, source-only preprocessing
lineage, durable new attempt ledger, clinical/calibration integration and a
separate locked evaluation executor require further implementation and approval.
The in-memory synthetic-origin declaration is not an adversarial security boundary.
No new CPU/MPS equivalence claim is made; CPU is the tested execution path.

## Workstream C: recommended next experiment

[Internal pilot proposal](INTERNAL_TRANSFER_PILOT_PROPOSAL.md): folds1–9 only,
age18–59 source /60–89 target, patient/duplicate-group disjoint, no fold10 access.
One completely newly trained random-initialized source model, new source-only
normalization and a new `(CD,STTC)` annotation task. No old source checkpoint,
head or learned transform may be reused: old training/selection already included
potential newly assigned target participants. Resetting a head does not remove
that history. Source pretraining and source stopping must both exclude all target
roles, including evaluation.

This proposal tests selective adaptation, not a new backbone. Two target draws,
F128/512, separate G256, E128/C128, sealed target O. The primary A−Full-cost
contrast at F512 uses1,024 distinct target records of supervision per procedure,
with actual per-label positives/negatives and co-occurrence reported. Source-only,
head, full, cost-matched head/full, random, A/B and a bounded target-only control
are defined; no outcome-driven label/domain selection. **33 planned predictor
fits, proposed hard ceiling36 including failures**, not authorized by this sprint.
Label-count feasibility has not been checked on real participants.

This is **internal exploratory selective-transfer evaluation**, not external
clinical validation. Its population and research direction were already studied;
newly clean source/target roles do not erase researcher-level dataset reuse.
An independent second source model/strong XResNet extension would require new
pretraining on the same newly assigned source patients, not old weights, and
separate budget approval. A future external target must pass its own gates.

## Approval sequence requested for future work (not executed)

1. Review the internal annotation endpoint, age domains, patient/duplicate
   grouping, source-training exclusions, exact schedules, primary contrast,
   label-cost accounting, conditional bootstrap and36-attempt ceiling.
2. Separately authorize folds1–9 **metadata/data preparation only**, validate
   actual counts and hashes, without training or inference. Do not access fold10.
3. Authorize the new real-data adapters/ledger/calibration/lock implementation
   with synthetic integration and source-history verification. Fail closed on
   overlap, unknown lineage, wrong labels or insufficient stopping/evaluation
   support. Original study ledger and results remain immutable.
4. Only after those receipts pass, authorize fresh source training and the
   bounded internal adaptation matrix. Lock all policies, checkpoints and
   calibration before a separately authorized one-time internal O evaluation.
5. Continue external evidence resolution independently. Internal/synthetic success
   does not remove missing clinical negatives or establish independent patients.

## Complete changes and Git status

Created11 files this sprint:

- `src/dmvmat_ecg/phase2/__init__.py`
- `src/dmvmat_ecg/phase2/contracts.py`
- `src/dmvmat_ecg/phase2/evidence.py`
- `src/dmvmat_ecg/phase2/adaptation.py`
- `tests/phase2/test_evidence.py`
- `tests/phase2/test_adaptation.py`
- `configs/phase2/synthetic_selector.json`
- `docs/phase2/EXTERNAL_EVIDENCE_RESOLUTION.md`
- `docs/phase2/SELECTOR_IMPLEMENTATION_REPORT.md`
- `docs/phase2/INTERNAL_TRANSFER_PILOT_PROPOSAL.md`
- `docs/phase2/PHASE2A_DECISION.md`

Updated three existing, initially untracked plans by status/addendum:
`EXTERNAL_ECG_DATASET_AUDIT.md`, `DIAGNOSTIC_LABEL_CROSSWALK.md`,
`MULTILABEL_DMV_MAT_METHOD.md`. The other three initial planning files are
unchanged. All95 tracked files remain byte-identical to the starting inventory;
48 prior locked artifacts and16 old checkpoint files match stored SHA-256.
Existing Kaggle/Phase1 repositories were not modified.

Current branch `main`; HEAD `1f0ff9031e9e1e2adc97962922fd97705570baf2` unchanged.
Tracked diff empty; untracked paths are the10 Phase2 documents (six pre-existing,
four new), four new source files, two tests and one configuration. Synthetic
temporary files/caches are ignored. Dependency pins/uv.lock remain unchanged.
The old stored source lock remains unchanged; new additive src/config files
necessarily change the current-tree hash and cannot reopen the historical lease.

No real source/target fitting, real waveform inference, final performance metric,
new patient-role manifest or external clinical evaluation occurred. No commit,
push, merge or PR. Stop here pending review; the proposed36 fits are not launched.
