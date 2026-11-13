# Phase 2D — amended execution integration

**Implemented and synthetic-tested; scientific adoption and real execution are
DENIED pending separate human decisions.** Historical implementation unchanged.

## New version, exact bindings and private preparation

Namespace `real:phase2d-joint44301-v1`;
execution version `phase2d-amended-v1`, split `joint-pattern-hamilton-v1`, seed44301.
Candidate auditor hash `c809d6db691d7ad3094dcd3635546ac58941529d1c6a3d06be35129f8ca0f4c9`.
Sealed worker hash `65cf652084396403077918e0e9be560eaedd009ab4fc8587260c833ddcf4f816`.
Contract hash `dbc0afe9bee29e7323a3a353b37660b8dff20733e3fe62fa94958d95400ecb28`.
Source/config inventory hash `5440faf264ba3e76aa432556c6996146e2dee5fa23595dbf0caf677b5c7cc942`.
Dependency/environment hash `195cd5ae983b6a3e196bc52ac0cf0997b26f60f45fb801e53256b8b57f881424`.
The [preparation receipt](../../results/phase2d/EXECUTION_READINESS.json) includes
all source dataset checksums, source/target domains, actual role hashes, method
configuration, source anchors, G support, permission refusals and ledger snapshot.
Role vaults/checkpoints/assignments remain private/ignored.

[Contract](../../src/dmvmat_ecg/phase2d/contract.py) preserves the earlier source/
target schedules, label budgets, eight trained methods and33+3 matrix. Two old
fields that would otherwise contradict the amendment (`source_split_seed33002`,
`target_split_seed33001`) now explicitly say44301; superseded seeds are marked
historical. This integrates the authorized split proposal, not a new seed search.
No endpoint/canonical selection/architecture/selector threshold was changed.
P is explicitly absent; the external plan's P cannot become extra pilot labels.

[Preparation CLI](../../scripts/phase2d_prepare.py) accepts only `--prepare-only`.
It loads and hash-verifies the existing candidate, checks its actual source
anchors/registered G counts and exact config/unsigned-template consistency;
it does **not** regenerate roles, authorize work, load signals or initialize a
predictor. The new real SQLite ledger was created draft,0/36; preflight counts
are recorded separately from human adoption and training/evaluation grants.
The original Phase2B draft ledger and Phase2C candidate are not overwritten,
reopened for execution, aliased or cleared.

## Execution modules and reuse boundaries

- [ledger.py](../../src/dmvmat_ecg/phase2d/ledger.py): new namespace/config identity,
  explicit complete review record, distinct development/evaluation grants and
  live contract checking. Inherits unchanged Phase2B durable reservation,
  publication, failure/retry, calibration-lock and single-use evaluation methods.
  Foreign schemas are probed read-only and rejected before mutation.
- [access.py](../../src/dmvmat_ecg/phase2d/access.py): exact candidate/endpoint
  gate, per-role sealed label vaults and separate O membership vault. **O record
  IDs and labels are absent from the worker's loaded role table**. A valid
  post-lock lease precedes O identity reads; complete prediction seal additionally
  precedes O outcome reads. Trainer/selector calls to O always fail.
- [execution.py](../../src/dmvmat_ecg/phase2d/execution.py): independently versioned
  copy of the guarded Phase2B training/calibration/prediction executor, keeping
  optimizer, data, stopping, loss and inference semantics. Added pre-reservation
  integrity checking and runtime/trainable-count/module-drift/random-match logs.
  The original file is untouched. The unchanged Phase2B normalization/source/
  representation helpers and Phase2A mask/MMD/geometry utilities are reused.
- [annotation_audit.py](../../src/dmvmat_ecg/phase2d/annotation_audit.py): separate
  metadata-only reviewer; it is not an outcome or predictor interface.

Real worker initialization rejects the pre-amendment full/worker schema, wrong
contract, changed CD/STTC order, synthetic manifests, changed candidate hash and
missing integrity pins. Role hashes bind the exact canonical records, metadata,
label-vault contents and dataset provenance; misleading role strings cannot
relabel data. A live manifest/code/config/dependency change fails access.
Historical checkpoint IDs, false fresh-source provenance and old normalization
memberships fail the new-source checks. The API is a controlled-code boundary,
not an OS sandbox or authentication against an operator who edits files/SQLite.

## Exact method/episode matrix and source isolation

| Method | Gradient pool | Selector pool | Other target roles | Distinct target people F128 / F512 | Max updates |
|---|---|---|---|---|---|
| source-only | none | none | none raw / C calibrated | 0 / 128 | no extra fit |
| head_f | F | none | E/C | 384 / 768 | 40 / 160 |
| head_cost | F∪G | none | E/C | 640 / 1,024 | 40 / 160 |
| full_f | F | none | E/C | 384 / 768 | 40 / 160 |
| full_cost | F∪G | none | E/C | 640 / 1,024 | 40 / 160 |
| random_a | F | G for A mask-cost | E/C | 640 / 1,024 | 40 / 160 |
| selective_a | F | G | E/C | 640 / 1,024 | 40 / 160 |
| selective_b | F | G | E/C | 640 / 1,024 | 40 / 160 |
| target_only | F∪G | none | E/C | 640 / 1,024 | 40 / 160 |

One fresh source seed3101 plus these eight trained target methods×draws4101/4102
×F128/F512 =33 planned fits, ceiling36 including technical failures/retries.
Paired target seeds5101/5102; fixed G256, E128, C128; no additional selection
S/P pool. E chooses checkpoints only; C fits calibrators/operating thresholds;
O is sealed exploratory evaluation. Primary remains Selective-A−Full-cost at
F512 (same1,024-person total-label union). Audit/stratification labels and later
O labels are disclosed separately, and repeated use is union-counted.

Each adaptation receives the exact same newly trained source checkpoint and
source-only normalization; target-only is the explicitly fresh F∪G reference.
No old five-view/XResNet weights, heads, optimizer states or source statistics
are admissible. Source_train6,492 and source_stop1,146 exclude all target people.
Fresh initialization and preprocessing attestation bind source seed, exact role
hashes and attempt ID. Optimizers are newly constructed. No exact continuation
from model-only weights is claimed; interrupted retries use a fresh attempt and
initialization (or the same immutable new source for adaptation).

Source50×ceil(6,492/128)=2,550 maximum updates; target32 fits total≤3,200 updates.
Common CPU float32, four threads/fit, at most two future active workers; no new
MPS training/parity claim. Epoch zero is eligible for pretrained methods and
ineligible for scratch checkpoint retention. SHARE parameters and all frozen
BN buffers/dropout modes are verified after updates and retained checkpoint
loading. Parameter-matched random selection includes exact-equality degeneracy;
its matching record is logged, not replaced with an unfair larger mask.

## Actual selector support, unchanged policies

All joint vectors below are ordered00,10,01,11.

| Evidence | N | Joint cells | CD+ / CD− | STTC+ / STTC− | Observed |
|---|---:|---|---|---|---|
| G4101 (both tiers) |256|141,48,50,17|65 /191|67 /189|100% each|
| G4102 (both tiers) |256|122,51,62,21|72 /184|83 /173|100% each|
| Source analysis for CD |512|228,218,28,38|256 /256|66 /446|100% each|
| Source analysis for STTC |512|227,29,218,38|67 /445|256 /256|100% each|

MMD cap256/domain, four bandwidths.5/1/2/4,999 domain permutations, plus-one
p-values and Holm correction over seven modules. Geometry source cap256 per
indexed class; minimum10 observed patients/class for A, per joint cell for B,
source D≥.1 and valid scatter/denominators;4,999 patient-vector bootstrap draws,
upper quantile1−.05/(7×2)=.9964285714. Any valid label votes; all-invalid defaults
SHARE with abstention. MMD remains evidence, not a loss. Metadata support passes
but source geometry/uncertainty and later actual masks are unmeasured.

## Budget, locks and current state

Each predictor attempt is atomically reserved before signal loading/model
initialization; errors/interruptions consume it.3 reserve attempts are global
technical retries, not new architectures, hidden smokes or unfavorable reruns.
Immutable completed artifacts are hash-verified; a pending/corrupt/orphan file is
not a completed checkpoint. No attempt is freed on restart. Synthetic concurrency
reaches36 and rejects further reservations; all tests use isolated temporary DBs.

Protocol adoption is separate from development authority, and development cannot
reach O. Lock requires all declared fits and calibration/fallback artifacts;
code/data hashes are checked again at access. Generate/seal all33 registered
synthetic predictions before outcome access. A source-only alias uses the one
source checkpoint with no extra fit. Evaluation failure consumes its lease;
no second authorization opportunity exists through ordinary restart.

Actual real state: draft,0/36, no adoption record, no development/evaluation
receipts, no access records. The remaining condition is **human adoption and
separate training authorization**, not an automatic grant by this report.
