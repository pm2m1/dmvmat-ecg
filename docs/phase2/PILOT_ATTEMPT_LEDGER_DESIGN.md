# Phase 2B attempt ledger and lock design

**Preparation only. The real ledger is draft, blocked,0/36, with no development
or evaluation authorization. Synthetic ledgers are separate temporary fixtures.**

## Identity and permissions

Auditor preparation writes separate checksum-bound label vaults per role; the
execution manifest contains **no per-record outcomes for any role**. The selector
worker therefore does not preload O/C/unused-reservoir labels. Only a permitted
operation opens its own role vault; O prediction access opens no outcome vault,
and O label access additionally requires the complete prediction seal. Group/
signal metadata and aggregate feasibility counts remain distinct from row labels.
This is a controlled-code boundary, not encryption/OS isolation from an operator
with arbitrary filesystem access. Vaults and assignments remain private/ignored.

[ledger.py](../../src/dmvmat_ecg/phase2b/ledger.py) stores a namespace, full private
manifest hash, code/config inventory hash, dependency/environment hash and draft
contract hash. Real namespace is `real:phase2b-internal`; synthetic namespaces
start `synthetic:`. Reopening with a different identity fails. A pre-existing
foreign/historical SQLite file is probed read-only and rejected before schema or
phase mutation. The historical `private/study.sqlite` is never adopted/reset.

Real configuration must equal the complete33-slot draft, ceiling36. Synthetic
rehearsals may declare a smaller exact plan while retaining the ceiling. Synthetic
factories, readers, reduced-step and reduced-Monte-Carlo overrides are forbidden
in real execution. API permissions are not an OS sandbox against arbitrary
filesystem/SQL access; approved executors and operator-controlled files remain
the trust boundary. No authorization is inferred from possessing source code.

Development requires an exact separate authorization/adoption receipt bound to
the identity. Current preflight blockers prevent that transition. Recorded
preflight blockers/audit hash cannot be silently changed: resolving them needs
prospective review and a new versioned contract/manifest/ledger context. No
development receipt was recorded in the real context.

## Reservations and counted failures

SQLite WAL, synchronous FULL, BEGIN IMMEDIATE serializes reservations. Before
model initialization or signal loading, `run_job` reserves an attempt with a UUID,
registered job specification, role-manifest/code/dependency identity, seed, PID,
fresh-start declaration and retry parent. The33 job specifications are generated
by [contract.py](../../src/dmvmat_ecg/phase2b/contract.py): source1 plus32 targets.
Source completion precedes target reservations. Two active real predictor attempts
maximum; resource limits are distinct from total36 accounting.

Attempted, failed and interrupted jobs consume capacity. Completed/active slots
cannot be rescheduled. Retry requires the latest failed/interrupted parent and
a fresh attempt UUID; at most three shared retry reservations and at most36
total rows. Unknown architectures/smokes are rejected. An exact-config technical
compatibility retry is counted, not a free probe; a new short smoke design would
need review and cannot masquerade as a registered full fit. Reserve cannot add
searches or replace unfavorable outcomes. No budget carries over from the old13
attempts or external20/172-fit drafts.

Failures during data loading, evidence or fitting remain recorded once reserved.
The fit wrapper records exceptions as failed; an abrupt process loss leaves an
attempted row, which requires explicit operator review before declaring interrupted.
No age-based timeout silently frees capacity. No exact optimizer/RNG/sampler
resumption is claimed. Retry starts from a fresh random source/target-only model
or the same immutable new source initialization for adaptation under its seed.
Model-only weights are never presented as continuation of a failed trajectory.

## Artifact publication and integrity

Checkpoint bytes go to unique attempt directories. Atomic publication writes a
new pending file, flushes/fsyncs, links it exclusively into its final name, and
fsyncs the directory. Existing final files cannot be overwritten. A crash-pending
file or a complete orphan not registered in SQLite is not a successful artifact.
Checkpoint/lineage absence, corruption or incompatible identity blocks completion.
Failed artifacts are preserved; new attempts get new directories.

Successful records are immutable through the API, with transactional terminal
updates. Checkpoint SHA, namespace, endpoint order, code/dependency/manifest hashes
and job ID are verified. Source lineage additionally requires a fresh-initialization
event, source-training/source-stopping hashes and fresh preprocessing from that
source attempt. New models have no historical checkpoint-path initialization API.
Pretrained target models inherit the verified new source; target-only preprocessing
must match its own F∪G membership union. Signal materialization/content hashes
bind preprocessing inputs to authorized roles. A misleading role string is
insufficient to authorize access.

## Lock and single-use evaluation

Development lock refuses omitted controls, active jobs, missing checkpoint/lineage
files or missing C-calibration/fallback artifacts. It records selected checkpoint
files, calibrators, attempts and the immutable context identity. Any file hash
change invalidates the lock/access path. Source/config/dependency pins are checked
again at data access. Calibration and training cannot proceed after lock.

Evaluation needs a distinct exact receipt including the lock digest. A transaction
inserts one access receipt and changes phase before any O record is returned.
Prediction access removes outcomes. Per-job predictions are immutable and must
cover every registered predictor before sealing; outcome/metric access refuses
until the seal and prediction hashes verify. The registered source checkpoint
provides source-only predictions without another predictor fit.

Use `evaluation_session()` to mark success only after sealing, or failure on an
exception; either consumes the sole lease permanently. Reopening/restarting cannot
produce a second authorization opportunity. Error recovery requires human review,
never new model fitting/recalibration based on observed outcomes. A synthetic
failure-after-O-access/restart test verifies the consumed state.

## Tests and current boundary

Synthetic tests cover concurrent duplicate scheduling, concurrent retry exhaustion
at36, failure before/during fitting, interrupted checkpoint publication, corruption,
terminal immutability, restart, namespace mismatch, unknown probes, blocked preflight,
wrong role access, source/preprocessing history, integrity drift, complete workflow,
prediction sealing, second-evaluation refusal and failure-consumed leases.
No test uses the real study DB for reservations. Real ledger creation/inspection
records only preflight state; its attempts and access rows remain zero.

The implementation does not resolve the scientific splitter failure. Passing a
ledger test does not justify the current source/target roles or a clinical claim.
See [preflight findings](PILOT_PREFLIGHT_REPORT.md) and
[authorization decision](PILOT_AUTHORIZATION_READINESS.md).
