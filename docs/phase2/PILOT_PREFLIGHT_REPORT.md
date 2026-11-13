# Phase 2B metadata preflight — scientific blocker found

**No real predictive fitting, inference, calibration or evaluation. No protocol
adoption. The candidate role allocation requires prospective review.**

## 1. Authority and exact task

This sprint authorizes implementation/synthetic verification and read-only
metadata/label-feasibility auditing, not the33-fit study. All ten preceding Phase2
documents and selector/test interfaces were reconciled. The actual internal
proposal specifies **two recorded diagnostic-superclass labels, CD then STTC**.
It specifies neither five labels nor IAVB/IRBBB. This endpoint was preserved.
Both geometry A and the explicitly two-label B can represent it; no five-label B
was invented. Missing clinical diagnoses were not manufactured as negatives:
the task predicts recorded tags with the original any-key/likelihood-zero
benchmark convention, not adjudicated disease absence. External clinical
harmonization remains unresolved.

The draft36-attempt design is **proposed**, not adopted/registered. Its endpoint,
domain/split rules, numerical conventions and future authorization need review.
Known PTB-XL fold10 has already been evaluated historically; it is permanently
excluded from this pilot. This reused folds1–9 population is internal exploratory
evidence even if new models are rigorously patient-disjoint.

## 2. Actual inputs and identity verification

Release [PTB-XL1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), existing local
100Hz data. Publisher-manifest provenance and four metadata files were checked.
No new dataset/archive was downloaded. Exact SHA-256:

| Input | SHA-256 |
|---|---|
| Publisher `SHA256SUMS.txt` |`b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d`|
| Original `ptbxl_database.csv` |`7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b`|
| `scp_statements.csv` |`ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f`|
| Original eligible train metadata |`f6a951bf329522eca415da4c8cfbae4c207659bd7b8406e72be1ca96f2deee4a`|
| Original eligible validation metadata |`5bde468990544df0489ffcd5b5a06080e8bf62a3560ed7b1d0f2411dca27863c`|

Labels were projected from **train/validation metadata only** after checking
their original five-class ordering. Original CSV parsing selects only ECG ID,
patient ID, age, fold and100Hz filename; `scp_codes` is never selected/decoded.
The file was hashed as bytes for provenance. No fold10 metadata-label cache,
waveform array or prediction was opened. The old global normalization was not
loaded as a pilot transform. No new real preprocessing statistics exist.

The audit used real patient IDs and publisher native-signal checksums for known
duplicate identity. The previous full physical-signal audit recorded zero exact
within/cross-partition duplicates. No new waveform or near-duplicate search was
performed, so unknown linkage/near-duplicates remain a limitation. The future
guarded signal reader checks each100Hz header/data hash, shape, lead order and
mV units after permission; it was not exercised on real records in this sprint.

## 3. Actual eligibility and proposed memberships

19,230 eligible original development recordings were verified. The18–89 age
rule retained18,857;373 records were outside that eligibility. Lowest eligible
ECG ID per actual patient retained16,415 canonical patients, excluding2,442
additional recordings. Source18–59:7,638 patients; target60–89:8,777.
No age threshold, exclusion or label was changed after inspecting counts.

The exact deterministic algorithm/seeds are in
[the execution contract](PILOT_EXECUTION_CONTRACT.md). Two in-memory generations
matched exactly. The full private auditor manifest hash is
`b5b3dcd127fa6d15317e82f3d704f25b193f269651db094a85e9fc6429ce644f`.
The private artifact is retained under `private/phase2b_internal/preflight/`.
Individual assignments/labels are not published. A separate label-free worker
manifest and per-role label vaults prevent predictive workers from loading
unassigned labels; their identity and role hashes are in the readiness receipt.
Sealing changes access packaging only, **not memberships or outcomes**.

| Base/active role | Patients/records | CD+ | STTC+ | CD− | STTC− | Both+ |
|---|---:|---:|---:|---:|---:|---:|
| source_train |6,466|870|897|5,596|5,569|19|
| source_stop |1,172|194|201|978|971|142|
| target A reservoir |4,208|1,031|1,111|3,177|3,097|**0**|
| E reserve |951|344|357|607|594|209|
| C reserve |950|343|356|607|594|208|
| selected E |128|39|41|89|87|20|
| selected C |128|38|37|90|91|18|
| sealed exploratory O |2,668|805|838|1,863|1,830|168|

E/C are subsets of their reserves, F/G subsets of A; do not sum these aliases
as different patients. The six base roles partition16,415 canonical patients.
Active source/F/G/E/C/O sets are group/record disjoint in each draw/tier;
no known duplicate native waveform crosses prohibited roles. Full coverage,
subset/nesting, deterministic roles and membership hashes passed.

| Draw | F size | F CD+/STTC+ | G size | G CD+/STTC+ | F/G both+ |
|---|---:|---|---:|---|---:|
|4101|128|25 /40|256|59 /68|0 /0|
|4101|512|126 /138|256|59 /68|0 /0|
|4102|128|24 /41|256|58 /61|0 /0|
|4102|512|109 /141|256|58 /61|0 /0|

All hard capacity and marginal class-support floors pass: source_train≥200,
source_stop≥20, E≥10, O≥50 per positive/negative label, A≥768, E/C128 available.
C also has≥20 positive/negative labels for potential Platt calibration, though
future numerical convergence is unknown. These are feasibility counts, not power
or predictive-performance evidence. O outcomes were accessed only by the
metadata auditor for partitioning/counts; no predictive evaluation receipt exists.

## 4. Why the allocation is not acceptable without review

The rare-pattern-first marginal objective puts **all585 target joint-positive
patients outside A**. Both G draws therefore have an empty joint-positive cell.
B requires≥10 in each joint cell: it would necessarily abstain for every module,
independently of any learned representation. It cannot be an informative test
of co-occurrence-standardized selection under this allocation.

The analogous source allocation places19 joint positives in85%-intended training
and142 in15%-intended stopping. Aggregate marginal proportions are approximately
stratified but **joint diagnoses are not**. Source checkpoint selection and target
evidence would see systematically different joint-label mixtures. This is a
split-design artifact, not evidence of clinical mislabeling, negative transfer,
MMD significance or selective-adaptation failure. No such model quantities were
computed. The initial manifest's hard-capacity `blockers` list is empty;
separate scientific-review flags correctly capture this additional finding.

The proposal allows individual low-support abstentions, but structural removal
of an entire joint phenotype from the complete reservoir requires human review.
No patients were silently removed/reassigned, no alternate seed was searched,
no thresholds were relaxed and no endpoint was replaced. Recommended **prospective
amendment for review only**: stratification that also preserves joint-label cells
across source training/stopping and target roles, with deterministic synthetic
fixtures, followed by a new metadata-only preflight. Do not implement or assume
that amendment adopted. Preserving B merely as a known head-only control would
also require explicit review and a revised scientific interpretation.

## 5. Execution preparation and real boundaries

The draft matrix is source1 + eight methods×two draws×two sizes =33 planned;
ceiling36 including failures, interruptions and retries. A metadata-only dry run
validates all33 job-role bindings without reserving fits. Fitting-matched head/full
use384/768 target records including E/C; full-cost/head-cost/random/selectors/
target-only use640/1,024. Source-only raw has zero target supervision; its C-calibrated
variant consumes128. Distinct label unions include evidence/stopping/calibration;
source labels and evaluation/audit exposure are separate. Same update counts
40/160 do not prove equal compute.

Fresh-source creation, source-only normalization, new-source checkpoint lineage,
selector-role binding, frozen BatchNorm/dropout, namespace separation, durable
reservation, crash/retry behavior, calibration boundaries and sealed evaluation
were exercised with fictional data only. No new real source checkpoint, optimizer,
preprocessing state, calibrator or prediction exists. The new real ledger is
draft and blocked at0/36; both authorizations are absent. The original study
remains evaluated at11 successful attempts/one consumed access.

## 6. Resources, integrity and limitations

Read-only hardware check: M5 Pro,15 cores,48GiB; about710GiB disk available.
Installed Python3.12.14 and pinned package versions match the original environment;
CPU float32 remains the policy. The current sandbox reports MPS built but not
available; no new MPS training/parity claim or probe was made. OneCycle defaults
match the proposed explicit settings. The synthetic real-model parameter tests
verify212,066 for the two-head backbone; no real-data initialization was run.

The candidate source would require51 batches/epoch, up to2,550 source updates;
targets up to3,200 updates across32 fits. These are workload counts, not timings.
Historical source training was around45 optimization minutes/50 epochs on17,084
records; a6,466-record new task is unbenchmarked. Model weights are under1MB
each, preprocessing/representation/kernel caches modest relative to48GiB; preserve
several GB of artifact/cache headroom. At most two CPU4-thread fits are allowed.
No external download/cloud spending is needed or authorized. Memory peaks and
evidence timings remain unmeasured on real roles.

Only this metadata audit was executed on real records. Remaining issues include
joint-stratification review, draft adoption, unknown near-duplicates, reused
research population and a single proposed source seed. All prior files/results
are preserved; exact final tests, file inventory, code/dependency/worker-manifest
hashes and permission/ledger receipts are in
[authorization readiness](PILOT_AUTHORIZATION_READINESS.md) and
`results/phase2b/PREFLIGHT_AGGREGATE.json`. No real training/evaluation is authorized.
