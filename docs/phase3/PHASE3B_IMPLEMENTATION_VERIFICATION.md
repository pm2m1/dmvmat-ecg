# Phase 3B adoption and synthetic implementation verification

Date: 2026-11-14. Protocol owner: **Parth Mishra**, exact `git config user.name`.
Design: **ADOPTED v1**, transcribing the explicit human decisions in this sprint.
Scope: implementation and fictional-data verification only. This is not a grant
for real metadata preparation, RESERVED access, predictor fitting or evaluation.

## 1. Decisions applied

All three Phase3B planning documents now carry ADOPTED v1/date/owner. Review
items1/2/3/5/6/8 are approved as drafted;4/7 are prospectively amended:

- B64/B128(primary)/B256; half F, half G, nested label-blind master lists.
- A and Random-A compute masks on G under the unmodified source and then fit
  F∪G. Head-cost, Full-cost and Target-only-cost use the same gradient records.
- Three fresh sources15101/15102/15103 and draws16101/16102/16103.
- No XResNet extension. Faithful-paper arm deferred to a separate Phase3C.
- CPSC first10s truncation explicitly accepted, identical across arms; labels
  may describe later signal. No padding, cropping search or window averaging.

The human-supplied Cui Algorithm1 rationale is recorded as such. This offline
sprint did not independently verify the complete original paper or implement
its faithful objective/selector. A remains the explicitly re-engineered
MMD/permutation/geometry A-record-v1 policy, not an original-paper reproduction.
The prior draft versions were copied unchanged into
`docs/phase3/archive/phase3b_pre_adoption/` before editing.
This amendment preceded any new Phase3B data access; it does not erase the
previously disclosed Phase3A FEAS-based design and stratification history.

## 2. Finite matrix and alias semantics

|Component|Arithmetic|Slots|
|---|---|---:|
|Fresh source training|one per source seed|3|
|Head-cost|3 budgets ×3 sources ×3 draws|27|
|Full-cost|3×3×3|27|
|Selective-A|3×3×3|27|
|Random-A|3×3×3|27|
|Target-only-cost|3×3×3|27|
|**Planned maximum**|3+135|**138**|
|Technical reserves, separately limited|one fresh retry per eligible interrupted job|6|
|**Hard maximum actual attempts**|including failed/interrupted fits|**144**|

No aliases are guaranteed prospectively. All-SHARE A/random are Head-cost;
all-ADAPT are Full-cost; identical mixed random/A masks also reuse the same
immutable fitting artifact. Every alias remains a reported logical procedure.
The job key includes actual data/order, normalization, initialization, module
modes, paired stochastic seed, optimizer,200 updates, model type/schema, code
and task; method name alone is excluded. Source-only reuses three source models
and has no new adaptation fit. Alias savings cannot fund additional candidates,
extra reserves or reruns. B32 has no slots. Primary analysis remains equal mean
of nine A−Full-cost mean CD/STTC raw-AP contrasts at B128, not pooled AP or a
best-performing seed. No primary metric was computed in this sprint.

## 3. Planning support probabilities

Only the already reported CPSC FEAS aggregates were used: CD506/1930,
STTC494/1930. For each label compute P(10≤K≤G−10), K~Binomial(G,p_FEAS).

|B|F|G|CD|STTC|
|---:|---:|---:|---:|---:|
|64|32|32|0.318770116748|0.289995614110|
|128|64|64|0.984909732369|0.980301019758|
|256|128|128|0.999999972870|0.999999940173|

These are marginal planning probabilities, not a two-label independence
assumption, an audit of RESERVED, a clinical count or a guarantee of valid
geometry. Unknown annotations, source D and uncertainty checks still apply.
No support enrichment, replenishment or seed search is implemented.

## 4. Implementation and verification coverage

|Requirement|New implementation|Executed synthetic check|
|---|---|---|
|Typed sampling units and role access|`roles.py`|Known source patients; target record_cluster/unknown; cross-role overlap and false patient claims rejected; real namespaces denied before providers|
|Exact source labels, unknowns and normalization|`roles.py`, `engine.py`|Lowest numeric source ECG per patient, no OR; masked NaNs have zero gradient; source-only versus F∪G target-only statistics; fitted-artifact normalization substitution rejected|
|Backbone|`models.py`|Fresh deterministic two-logit head,212,066 parameters;12×1000 input;64×5/128/64 representations; gradients and ownership|
|Source training|`engine.py`|Fresh50-epoch AdamW/OneCycle run; completed-epoch source-stop AP/BCE tie rule; no target stopping|
|Seven-module evidence|`evidence.py`|All seven modules, whole-vector weights, shared bootstrap, seven-test Holm, constant/insufficient/unknown/translation/reversal controls, reproducible hashes|
|Production numeric settings|`evidence.py`|999-permutation p-value arithmetic and4999-draw geometry tested; production thresholds unchanged|
|Freezing|Unchanged Phase2 primitive reused by new runner|All-SHARE/all-ADAPT/seven single/mixed masks; optimizer ownership; BN affine/running buffers and dropout; head gradients; source state unchanged|
|Target adaptation|`engine.py`|Exactly200 updates/batch32 with replacement, terminal artifact; no AP/epoch-zero selection; G receives gradient exposure; no target E/C/P roles|
|Random control|`models.py`|128-mask enumeration; same ADAPT count and exact module cost including A; fixed RNG, equality probability, all-SHARE/all-ADAPT degeneracy|
|Identical-job reuse|`jobs.py`, `engine.py`|A/random all-SHARE aliases Head-cost; A all-ADAPT aliases Full-cost; no duplicate fit; changed data and corrupted artifacts cannot silently reuse|
|Attempt accounting|`jobs.py`|Atomic concurrent reservation; crash-before-fit consumes attempt; failures persist; six-reserve/144 ceiling and one-retry limit; nonfinite failures require review|
|Prediction sealing|`sealing.py`|Complete registered roster; checkpoint/normalization/code lineage; lazy O fixture outcomes unavailable until full seal; one outcome access and restart rejection; tampering rejected|

The in-memory adapter accepts only `synthetic:phase3b:*` namespaces. It contains
no discovery, loading or preparation of real R/O, no real authorization grant,
and no interface to the Phase2 ledger. Real providers are deliberately not
activated by these tests. Python fixture providers are trusted test inputs;
namespace naming is not evidence that a real cohort has become authorized.

The full ECG backbone was also exercised with **fictional12×1000 waveforms**:
50 source epochs, followed by200 head-adaptation updates. SHARE-module drift
was exactly zero in this fixture. This is an engineering result, not PTB-XL or
CPSC performance. Smaller fictional vector networks exercise all comparator
arms, mask evidence, aliases, failed/restarted jobs and one-time synthetic seals.
All test artifacts are in pytest temporary directories, separate from real
studies. No real Phase3B predictor attempt was consumed.

## 5. Commands and final results

- `.venv/bin/python -m pytest -q`: **283 passed,0 failed,0 skipped**,7 warnings,
 53.74 seconds. Includes **57 new Phase3B cases** and226 existing cases.
- `.venv/bin/ruff check .`: **PASS**.
- `.venv/bin/ruff format --check .`: **PASS**,153 Python files already formatted.

The seven warnings are existing WFDB/pandas deprecations, a deliberately
unsupported-label metric fixture and an old synthetic training tensor-scalar
warning. Historical code was not edited to suppress them.

During implementation, an extra-parenthesis syntax error and fixture import/path
errors were corrected. A newly strengthened shifted-control assertion initially
expected Holm significance from19 fixture permutations, whose minimum p=.05
cannot support seven-test Holm rejection. The test now checks rejection using
the unchanged **999-permutation production setting**, while explicitly recording
the reduced fixture's resolution limit. No scientific threshold, source/target
seed or fit allocation was altered to pass tests. Final suite has no failures.

## 6. Environment and integrity

Python3.12.14; PyTorch2.15.0; NumPy2.5.4; SciPy1.18.1;
scikit-learn1.9.1; pytest9.2.0; Ruff0.17.4; WFDB4.3.1.
All new execution fixtures are CPU float32/four Torch threads. No MPS/CUDA
training equivalence is claimed. Existing dependency pins and lockfile are
unchanged. Current new code/config identity:
`d4f0dd6361cf115d79c8384905682d76a46ee2f9fb79446680dfadafc7cc7ebb`.
This is an integrity receipt, not an authorization signature.

The217-path tracked/Phase3A preservation set was rehashed and remained identical:
`e5cc51930b035d43af32579e7f6ba8f3a632c5767a7d7f76044949d24f200b83` before and after (canonical sorted JSON path→SHA map).
Detailed map and new file hashes are in
`results/phase3b_synthetic/VERIFICATION.json`. This covers the explicitly
snapshotted public files; protected private Phase2 data/checkpoints/ledgers
were not opened or rehashed. No commands wrote to them. No real RESERVED,
Phase2 O or PTB-XL fold10 provider was invoked, and no new real predictions,
normalization fits, learned representations or evaluation metrics were generated.

Branch: `phase3-cross-hospital`. HEAD unchanged:
`c7b795553ee9806caeb1835e7a829dc0469872eb`.
`git diff --name-only` and staged diff are empty. The three editable documents
were already untracked before this sprint; Git therefore shows the Phase3
folder as untracked, rather than a tracked modification. Pre-existing Phase2
reports/results and Phase3A untracked work remain preserved. New Phase3B code,
tests, configuration, archives and reports are visible to Git; no staging,
commits, pushes, merges or PRs occurred.

## 7. Exact task file inventory

Modified only these three existing documents:

- `docs/phase3/PHASE3B_TRANSFER_PROTOCOL_DRAFT.md`
- `docs/phase3/PHASE3B_METHOD_AND_EVIDENCE_GATES.md`
- `docs/phase3/PHASE3B_COMPUTE_AND_REVIEW.md`

Created:

- `configs/phase3b/adopted_v1.json`
- `configs/phase3b/fit_matrix_v1.json`
- `configs/phase3b/support_probability_v1.json`
- `docs/phase3/archive/phase3b_pre_adoption/PHASE3B_COMPUTE_AND_REVIEW.md`
- `docs/phase3/archive/phase3b_pre_adoption/PHASE3B_METHOD_AND_EVIDENCE_GATES.md`
- `docs/phase3/archive/phase3b_pre_adoption/PHASE3B_TRANSFER_PROTOCOL_DRAFT.md`
- `src/dmvmat_ecg/phase3b/__init__.py`
- `src/dmvmat_ecg/phase3b/engine.py`
- `src/dmvmat_ecg/phase3b/evidence.py`
- `src/dmvmat_ecg/phase3b/jobs.py`
- `src/dmvmat_ecg/phase3b/models.py`
- `src/dmvmat_ecg/phase3b/protocol.py`
- `src/dmvmat_ecg/phase3b/roles.py`
- `src/dmvmat_ecg/phase3b/sealing.py`
- `tests/phase3b/__init__.py`
- `tests/phase3b/fixtures.py`
- `tests/phase3b/test_contracts.py`
- `tests/phase3b/test_evidence.py`
- `tests/phase3b/test_execution.py`
- `docs/phase3/PHASE3B_IMPLEMENTATION_VERIFICATION.md`
- `results/phase3b_synthetic/VERIFICATION.json`

Generated Python/pytest/Ruff caches and fictional temporary artifacts are ignored
operational files, not real study artifacts. No real Phase3B ledger, R/O split
manifest, participant assignments or checkpoint was created.

## 8. Stopping point and remaining conditions

Adoption and the requested synthetic engineering sprint are complete. Independent
external **clinical validation remains BLOCKED**: unknown target patient linkage,
annotation-not-clinical-negative semantics, prior FEAS selection and waived source
waveform overlap checking remain explicit limitations.

Next action requires separate authorization for metadata-only preparation and
reviewed real-provider integration: bind the frozen mapping/source eligibility,
actual group memberships/duplicates and all dataset hashes; generate R/O and
three nested label-blind draw memberships exactly once; verify actual role and
observation boundaries without O development access. The current fictional-only
execution guard must not be bypassed by relabeling real data as a fixture.
Real training and one-time evaluation require their own subsequent instructions.
No R/O preparation, RESERVED access, real fitting, evaluation or commits were
performed. Do not interpret passing tests as selective-transfer superiority or
permission to begin the144-attempt study.
