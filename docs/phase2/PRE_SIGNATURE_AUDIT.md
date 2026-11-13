# Pre-signature protocol audit

2026-11-13. Independent read-only review of `/Users/kiyo/Desktop/dmvmat-ecg`,
branch `main`, historical HEAD `1f0ff9031e9e1e2adc97962922fd97705570baf2`.

**Overall verdict: DO NOT SIGN — FOUNDATION OR INTEGRITY PROBLEM.**

The multiview foundation is sufficient for an exploratory transfer pilot; the
blocking finding is execution integrity, not a demonstrated failure of the
architecture. The earlier engineering-readiness recommendation was too strong.
The existing default-DENY state is safe, but complete live code-inventory binding
and a single authoritative real ledger are not enforced by the execution API.
These need correction and a fresh read-only audit before signing this package.
Nothing was adopted, authorized, fitted, inferred or evaluated by this audit.

## Summary A–K

PASS means the stated audit requirement is supported within the existing pilot's
scope; it is not permission to execute. UNCLEAR means insufficient evidence or
precision, not a finding of a data breach. File:line references use the current,
unchanged checkout. The companion amendment is **UNAPPLIED**.

| Item | Status | One-line finding | Exact evidence |
|---|---|---|---|
| A. Multiview foundation | PASS | Paired AUROC gain over flat is supported conditionally; AP superiority/noninferiority is not established, but an exploratory mechanism pilot is justified. | `docs/EXPERIMENT_RESULTS.md:25`, `:26`, `:33`, `:39`, `:191` |
| B. Primary method | PASS | A is explicitly primary and B explicitly secondary; no best-of-A/B test selection is specified. | `docs/phase2/MULTILABEL_DMV_MAT_METHOD.md:123`, `:180`; `src/dmvmat_ecg/phase2b/contract.py:115` |
| C. Primary endpoint | PASS | Raw mean CD/STTC AP, A−Full-cost at F512, averaged over the two paired draws; secondary family and multiplicity are separate. | `docs/phase2/PROTOCOL_ADOPTION_REVIEW.md:82`; `docs/phase2/INTERNAL_TRANSFER_PILOT_PROPOSAL.md:203`, `:217` |
| D. Abstention | PASS | All-invalid module evidence gives SHARE; all-SHARE produces head-only adaptation, with source epoch zero eligible; both fixed episodes remain in the primary mean. | `src/dmvmat_ecg/phase2/evidence.py:230`, `:317`; `src/dmvmat_ecg/phase2/adaptation.py:58`; `docs/phase2/INTERNAL_TRANSFER_PILOT_PROPOSAL.md:203` |
| E. O support/power | UNCLEAR | Draft N=2,633; non-O estimates give about757/798 positives; plausible AP MDEs exceed0.03 and +0.005 power is unsubstantiated. | `docs/phase2/JOINT_PATTERN_FEASIBILITY.md:96`, `:98`, `:100`, `:101`; this audit §E |
| F. Random-mask fairness | PASS | Exact parameter matching also implies equal module count for this architecture; only1 or3 masks qualify, with at most two underlying mask draws. | `src/dmvmat_ecg/phase2/adaptation.py:94`; `docs/phase2/SELECTOR_IMPLEMENTATION_REPORT.md:76`; `src/dmvmat_ecg/phase2b/contract.py:52` |
| G. Label-budget fairness | PASS | A/B/random and cost-matched full/head use the same640/1,024 target-person label union; fitting-only controls are explicitly smaller. | `docs/phase2/INTERNAL_TRANSFER_PILOT_PROPOSAL.md:136`, `:148`; `src/dmvmat_ecg/phase2b/contract.py:46` |
| H. Authorization/integrity | FAIL | Caller pins are not bound back to the approved code hash; real ledger path is unrestricted, so36/single-use enforcement is per DB, not globally authoritative. | `src/dmvmat_ecg/phase2b/access.py:33`; `src/dmvmat_ecg/phase2d/access.py:90`; `src/dmvmat_ecg/phase2d/ledger.py:35`, `:47` |
| I. Fit matrix | PASS | One fresh source plus8 methods×2 draws×2 tiers=33; three technical reserves give36. | `src/dmvmat_ecg/phase2b/contract.py:31`; `configs/phase2d/amended_execution_draft.json:536`; `docs/phase2/INTERNAL_TRANSFER_PILOT_PROPOSAL.md:175` |
| J. Domain realism | UNCLEAR | Age support differs by construction and non-O tag prevalence differs substantially; learned-ECG shift and literal single-hospital provenance are unverified. | `docs/phase2/INTERNAL_TRANSFER_PILOT_PROPOSAL.md:37`; `docs/phase2/JOINT_PATTERN_FEASIBILITY.md:58`, `:96`, `:98`, `:100` |
| K. Desktop/iCloud | UNCLEAR | Desktop is not a symlink and CloudDocs/Desktop is absent; brctl confirms active cloud containers but cannot establish Desktop-sync setting. | `docs/phase2/PRE_SIGNATURE_AUDIT.md:506` (read-only command receipts) |

## Scope and evidence limitations

Read the Phase2D review/readiness/unsigned template, referenced method, internal
pilot, split amendment/feasibility and integration documents; original completed
classification reports and aggregate comparison JSON; selector/comparator,
access, execution, ledger and pin-inventory code; and synthetic test source.
No successor adoption/approval JSON was found by filename inventory under
`configs`, `private` or `docs`; administrative SQLite reads show no stored adoption
or development/evaluation grant. The only review JSON found is the unsigned
Phase2D template. This is a search of the workspace, not a claim about documents
held by a reviewer elsewhere.

No study class, preprocessing routine, model, test suite, Ruff command or training
script was executed. Previous186 passing tests are **historical evidence**, not
new verification. New calculations were standard-library integer enumeration and
analytical power quadrature, with no real prediction scores or model inference.
SQLite was opened with `mode=ro&immutable=1` for administrative state only.

**O boundary:** no O outcomes were decoded, displayed or used; no O signals,
learned representations or predictions were loaded. Previously published role
counts were read as aggregates. The initial integrity snapshot hashed existing
private JSON containers opaquely, including sealed manifests/vaults, without
parsing their labels. This touched their bytes and is disclosed rather than
claiming that no sealed container was opened at filesystem level. Subsequent
reasoning and all power inputs use published aggregates/non-O counts only; no
individual O label information was obtained. This snapshot is not an O access
receipt and does not authorize predictive access.

## A. Multiview foundation — PASS, with a metric-specific conclusion

### Actual historical classification evidence

These are five-superclass results, **not the proposed two-label transfer task**.
Completed-study scores are means of two seed-specific metrics, not a probability
ensemble (`docs/EXPERIMENT_RESULTS.md:3`). Seeds101/202 share the same participants.

| Model | Fold9 validation AUROC, seeds101 /202 | Fold9 validation AP, seeds101 /202 | Locked fold10 mean AUROC | Locked fold10 mean AP |
|---|---|---|---:|---:|
| Flat CNN |0.92269 /0.92359|0.79918 /0.80529|0.91733|0.80657|
| Five-view, no augmentation |0.92264 /0.92436|0.80453 /0.80786|0.92139|0.81405|

Evidence: `docs/EXPERIMENT_RESULTS.md:25–26` and `:187–194`; explicit seed-wise
locked scores are at`:229–232`. Fold10 contains2,158 records/1,877 patients
(`:7`). All full fits completed50 epochs; each retained checkpoint was chosen
using fold9, not fold10 (`:198`). The 0.92139 number is the **locked fold10,
no-augmentation five-view model**, not a validation score, XResNet score or
transfer result. XResNet's validation-best locked result is0.92985/0.82759
(`:23`); its final-epoch procedure is distinct (`:24`). Approximate validation
means are five-view AUROC0.92350/AP0.80620 and flat0.92314/AP0.80224, using rounded
seed rows; no validation superiority CI was established.

Five-view−flat locked ΔAUROC=+0.00405510, primary97.5% Bonferroni paired CI
[+0.00008502,+0.00785551]. ΔAP=+0.00748101, exploratory95% CI
[−0.00078093,+0.01599541]. The public `results/locked_metrics.json` comparison
entry agrees with `docs/EXPERIMENT_RESULTS.md:33`; patient-cluster pairing,
2,000 bootstrap draws and conditional limitations are specified at`:39`.

Thus there is narrow conditional AUROC improvement over the matched flat
control. AP has only a positive point estimate; neither AP superiority nor a
formal AP noninferiority claim is justified. No prespecified AP noninferiority
margin was tested. The five-view model also underperformed XResNet in the
historical paired comparison (`docs/EXPERIMENT_RESULTS.md:35`).

**Foundation verdict:** a modular, competently performing backbone is sufficient
to investigate selective versus full adaptation. This does not require proving
it is the best backbone first. Proceeding to an exploratory mechanism pilot is
scientifically reasonable after execution-integrity corrections; claiming broad
multiview superiority or externally confirmed transfer is not.

## B. One primary selective method — PASS

Exact protocol headings:

> “Formulation A: per-label source-direction retention (primary proposal)”

> “Formulation B: co-occurrence-standardized geometry (secondary proposal)”

Evidence: `docs/phase2/MULTILABEL_DMV_MAT_METHOD.md:123` and`:180`. The task in
that generic document also discusses the blocked external endpoint; the actual
internal pilot fixes CD/STTC at `INTERNAL_TRANSFER_PILOT_PROPOSAL.md:47`.
The executable contract's primary string is
`mean_draw_macro_AP(selective_a - full_cost), n_fit=512`
(`src/dmvmat_ecg/phase2b/contract.py:115`, inherited by Phase2D).
B−A is in the secondary comparison family, not an alternative primary winner
(`INTERNAL_TRANSFER_PILOT_PROPOSAL.md:217`). No test-based best-of-A/B procedure
is authorized. “Primary proposal” still requires human adoption; it is not a
retrospective claim that the draft is already registered.

## C. One primary endpoint and comparison — PASS

The exact primary estimand is

`mean_draw { 0.5[AP_CD(A)+AP_STTC(A)] − 0.5[AP_CD(Full-cost)+AP_STTC(Full-cost)] }`

at F512/G256, over the two fixed paired draws. Raw continuous sigmoid scores,
scikit-learn non-interpolated AP, equal label weights; **not CD-only AP**, pooled
predictions, calibrated micro AP, AUROC, best draw or best budget. Protocol:
`docs/phase2/PROTOCOL_ADOPTION_REVIEW.md:82–88` and
`INTERNAL_TRANSFER_PILOT_PROPOSAL.md:203–225`. `macro_ap()` calls exact
`average_precision_score` and averages the two label scores
(`src/dmvmat_ecg/phase2/adaptation.py:174–187`).

Primary95% conditional paired patient-cluster CI;10,000 draws, seed64001,
shared complete vectors/weights across methods, labels and episodes. Five
secondary AP contrasts A−Head-cost, A−Random, A−Full-F, B−A, A−raw Source-only
have ordinary95% and Bonferroni99% intervals. Other budgets, per-label metrics,
AUROC, Brier/log loss/ECE, calibration and operating-point metrics are secondary/
exploratory, separately listed. Training seeds are not independent patients;
undefined-label bootstrap handling and >1% unreliability rule are specified.

Selector-internal multiplicity is separate: Holm over seven MMD tests and
upper geometry quantile1−.05/(7×2), not a substitute for final comparison
multiplicity (`MULTILABEL_DMV_MAT_METHOD.md:154`;
`src/dmvmat_ecg/phase2/evidence.py:295`). The primary/secondary distinctions are
adequate. The full signed analysis-document bundle should additionally be bound
as recommended under H; the approved-label/data scope must not change later.

## D. Abstention and primary inclusion — PASS

A label with invalid geometry casts no ADAPT vote
(`src/dmvmat_ecg/phase2/evidence.py:230–232`). Any valid label can vote; otherwise
the module is SHARE (`:317–322`). Partial invalidity is recorded, not fabricated
invariance. Protocol text explicitly says all-invalid is SHARE with an abstention
flag (`MULTILABEL_DMV_MAT_METHOD.md:166–178`).

If all seven modules abstain, the produced model is **head-only adaptation**
from the new source, not an automatically substituted source-only comparator:
`SelectiveModel` freezes SHARE owners but keeps the head trainable unless
`source_only=True` (`src/dmvmat_ecg/phase2/adaptation.py:46–58`). The Phase2D
executor applies that wrapper to A/B/random (`execution.py:121–137`). Pretrained
epoch zero remains checkpoint-eligible (`:169`); E may therefore retain the
untouched source if head/adaptation updates do not improve its objective. These
are two stages of a fixed procedure, not post-test fallback choices.

Both declared A episodes remain in the primary two-draw arithmetic mean regardless
of mask validity/count. There is no protocol instruction to exclude abstainers:
`INTERNAL_TRANSFER_PILOT_PROPOSAL.md:10`, `:127`, `:203–206`. The ledger cannot
complete the registered development lock if a required A/B job is omitted
(`src/dmvmat_ecg/phase2b/ledger.py:298–310`). A genuine technical fit failure
requires a counted retry or an incomplete-study report, not deletion of an
unfavorable/abstaining episode. The future analysis implementation must preserve
this unconditional episode inclusion; it was not run or newly tested here.

## E. O sample support and approximate power — UNCLEAR

### No O outcomes used

There is **no approved manifest yet**: the adoption template is unsigned and the
real ledger has no adoption record. The proposed O size is2,633 records and2,633
patients (one canonical ECG/person), from already published aggregate manifest
report `docs/phase2/JOINT_PATTERN_FEASIBILITY.md:101`. It is a draft design size,
not an approved test population. Earlier documents also publish O positive counts;
under the requested approval condition this audit does not use them as power
inputs or claim they are approved. It does not reopen any per-record O data.

Use disjoint non-O target **base roles only**: A4,388, E-reserve878, C-reserve878,
N_nonO=6,144. Do not add their overlapping F/G/E/C subsets. CD positives
1,262+252+252=1,766; STTC positives1,331+266+266=1,863. Evidence:
`JOINT_PATTERN_FEASIBILITY.md:96`,`:98`,`:100`.

| Label | Non-O prevalence | Estimated O positives | Estimated O negatives |
|---|---:|---:|---:|
|CD|1,766/6,144 =28.74349%|756.816 ≈757|1,876.184|
|STTC|1,863/6,144 =30.32227%|798.385 ≈798|1,834.615|

These are **estimates**, not a fresh O-label audit. Both exceed100 expected
positives comfortably, so the requested positive-support failure flag is **not**
triggered. Joint-stratum allocation makes prevalence transfer plausible, but
support alone does not determine paired AP power.

### Transparent, outcome-free MDE sensitivity calculation

80% power, two-sided alpha.05 normal approximation:
`MDE ≈ (z_.975+z_.80) SE(estimated paired difference) =2.801585×SE`.
No model was fitted and no real/synthetic patient predictions were generated.
Only deterministic integration of hypothetical distribution functions was used.

Assumptions: independent patient units, one record each; two similarly performing
methods per label; class-conditional continuous scores N(d,1) for positives and
N(0,1) for negatives with `d=√2 Φ^-1(a)` and assumed AUROC a=.80 or.90. Use the
non-O prevalences above, N=2,633. Per-label variance comes from the influence-
function quadrature in the appendix below, not a binomial AP shortcut. Let r be
the correlation of the **two metric estimators**, not raw scores, and c the
correlation between their paired differences across CD/STTC. Neither is known.
Use `Var(Δ_j)=2(1−r)V_j` and
`Var(mean Δ)=[VΔ_CD+VΔ_STTC+2c√(VΔ_CD VΔ_STTC)]/4`.
No independent-sample √2 credit is taken for the two fitted target draws: they
share participants and a source. Scenarios assume their sampling errors are
perfectly correlated, a conservative planning choice, not an estimated fact.

| Assumed per-label AUROC | Method-estimator r | Cross-label c | Macro AP MDE | Macro AUROC MDE |
|---:|---:|---:|---:|---:|
|.80|0|.5|.06265 **> .03**|.03184 **> .03**|
|.80|0|1|.07233 **> .03**|.03676 **> .03**|
|.80|.5|.5|.04430 **> .03**|.02251|
|.80|.5|1|.05115 **> .03**|.02599|
|.80|.9|.5|.01981|.01007|
|.80|.9|1|.02287|.01163|
|.90|0|.5|.04521 **> .03**|.02194|
|.90|0|1|.05219 **> .03**|.02534|
|.90|.5|.5|.03197 **> .03**|.01552|
|.90|.5|1|.03691 **> .03**|.01792|
|.90|.9|.5|.01430|.00694|
|.90|.9|1|.01651|.00801|

These are scenario estimates, **not measured pilot AP/AUROC, guaranteed bounds,
or a completed power analysis of the fitted procedures**. Different score shapes,
pair correlations, ties, ascertainment and source-training variation can change
them materially. Much higher correlation could yield smaller MDEs; it is not
known now. The proposed+.005 AP relevance signal is below every tabulated MDE.
Do not claim a powered+.005 superiority experiment or infer “no useful effect”
from a null result. Treat this as feasibility/effect-size estimation unless a
prospective, non-O-based precision justification establishes more. Do not open,
resplit or enlarge O based on outcomes to improve the calculation.

For methodology context, [DeLong et al.](https://pubmed.ncbi.nlm.nih.gov/3203132/)
accounts for covariance of correlated AUROCs, and
[Su, Yuan and Zhu](https://arxiv.org/abs/1310.5103) derives AP asymptotic variance.
Neither makes power identifiable from N and prevalence alone. The numerical
scenario below is an explicit auditor derivation, not claimed as those authors'
exact implementation.

### Quadrature details for reproducibility

For a label with prevalence pi, let f1/f0 be the two normal densities,
A(t)=P(S≥t|Y=1), B(t)=P(S≥t|Y=0), b(t)=pi A+(1−pi)B,
q(t)=pi A/b and T=∫f1(t)q(t)dt. Its continuous-score AP influence functions are

`IF1(s)=[q(s)−T]/pi + ∫_{−∞}^s f1(t)[1−q(t)]/b(t)dt`

`IF0(s)=−∫_{−∞}^s f1(t)q(t)/b(t)dt`.

`V_AP=[pi∫f1 IF1² +(1−pi)∫f0 IF0²]/N`.
For AUROC a, the U-statistic projection calculation is
`V_AUC=[∫f1(s)(F0(s)−a)²/pi +∫f0(s)(A(s)−a)²/(1−pi)]/N`.
The AP formula follows by contaminating the joint (Y,S) distribution and
linearizing both the precision function and positive-class averaging; it includes
prevalence variation in the unstratified patient bootstrap. No equality of AP
and trapezoidal PR-AUC is asserted. Continuous distributions have no ties.

Integration used16,000 uniform trapezoidal intervals from−9 to d+9, standard
`math.erfc`, and cumulative trapezoids for the inner integrals. Weighted mean IF
residuals were below5e−8. Assumed APs were about.63176/.64720 when a=.80 and
.80452/.81425 when a=.90; these numbers specify the hypothetical distributions,
not achieved model scores. No file or record-level score array was created.

## F. Random-module control fairness — PASS, narrow control

Code enumerates all128 seven-module masks, chooses a uniformly seeded mask
among nearest parameter costs, and includes the reference itself
(`src/dmvmat_ecg/phase2/adaptation.py:94–116`). Since the reference is in the
candidate set, the minimum parameter error is exactly zero for this unchanged
backbone. The common head is always trainable and cancels in matching.

Using the documented owner counts [32,576,31,904,31,904,31,904,33,920,41,344,8,384]
(`docs/phase2/SELECTOR_IMPLEMENTATION_REPORT.md:76–82`), exhaustive non-training
integer enumeration of all128 masks found64 distinct parameter-cost classes;
**every exact-cost class has one module cardinality**. Thus parameter matching
also matches the number of adapted modules for this particular model, although
the function does not explicitly constrain cardinality. This is a verified
architecture-specific equivalent, not a general guarantee for future backbones.

Important limitation: each eligible set has only1 or3 masks. Alternative exact-
cost choices can permute **only view_2/view_3/view_4**, which share a size;
view_1, view_5 and both fusion decisions remain fixed. There may be no distinct
random control at all. Equality probability is1 or1/3, and must be reported.
It is not a broad random-location ablation over all clinical views/fusions.

There is one random mask choice per declared episode, four fitted random jobs.
Seed5101 is used for draw4101 and5102 for4102
(`src/dmvmat_ecg/phase2b/contract.py:42–60`; Phase2D executor`:135–136`). G and
source initialization are the same across F tiers, evidence RNG is fixed and F
is not used by the selector (`phase2b/integration.py:73–100`), so the two tiers
reuse a mask choice for a given G/cost. There are **at most two distinct seeded
random choices**, not four independent random replicates. No additional masks
or fits are proposed within the existing budget; interpretation must acknowledge
this restricted control. Future stronger random controls need separate approval.

## G. Complete target-label accounting — PASS

F128/F512, G256, E128, C128 are disjoint in each episode. No P, meta-training
or architecture-selection S pool exists. Source pretraining/stopping labels are
shared source costs; cohort-auditor/stratification and evaluation labels are
additional disclosed study-level access, not free clinical annotation.

| Method | Gradient labels | Evidence/mask-selection labels | Stopping | Calibration/threshold | Distinct target people, small /large | Observed target-label cells, small /large |
|---|---|---|---|---|---:|---:|
| Raw source-only |none|none|none target|none|0 /0|0 /0|
| Calibrated source-only |none|none|none target|C128|128 /128|256 /256|
| Head-F |F|none|E128|C128|384 /768|768 /1,536|
| Full-F |F|none|E128|C128|384 /768|768 /1,536|
| Head-cost |F∪G|none additional|E128|C128|640 /1,024|1,280 /2,048|
| Full-cost |F∪G|none additional|E128|C128|640 /1,024|1,280 /2,048|
| Random-A |F|G256 to determine A-matched cost|E128|C128|640 /1,024|1,280 /2,048|
| Selective A |F|G256|E128|C128|640 /1,024|1,280 /2,048|
| Selective B |F|G256|E128|C128|640 /1,024|1,280 /2,048|
| Target-only |F∪G, fresh model|none additional|E128|C128|640 /1,024|1,280 /2,048|

Cells count the two observed annotation statuses per person, not two independent
people or verified clinical negatives. Repeated roles/draw overlap are union-
counted when reporting study-wide totals; the table is per episode. C cannot
select checkpoints/masks, and source-only calibration is not zero-target-label
prediction. Full-cost and Head-cost can train on G, so they have the same total
accessible target labels as A, B and Random. Full-F/Head-F intentionally have
smaller budgets and cannot be the sole fairness claim.

Evidence: `INTERNAL_TRANSFER_PILOT_PROPOSAL.md:136–165`;
`AMENDED_SPLIT_EXECUTION_INTEGRATION.md:69–87`;
`src/dmvmat_ecg/phase2b/contract.py:46–60`.
All adaptation methods have maximum40/160 updates, including the larger F∪G
pool; they do not have equal data passes or measured FLOPs. Shared new source,
fixed E rules and epoch-zero eligibility are documented. Target-only's short
scratch schedule may underfit and uses its own permitted normalization; it is
not asserted to be optimally trained. Read-only review found no hidden extra
selector-label role in the proposed primary comparison.

## H. Approval enforcement and authoritative ledger — FAIL

### What is correctly present

Unsigned adoption is rejected; adoption and development grants are separate
(`src/dmvmat_ecg/phase2d/contract.py:65–89`; `ledger.py:84–110`). The review binds
the exact candidate hash and execution-contract hash; the latter transitively
binds endpoint, canonical policy, matrix and36 ceiling. The development receipt
binds namespace, worker manifest, code, dependencies, contract and the adoption
record's semantic hash. Separate evaluation authorization is required after lock
(`ledger.py:112–122`). These are operator-supplied authority records, not
cryptographically authenticated reviewer identities; that disclosed trust
boundary is not itself a defect.

`run_job()` checks integrity then reserves before signal loading/model creation
(`execution.py:41–54`). Actual data reads go through role membership checks
(`phase2b/access.py:49–108`); O has a distinct post-lock lease and prediction
seal before labels (`phase2d/access.py:94–119`). Default-denial is real code,
not only a CLI flag. Both real ledgers currently remain draft,0 attempts/0
accesses; no adoption/development/evaluation record exists. This audit did not
exercise or change their permission state.

### H1 — blocking: live pin inventory is not bound to signed code identity

`AmendedAccess.__init__` accepts **any nonempty** caller-supplied `integrity_files`
(`phase2d/access.py:90–92`). Its inherited `_verify()` checks only the files
supplied, plus one uv.lock/dependency check (`phase2b/access.py:33–47`). It never
computes `code_identity(integrity_files)` and compares it with the ledger's
approved `identity['code_hash']`, and never requires the full expected path set.

Consequently, a partial map containing the valid uv.lock/dependency context can
omit model/training/selector code while satisfying that local verification logic.
Calling `run_job()` first does not repair it: that function invokes the same
incomplete `_verify()`. This is a static API finding; **no such map was supplied,
no fit was reserved and no data access was attempted** by the auditor.

The preparation script correctly builds a full inventory and computes the ledger
hash (`scripts/phase2d_prepare.py:47–70`), but correctness at one CLI caller is
not enforcement at the read/fit boundary. The prior blanket claim that any live
code change fails access (`AMENDED_SPLIT_EXECUTION_INTEGRATION.md:58–62`) is
therefore stronger than the implemented API guarantee.

### H2 — blocking:36 and single-use evaluation are per database, not per study

`AmendedLedger(folder,...)` resolves and creates a caller-selected directory and
opens `folder/study.sqlite` (`phase2d/ledger.py:35–65`). Real namespace is checked,
but there is no canonical real study-path/registry restriction. The signed
identity contains namespace/manifest/code/dependency/contract only (`:47–53`),
not a unique ledger ID or approved path.

An existing different-identity DB is correctly rejected. A **new directory with
the same identity**, however, receives a fresh draft database (`:77–80`). The
same adoption/development identity can be replayed there because its expected
receipt does not distinguish paths/ledger instances (`:92–110`). Atomic
`BEGIN IMMEDIATE` and the36 count work inside one DB
(`phase2b/ledger.py:141–198`), not across those independent databases. The
single-use evaluation access counter is likewise DB-local (`:345–364`).

There is one intended real ledger today, but the constructor does not enforce
that it is the **only authoritative** ledger. No duplicate was created or
reservation made in this audit. This is not a claim of actual budget overrun or
O leakage. It is a missing enforceable invariant for the promised global ceiling
and one locked evaluation. Mere replay into a different folder does not require
editing the original SQLite records.

### H3 — required hardening: revalidate authority and signed protocol scope

At grant time both records are checked. At fit reservation the inherited guard
checks phase and a truthy development receipt, not the exact current receipt/
adoption hash (`phase2b/ledger.py:141–143`). At active-attempt reads it checks
phase, status and identity, not both approval records (`:203–212`). C reads rely
on phase/completion (`phase2b/access.py:57–65`). These are normally reached through
the proper transitions, but missing, stale or inconsistent authority records
are not independently fail-closed at every requested boundary. This is a
robustness gap, not a demonstrated attack through arbitrary OS privileges.

Also, the current code inventory includes only two older protocol documents
(`phase2b/runtime.py:32–48`), not the Phase2D adoption review, full method or split
amendment documents. The review JSON itself signs contract/candidate hashes,
not a complete analysis-document bundle (`phase2d/contract.py:51–78`). Code and
contract carry important transitive protections, but exact human-reviewed analysis
text should have an explicit signed bundle/precedence manifest, rather than
relying on reconciling several drafts whose headings remain historical.

These findings supersede the earlier engineering “ready” recommendation. A
passing test count does not establish properties those tests did not exercise.
Correct them only in a separately authorized implementation sprint, add synthetic
negative tests, preserve all old artifacts, then re-audit and regenerate the
unsigned review package. Do not sign the current identity hoping later fixes
will leave the approval valid.

## I. Exact fit matrix — PASS

| Slot family | Episodes /seeds | Predictor fits |
|---|---|---:|
| Fresh source |seed3101, source_train fitting/source_stop stopping|1|
| Head-F |4 target episodes|4|
| Head-cost |4 target episodes|4|
| Full-F |4 target episodes|4|
| Full-cost |4 target episodes|4|
| Random-A |4 target episodes|4|
| Selective A |4 target episodes|4|
| Selective B |4 target episodes|4|
| Target-only scratch |4 target episodes|4|
| Planned total |1+8×4|33|
| Shared technical failure reserve |not extra search or optional candidates|3|
| Hard maximum, including failures/retries |per intended authoritative study|36|

Four episodes are draw4101/F128, draw4101/F512 (training seed5101) and
4102/F128,4102/F512 (training seed5102). There is **not** an independent cross
product of two training seeds with each draw: seeds are paired to draws.
Source-only predictions alias the single new source checkpoint, adding no fit.
Up to33 two-label calibrators/66 label mappings are separate calibration work,
not more predictor slots. There is no P role, strong-XResNet transfer extension
or external20/172-fit budget in this matrix.

The JSON declares33 planned jobs and ceiling36; source generation loops match
it exactly (`phase2b/contract.py:31–62`; internal protocol`:175–189`). Failures,
interruptions and predictive smoke fits must consume capacity; no favorable
reruns or use of freed aliases for new policies. H2 must be fixed before claiming
this ceiling is globally enforced rather than merely correct arithmetic.

## J. Domain-shift realism — UNCLEAR, scope is internal

Age18–59 source /60–89 target, based on the canonical eligible recording's
recorded age; age is not an added neural input. Source domain7,638 people
(train6,492/stop1,146). Target domain8,777 people, of whom6,144 are in non-O
base partitions; O's proposed2,633 are excluded from prevalence calculation.

| Domain | Included non-O patient count | CD prevalence | STTC prevalence | Joint-positive prevalence |
|---|---:|---:|---:|---:|
|Source train∪stop|7,638|1,064/7,638=13.93035%|1,098/7,638=14.37549%|161/7,638=2.10788%|
|Target A∪E-reserve∪C-reserve|6,144|1,766/6,144=28.74349%|1,863/6,144=30.32227%|409/6,144=6.65690%|

Evidence: `INTERNAL_TRANSFER_PILOT_PROPOSAL.md:37–45`,
`JOINT_PATTERN_FEASIBILITY.md:58`,`:96`,`:98`,`:100`. Do not use the whole-target
row to recompute prevalences for this audit; that would include O counts.

There is genuine **age-covariate support separation by construction** and a
large observed annotation-prevalence/co-occurrence difference. The audit cannot
say whether model-relevant waveform/representation shift is mainly conditional
covariate shift, label shift or their mixture: no permitted learned-ECG shift
analysis has been executed. MMD on marginal representations can respond to
prevalence changes without proving that adapting a module is beneficial.

This is an **internal same-dataset/cohort age-subgroup pilot**, not cross-hospital,
cross-country or independent clinical validation. The requested phrase “same-
hospital” cannot be verified literally: the protocol does not restrict/verify
one hospital and explicitly forbids inventing sites. Do not replace one
unsupported external-validation claim with an unsupported single-hospital claim.
Use the precise same-dataset description until facility provenance is established.
No domain boundary or participant choice should change to improve apparent shift.

## K. Environment — UNCLEAR

Read-only command observations (no files moved or settings changed):

- `/Users/kiyo/Desktop` exists, is not a symlink, resolves to itself.
- `~/Library/Mobile Documents/com~apple~CloudDocs/Desktop` does not exist;
  consequently it is not the same filesystem object as `~/Desktop`.
- `/usr/bin/xattr ~/Desktop` lists only `com.apple.macl`; this security attribute
  does not establish cloud synchronization state.
- `/usr/bin/brctl status` initially refused a sandboxed process. A separately
  approved **read-only** query outside the sandbox succeeded, listing26 cloud
  containers including an active CloudDocs container, but no Desktop-specific
  binding or Desktop-sync setting. No raw status tokens/account metadata are
  reproduced here.

These checks provide no positive filesystem evidence that Desktop is redirected
to the tested CloudDocs location, but do **not certify that Desktop sync is off**.
Confirm the Desktop & Documents setting and actual storage provider before
running authoritative SQLite/WAL/checkpoint workloads. Any relocation requires
separate permission; this audit neither moves nor reconfigures anything.
This environmental uncertainty is independent of the demonstrated code gaps H1/H2.

## Approval fields: human checklist, not an approval payload

**Do not fill/sign the current record yet.** After authorized fixes and a new
read-only audit, derive the then-current hashes again. This checklist describes
existing field names and provenance; it does not create a signed record or
supply a fabricated reviewer/adoption hash. Amended schemas may add fields.

### Protocol adoption record

Make a separate private copy only when authorized; leave the pinned unsigned
template unchanged. Existing source:
`configs/phase2d/HUMAN_REVIEW_UNSIGNED.json:1–85`; validator
`src/dmvmat_ecg/phase2d/contract.py:51–89`.

For **each** `decisions.<item>`, the human supplies `approved`, `reviewer`,
`reviewer_role`, `date`, `notes`. Items are:

1. `endpoint_annotation_semantics`
2. `canonical_exact_record_policy`
3. `age_domains_and_eligibility`
4. `joint_pattern_seed44301`
5. `replaced_O_and_audit_history`
6. `A_B_validity_and_abstention`
7. `matched_labels_and_updates`
8. `33_plus_3_ceiling`
9. `stopping_20_post_update_looks_plus_epoch_zero`
10. `calibration_failure_identity_and_threshold_point5`
11. `internal_exploratory_limits`

Use actual identity, responsibility and date; do not sign for another reviewer.
All decisions must actually be accepted before using the adopted-status string
specified at `contract.py:67`. A rejection/unresolved decision is not adoption.
`execution_contract_hash` comes from semantic SHA-256 of the reviewed execution
JSON; `candidate_auditor_hash` comes from the exact candidate's prior manifest
receipt/constant (`phase2d/contract.py:12`). `status` is a human decision, not an
audit result. `training_authorization` and `evaluation_authorization` stay false
in adoption, as required by `contract.py:69–78`. No field was populated here.

### Separate training/development authorization

After actual adoption, existing expected fields are defined at
`src/dmvmat_ecg/phase2d/ledger.py:94–104`:

| Field | Where the human obtains its exact reviewed value |
|---|---|
|`namespace`|Authoritative ledger `meta.identity`, cross-check `EXECUTION_READINESS.json` and `phase2d/contract.py:13`|
|`manifest_hash`|Approved sealed worker-manifest semantic digest, same as authoritative `meta.identity`; includes per-role metadata/vault hashes|
|`code_hash`|Complete reviewed source/config inventory via `phase2b/runtime.py:32–54`, cross-check authoritative ledger identity; completeness enforcement must first be fixed|
|`dependency_hash`|Pinned environment receipt semantic digest, same authoritative ledger identity; uv.lock raw SHA cross-check|
|`contract_hash`|Exact reviewed execution configuration semantic digest; binds endpoint/method matrix/36 ceiling|
|`statement`|The exact separate development-authorization statement required by `phase2d/ledger.py:96`, entered only after actual authority is granted|
|`adoption_hash`|Semantic digest of the **actual human-completed adoption record** stored in `meta.protocol_adoption`; no value exists yet and none should be inferred from the blank template|

Current schema transitively binds endpoint/matrix/ceiling via the contract hash.
The proposed amendments require additional canonical ledger identity and signed
protocol-bundle fields, to be defined/verified before signing. Do not append
unsupported fields to today's strict receipt and assume it will validate.
Evaluation requires a different future statement plus `lock_hash` after complete
development locking (`phase2d/ledger.py:112–122`); it is not part of training
permission. This audit supplies neither receipt nor authorization.

### Current identity reference, not values prefilled into an approval

Semantic hashes use SHA-256 of `json.dumps(object,sort_keys=True,allow_nan=False)`
with the project's default separators, **not** raw-file SHA
(`src/dmvmat_ecg/phase2b/contract.py:19–20`). Values below identify the audited
state only and will need regeneration if the proposed fixes change it.

| Artifact/identity | Current digest |
|---|---|
|Execution contract semantic digest|`dbc0afe9bee29e7323a3a353b37660b8dff20733e3fe62fa94958d95400ecb28`|
|Exact candidate digest from existing receipt (not re-decoded here)|`c809d6db691d7ad3094dcd3635546ac58941529d1c6a3d06be35129f8ca0f4c9`|
|Worker manifest identity from administrative ledger|`65cf652084396403077918e0e9be560eaedd009ab4fc8587260c833ddcf4f816`|
|Code inventory identity from administrative ledger|`5440faf264ba3e76aa432556c6996146e2dee5fa23595dbf0caf677b5c7cc942`|
|Dependency identity from administrative ledger|`195cd5ae983b6a3e196bc52ac0cf0997b26f60f45fb801e53256b8b57f881424`|
|Raw execution JSON SHA-256|`787da75edfb90fec89e340ceead0c8a568d69adfd51f565fe35036feea7491f6`|
|Raw unsigned-review-template SHA-256|`e85eb4249cd1eea06bc82193a8a13fae9bc9ac78f788702eb981cb18498f144a`|
|Raw uv.lock SHA-256|`a69f264030cf30ee4392cabc05e72a2184a9441ff5fd50e92092015f65b4bed3`|
|Raw adoption-review Markdown SHA-256|`3238fb14d6eaab66f6b4f24c8f59abcb8b19b5143c901b433a79569f64b48172`|
|Raw internal-pilot-proposal SHA-256|`a5a2408f7b04e59624497a9c91a692788b9965d55dee635aac01950c314effe6`|
|Raw method Markdown SHA-256|`0d3d73fe01d2c8c7666fe599437426962882b377ca36e41037dd7aa6698ef9f3`|
|Raw split-amendment Markdown SHA-256|`6ba42b1803f25c86d659e8ccbeaab3215c6a9cee7ad2a2401e68fce2bf7709ac`|

## Final disposition and file boundary

The foundation, primary method/endpoint, abstention procedure, declared label
budgets and matrix are coherent. The unresolved power and hospital/cloud claims
need qualified wording. **H1/H2 are the blocking execution-integrity defects**;
H3 strengthens the binding of what a human signs to what actually runs.
No evidence of an actual unauthorized fit, budget overrun or O prediction exists.
Both real ledgers were observed draft,0/36, no adoption/training/evaluation grant.

Only this audit and the optional unapplied amendment draft were created. No
protocol/config/code/manifest/checkpoint/approval/ledger was edited; no test,
predictor, source normalization, calibration or evaluation was executed. No
commit, push, merge, PR, workspace move or iCloud setting change. The original
fold10 results were read only as published historical aggregates, never as
individual prediction data or new tuning input. Existing186-test receipts were
not represented as a new test run. Stop pending authorized corrective work and
another pre-signature review.

Final preservation check: all164 pre-existing Git-visible files (including95
tracked baseline files) match their opening SHA-256 snapshot. Tracked/staged
diffs are empty; branch remains `main` at the historical HEAD above. The only
new Git-visible paths are this report and the UNAPPLIED amendment draft. Both
real ledgers were rechecked administratively at draft,0 attempts/0 accesses;
all unsigned approval fields remain null. No private O-container hashing or
label-file parsing was repeated for this terminal check.
