# Diagnostic crosswalk and endpoint decision

**DRAFT — semantic audit, not an adopted label-generation algorithm.**

## Evidence and status definitions

Source statements were read from the existing official
`data/raw/ptbxl_v1_0_3/scp_statements.csv`, SHA-256
`ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f`.
This audit read the statement dictionary, not source participant labels.
Existing five-class [label code](../../src/dmvmat_ecg/labels/superclasses.py):9–27
uses all diagnostic keys, including likelihood zero. That historical convention
is preserved; it is not automatically adopted as clinical truth for Phase 2.

Target dictionaries are the official combined-release condition mapping [D6]
and Challenge mapping files pinned at
`e2a75fc01f729cb74cc4e853e054ce81e28381fc`:

- [scored mapping](https://github.com/physionetchallenges/evaluation-2021/blob/e2a75fc01f729cb74cc4e853e054ce81e28381fc/dx_mapping_scored.csv),
  SHA-256 `fad13ad9f7ca230e7e6392ac8a264cb7cd157879525129f964c5f708eabb41d0`;
- [unscored mapping](https://github.com/physionetchallenges/evaluation-2021/blob/e2a75fc01f729cb74cc4e853e054ce81e28381fc/dx_mapping_unscored.csv),
  SHA-256 `ce53c9e35406e922f1634a38bb13c31e8cf7603291641a16c201d8e679f4fe94`.

D1–D7 resolve to official sources in [dataset audit](EXTERNAL_ECG_DATASET_AUDIT.md).
“Verified positive correspondence” means dictionaries identify the same named
phenotype; it is not independently adjudicated diagnostic accuracy. “Unknown
negative” means absence of a code does not establish clinical absence.
Challenge scoring equivalence is a competition convention, not proof that
two diagnoses or their negative classes are medically identical [D7].

## Five-superclass assessment

| Source superclass/statements | Target code/meaning considered | Positive correspondence | Reliable negative? / decision |
|---|---|---|---|
| NORM: `NORM`, normal ECG | `426783006`, sinus rhythm (NSR/SR) | **Rejected**: rhythm can coexist with infarction, hypertrophy or conduction abnormalities | Neither SR nor “no other listed code” proves normal ECG. No verified complete normal-ECG target label |
| MI: `IMI`, `AMI`, `ASMI`, `LMI`, `PMI`, other regional patterns **and injury statements** | `164865005` unspecified MI; `57054005` acute MI; `54329005` anterior MI; older/chronic infarction codes | Partial concepts only; history, acute disease and ECG patterns have different observation semantics | Missing MI code is not a negative. Do not OR these into the source MI superclass without clinical adjudication and timing definitions |
| STTC: `NDT`, `NST_`, `ISC_` and regional ischaemic changes; also `DIG`, `LNGQT`, `EL`, `ANEUR` | `164934002` T-wave abnormality, `428750005` nonspecific ST-T abnormality, `429622005` ST depression, `164930006` ST elevation, `59931005` T inversion | Some specific findings overlap; they do not exhaust source STTC | Unlisted repolarization findings are unknown. A T-abnormality union is not the complete STTC endpoint |
| CD: `1AVB`, `2AVB`, `3AVB`, `CRBBB`, `IRBBB`, `CLBBB`, `ILBBB`, `LAFB`, `LPFB`, `IVCD`, `WPW` | `270492004` IAVB; `713426002` IRBBB; `713427006` CRBBB; `59118001` unspecified RBBB; `164909002` LBBB; `733534002` CLBBB; `445118002` LAFB; `698252002` nonspecific intraventricular block | Individual subtype candidates exist; broad CD coverage and completeness are not established | Unspecified, complete and incomplete blocks cannot be merged silently. No reliable superclass negative defined |
| HYP: `LVH`, `RVH`, `LAO/LAE`, `RAO/RAE`, `SEHYP` | `164873001` LVH; `89792004` RVH; `446358003` right atrial hypertrophy; atrial hypertrophy codes | Partial correspondence, not a full superclass | High voltage, chamber enlargement/overload and anatomically verified hypertrophy are not interchangeable. No verified full HYP negative |

**Reject a direct five-superclass transfer claim under the presently documented
crosswalk.** This is an endpoint/provenance result, not an empirical model result.

## Code collisions requiring special protection

The official combined dictionary maps `LBBB`, `LBBBB` (“left back bundle branch
block”) and `LFBBB` (“left front bundle branch block”) to **164909002**. Thus
conversion may erase distinctions needed for LBBB versus fascicular-block
phenotypes. A target LBBB code cannot safely be reverse-mapped to a specific
PTB-XL subtype. Recover original labels/curator clarification or mark unknown.

Other traps: external abbreviation `AF` can mean atrial flutter (`164890007`),
whereas Challenge abbreviation AF denotes fibrillation (`164889003`). Unscored
`CD`/`698247007` means **cardiac dysrhythmia**, not the PTB-XL conduction superclass.
Use versioned numeric codes and meanings, not acronym/string matches. The
combined condition dictionary also does not enumerate every code appearing in
Challenge count tables, including the IRBBB count reported for Ningbo. Resolve
that coverage/version discrepancy before constructing its labels.

## Proposed narrower task: two conduction phenotypes

An explicitly **different**, two-label ECG morphology task is the best candidate
for a first multi-label transfer study. It is chosen for identifiable clinical
concepts and potential support, not anticipated AUROC.

| Endpoint | PTB-XL source candidate | Target candidate | Current verdict |
|---|---|---|---|
| First-degree AV block (IAVB) | `1AVB`, “first degree AV block” | `270492004`, “1st degree av block”; original `1AVB` | Positive concept corresponds; generic AV block `233917008` and prolonged PR alone `164947007` are **not automatic substitutes** |
| Incomplete right bundle branch block (IRBBB) | `IRBBB`, “incomplete right bundle branch block” | `713426002`, IRBBB | Positive concept corresponds; generic RBBB `59118001` and complete RBBB `713427006` are **not substitutes** |

For both endpoints, negatives remain unverified. Require documented comprehensive
review for that specific phenotype or blinded expert adjudication. A normal
sinus-rhythm tag alone supplies neither negative. Keep an observation mask
`O[i,label]`: affirmative/reliably absent → observed 1 with target 1/0;
unassessed, ambiguous, conflicting or conversion-collapsed → observed 0, excluded
from that label's loss, metric and geometry. Do not fill missing labels with zero.
Source zero-likelihood keys require adjudication/approved interpretation; do not
silently inherit the old any-key rule or invent a confidence threshold.

Expert review should use two independent ECG readers and a third for disagreement,
blinded to model scores and method identity; retain source reports/confidence,
adjudication and unresolved status. If diagnosis-complete negative coverage is
affirmed by curators instead, document its release scope and validation evidence.
Assess selective verification bias: performance on the observed subset does not
automatically generalize to the unobserved population.

The new task needs a two-output head and new source-supervised task adaptation.
The current broad CD head is not an IAVB/IRBBB classifier. Preserving the old
backbone while fitting a new head does not preserve the old endpoint or make its
five-class score comparable. Expected five-view count: 211,936 internal parameters
+ 130 head = **212,066**, to verify synthetically after implementation approval.

## Published support indications, not eligibility counts

The pinned scored mapping reports IAVB counts Georgia 769 / Chapman 247 /
Ningbo 893; IRBBB counts 407 / 0 / 246. These are **record/code counts** before
duration, duplicate, patient, label-certainty and observation-mask checks.
Zero IRBBB in Chapman is absence of recorded examples, not proof of 10,247
negative ECGs. Consequently Chapman is not presently supported for this two-label
task; a single-label IAVB study would be a separate protocol, not a quiet fallback.

No reliable count of negative patients or feasible positive quotas has been
established. Source counts in the Challenge mapping refer to an older PTB-XL
conversion, not our exact eligible source pool. No source/target per-person
label audit was executed in this design sprint.

## Other endpoints and acceptance gate

AFIB (`164889003`) is a potentially coherent future rhythm task; PTB-XL has a
rhythm statement but our current diagnostic head does not predict it. AFIB versus
adjudicated sinus rhythm would change population/task, need a new source head,
handle transitions/co-occurrence and exclude unresolved cases. It is not an
approved replacement chosen for an easier score.

An annotation-presence benchmark could label “code recorded/not recorded,” but
its negative means no recorded code, not no clinical abnormality. It may be an
explicitly labeled engineering control; it cannot answer the proposed clinical
transfer question or bypass this sprint's missing-negative restriction.

**Go/no-go:** obtain clinical sign-off on positive meanings, reliable negatives,
confidence/conflict handling, coverage, timing and code-version crosswalk;
then freeze eligible counts and label manifest hashes before training. Until
then even the narrower task is **BLOCKED for predictive execution**.

## Phase 2A negative-label gate

The official evaluator initializes missing code entries to zero
([`load_labels`](https://github.com/physionetchallenges/evaluation-2021/blob/e2a75fc01f729cb74cc4e853e054ce81e28381fc/evaluate_model.py#L123)).
That software convention verifies annotation-presence scoring, not exhaustive
clinical absence. Neither IAVB nor IRBBB currently has an established admissible
clinical-negative population here. Thus **external transfer remains blocked**;
eligible unique-patient counts cannot be calculated by subtracting the published
positive-code counts from 10,344. [Evidence resolution](EXTERNAL_EVIDENCE_RESOLUTION.md)
defines true-positive confirmation, true-negative evidence, unknowns and exclusions.

The proposed internal CD/STTC annotation pilot is an explicitly different
within-release task with new source training. It does not resolve or replace
this external clinical crosswalk, and is not authorized by the synthetic sprint.
