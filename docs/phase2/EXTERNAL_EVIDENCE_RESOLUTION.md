# Phase 2A external evidence resolution

**2026-11-12. Decision: EXTERNAL CLINICAL TRANSFER — BLOCKED.**

The scientific question remains whether evidence-guided selective adaptation
improves target classification versus full fine-tuning under matched label
budgets. Neither architecture optimization nor maximizing an already opened
PTB-XL score substitutes for that question. Engineering progress does not resolve
clinical label completeness or patient identity.

## Evidence reviewed and what it does establish

The original public Georgia distribution is documented through the
[Challenge 2020 release, Data Description/Data Access](https://physionet.org/content/challenge-2020/1.0.2/).
It describes 10,344 public recordings at 500 Hz and record-level diagnoses in
WFDB headers. The later
[Challenge 2021 release, Data Sources/Data Format](https://physionet.org/content/challenge-2021/1.0.3/)
allows 5–10-second Georgia recordings. Use the actual header rather than assume
every recording is ten seconds. The public waveform archive was not downloaded.

The Challenge authors' [methods paper, §2.1 and Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC9469795/)
reports **15,738 Georgia patients across 20,672 recordings**, with repeated
patients confined to one official Challenge partition. It does not give a
public training-file-to-patient crosswalk. Its annotation discussion describes
heterogeneous label review and approximate SNOMED harmonization, not a
Georgia-specific guarantee of exhaustive assessment of every phenotype.

Thus the authors' partition guarantee helps interpret their original experiment,
but cannot reconstruct our new patient-disjoint adaptation/stopping/evaluation
roles from record filenames. It also does not establish a PTB-XL/Georgia
cross-source patient linkage guarantee. A label-independent waveform audit and
curator-supported person linkage remain necessary. No such new waveform audit
or identification inference was executed here.

## Annotation semantics and software conventions

The official
[scored codebook](https://github.com/physionetchallenges/evaluation-2021/blob/e2a75fc01f729cb74cc4e853e054ce81e28381fc/dx_mapping_scored.csv)
names 270492004 as first-degree AV block and 713426002 as incomplete right
bundle branch block. These identify **recorded positive concepts**, not perfect
clinical ground truth, adjudication certainty, completeness or confirmed negatives.
The pinned mapping's 769/407 Georgia code counts remain recording counts, not
eligible unique patients. The earlier [crosswalk](DIAGNOSTIC_LABEL_CROSSWALK.md)
contains its checksums. No defensible negative/eligible-patient count is available.

The official evaluator's
[`load_labels`, lines 123–141](https://github.com/physionetchallenges/evaluation-2021/blob/e2a75fc01f729cb74cc4e853e054ce81e28381fc/evaluate_model.py#L123)
starts a zero matrix and marks listed codes positive. That is the competition's
**annotation-presence scoring convention**. It is not an independent clinical
validation that every zero means a physician excluded that phenotype. Using
this convention would require explicitly changing the endpoint to predicting
recorded codes; it cannot silently clear our clinical-negative gate.

The release links a public
[label-mapping workbook](https://docs.google.com/spreadsheets/d/1Q4m9axOlE1rEb7Fi2t4fPbvpw8JPvikLBO_j-lQcuuE/edit?usp=sharing).
Its view-only default table was readable, but the browser returned an error for
full workbook navigation. It supplies no inspected Georgia-specific negative
ascertainment guarantee. Do not call an inaccessible sheet verified. No workbook
or participant table was saved. Searches did not identify an admissible original
Georgia document establishing exhaustive negative assessment or a public stable
patient mapping. This is **missing evidence**, not proof such records cannot exist.

## Endpoint rules pending clinical sign-off

| Endpoint | Positive evidence | Negative evidence required | Unknown/ambiguous | Current exclusions/support |
|---|---|---|---|---|
| IAVB | Exact target code 270492004 corresponds to PTB-XL `1AVB`; clinical-positive status additionally needs an interpretable, diagnosis-confirmed ECG/report | Explicit phenotype-negative adjudication, or documented complete assessment with validated absence semantics for this release | Unlisted code, generic AV block, isolated prolonged-PR code without approved diagnostic interpretation, uncertain/conflicting reports | Exclude unresolved cases from this label's clinical loss/metrics, never recode zero; published 769 code records do not yield an eligible count |
| IRBBB | Exact target code 713426002 corresponds to PTB-XL `IRBBB`; validate diagnostic criteria/report certainty | Explicit phenotype-negative adjudication or validated comprehensive assessment as above | Unlisted code, unspecified RBBB, conflicting incomplete/complete labels, uninterpretable ECG | No automatic merge with complete/unspecified RBBB; published 407 code records do not yield an eligible count |

A sinus-rhythm code supplies neither negative. Demographics, apparent absence
of other abnormalities or another diagnosis cannot manufacture negatives.
Maintain Y and an observed-status mask separately. If a record has one resolved
and one unknown endpoint, only the resolved endpoint may contribute, with coverage
reported; primary fully ascertained populations must be fixed before fitting.
Exclusion of unresolved records may introduce verification/spectrum bias.
No AUROC/AP is computable for a clinical endpoint using only verified positives;
the software correctly abstains rather than inventing comparison controls.

Source/target code names are compatible concepts, not proof of equivalent reader
criteria or clinical ascertainment. Source likelihood-zero statements, temporal
meaning, report revisions and target conversion errors also need sign-off.
No new source participant-label audit was performed. Clinical adjudication should
be blinded to algorithms/scores, use two independent readers with third-reader
disagreement resolution, and retain unresolved masks. No clinical cutoffs were
invented or fitted in this sprint.

## Independence matrix

| Guarantee | Status | Consequence |
|---|---|---|
| Different published acquisition provenance | Supported by official sources | Supports a cross-database question; does not prove different people |
| Original Challenge repeated-person split isolation | Authors report it | Cannot infer new roles' patient assignments without their linkage |
| Stable public Georgia person key/crosswalk | Not established by reviewed release material | Record-ID splitting is insufficient; repeated-person leakage possible |
| Entire Challenge independent of PTB-XL | False: source is included | Exclude the whole PTB-XL component, historical copies and conservatively PTB |
| Georgia 2020 versus Georgia 2021 independent | No; versions of shared distribution | Pin one release, audit conversions/duplicate exports; do not treat names as cohorts |
| No cross-source identical/cropped/resampled waveforms | Not tested here | Requires exact/physical-signal/near-duplicate audit before transfer |
| No shared patient with different ECGs across databases | Unverified | Waveform deduplication alone cannot establish it |

Pseudonymous patient IDs can support within-release grouping if documented and
stable, but do not by themselves link different institutions/releases. Anonymous
record IDs cannot substitute. No patient identity was guessed from age/sex or
record numbering. No emails/messages to curators were sent in this sprint.

## Decision and specific next evidence

**External transfer remains blocked.** Reduced IAVB/IRBBB may be feasible, but
neither has verified clinical negatives or a usable patient-linkage guarantee.
Require (1) curator/release linkage documentation and repeat-record crosswalk,
(2) complete phenotype assessment or independent adjudication with observation
masks, (3) cross-source/cross-release overlap audit under a separately approved
data-preparation step, and (4) eligible per-label patient counts after exclusions.
Only then can a new external execution protocol be approved. Do not contact or
download complete archives automatically to bypass this gate.

An internal patient-disjoint PTB-XL pilot can advance the **same selective-versus-
full question**, while explicitly answering a weaker, internal annotation-task
question. See [the pilot proposal](INTERNAL_TRANSFER_PILOT_PROPOSAL.md). It must
train new source models from scratch; none of the historical checkpoints is
certified independent of newly assigned target participants.
