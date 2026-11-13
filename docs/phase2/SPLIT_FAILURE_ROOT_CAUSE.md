# Phase 2C — original split root cause

2026-11-13. Metadata-only audit; no real waveform access, model fitting,
representation extraction, inference or predictive evaluation. All preceding
Phase2/2A/2B reports remain unchanged. The internal protocol remains a draft.

## Finding: allocation defect superimposed on real age-domain differences

The fixed age domains really differ in recorded diagnosis distributions: source
CD/STTC prevalence13.9303%/14.3755%, target28.7456%/30.3293%; joint positives
161/7,638 (2.1079%) versus585/8,777 (6.6651%). These are annotation counts,
not a causal age effect. The **19 training /142 stopping /zero target-A** pattern
is an additional, artificial within-domain allocation failure. The target
population did not lack joint cases. The age boundary did not remove them.

The domain and endpoint appear prospectively in
[Phase2A proposal §2](INTERNAL_TRANSFER_PILOT_PROPOSAL.md), before any transfer
fits, and the Phase2B code/manifest. These uncommitted draft documents are not
an adopted registration or independent timestamped preregistration. No real
transfer performance exists to have selected the age threshold; the opened
historical classification benchmark is a separate source of research-direction
bias and is not used to redesign these domains.

## Exact implementation and mechanism

Evidence is the unchanged [Phase2B preflight](../../src/dmvmat_ecg/phase2b/preflight.py):

- Lines21–24 sort by **joint-pattern frequency**, rarest first, then
  `digest([seed, patient_id])`. In both domains11 is rarest, so all double-positive
  patients arrive before every other pattern. This is not iterative allocation
  that revisits previously assigned cases.
- Lines25–26 construct five columns `[1,CD,STTC,1−CD,1−STTC]` and desired counts
  for each role. There is **no joint-cell column**. Pattern-aware sorting is not
  pattern-aware balancing. Equal formal coefficients also double-represent each
  label through positive/negative columns.
- Lines29–34 minimize the **incremental** sum of squared relative deviations,
  `Σ_j[((a_rj+v_j−d_rj)/max(1,d_rj))²−((a_rj−d_rj)/max(1,d_rj))²]`.
  For positive desired counts at initialization the term is
  `−2v_j/d_rj+v_j²/d_rj²`. Smaller intended roles therefore initially gain more
  by accepting an early positive. Squared inverse desired-count scaling gives
  smaller/rarer counts more leverage; equal term coefficients do not produce
  equal effective penalties. No role-capacity hard constraint prevents this.
- Source first11 deltas are train **−0.00466000744**, stop **−0.02634339157**:
  stopping wins. Target first11 deltas are O **−0.00590284290**, E/C reserves
  **−0.01768778232** each, A **−0.00354253559**: E/C win initially.
  Subsequent greedy choices consume all585 joint positives before A can receive
  any; there is no later swap/backtracking to repair their joint allocation.
- `np.argmin` at line32 breaks exact ties by role order. The identical E/C10%
  quotas favor E on their first tie; final209/208 split differs by one. Source
  order train/stop and target O/E/C/A affect exact ties, but tie order alone is
  not the main failure. Patient hash ties cannot fix an objective that omits
  co-occurrence. No alternate seed was searched here.
- Lines144–160 first select one age-eligible lowest-ID record per **actual
  patient**, then form age18–59 source /60–89 target. Each allocation unit has
  weight1. Extra recordings do not increase a person's objective weight. There
  is no descending multi-record-size term at this stage because canonicalization
  removed that multiplicity. Unknown labels are not supported by this old
  splitter; the actual annotation-presence task has no unknown entries.
- Lines161–175 assign roles and only afterwards draw E/C and G/F. The bias is
  already in the base roles; G sampling did not create A's zero cell.
  Lines178–182 checked marginal/capacity floors only, so did not catch it.

[Instrumented diagnostics](../../src/dmvmat_ecg/phase2c/diagnostics.py):13
re-executed this exact metadata arithmetic and reproduced **all original base
role aggregate counts**, including19/142 and209/208/168/0, without changing the
original manifest. Traces/deltas are preserved in the [metadata receipt](../../results/phase2c/METADATA_PREFLIGHT.json). They contain
no predictions or patient rows.

## Classification of the failure

This is a **split-objective/design defect**: rare-pattern ordering, inverse-quota
relative penalties, absent joint constraints and irreversible greedy allocation
interact. There is no evidence of swapped CD/STTC columns, lost patients,
record duplication, random patient leakage, waveform-dependent exclusion or
clinically wrong labels causing this outcome. The implementation matches the
Phase2B documented interpretation of the draft; the older prose did not fully
specify normalization placement. We do not claim an adopted algorithm was
violated, or silently switch to a different denominator interpretation.

Real age-domain prevalence differences remain important and are preserved.
The amendment corrects within-domain allocation without making the domains
label-identical, changing the age threshold or relabeling patients. Serial
annotation differences, canonical-record limitations and unresolved clinical
negative semantics are separately documented in
[feasibility](JOINT_PATTERN_FEASIBILITY.md); they do not explain the structural
empty target reservoir.

## Remedy proposed, not adopted

Use one patient-level canonical unit and allocate **each observed joint pattern
proportionally within its original domain**, with exact rational Hamilton quotas
and fixed hashed patient/tie ordering. This avoids the flawed marginal greedy
objective. The modest2-label/single-record-per-patient case does not require a
complex iterative solver. It is described in the
[prospective amendment](SPLIT_AMENDMENT_PROPOSAL.md). The old candidate, role
vaults and zero-fit blocked ledger are preserved, not repaired in place.
