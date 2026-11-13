# Phase 2C — joint-pattern and selector-support audit

**DRAFT candidate; metadata counts only. No actual MMD, geometry, bootstrap
uncertainty, mask or predictive metric was calculated on real representations.**

## Population, observation and grouping policy

Original folds1–9 eligible benchmark records:19,230 from16,740 patient IDs.
The existing18–89 age rule retains18,857 records. Selecting the lowest eligible
ECG ID per patient retains16,415 canonical records/patients, excluding2,442
additional eligible recordings and373 age-ineligible recordings. The eligible
study population below means these16,415 canonical patients. There is no unique
four-pattern patient label for all19,230 serial records without choosing a
record or a new aggregation endpoint; no OR rule was introduced.

The label order is `(CD, STTC)`. **All tables use joint order00,10,01,11**
(CD first); this differs from the old Phase2B JSON's internal00,01,10,11 ordering.
Old rows were recounted from their full auditor manifest, not mislabeled by
reordering old numeric arrays. The [metadata receipt](../../results/phase2c/METADATA_PREFLIGHT.json) explicitly carries this order in
its specification/diagnostics and all record-free aggregate tables.

The existing task is recorded superclass-annotation presence under the original
any-key/likelihood-zero convention, not clinically confirmed CD/STTC absence.
All19,230 cached record-label pairs have observed tag-presence status; unknown
annotation entries=0. That does not establish exhaustive clinical ascertainment.
Generic candidate code preserves genuine unobserved statuses as separate `?`
strata and excludes them from positive/negative/joint counts; a real unknown
would require prospective pilot policy review, never zero-filling.

Metadata checksum/mapping boundary:
[unchanged loader](../../src/dmvmat_ecg/phase2b/preflight.py):205 verifies release1.0.3,
publisher manifest, metadata, original train/validation caches and actual patient/
fold joins; only demographics/paths/folds are parsed from the CSV. CD/STTC use
cached columns3/2 after verifying NORM/MI/STTC/CD/HYP order. No fold10 label cache,
waveform or prediction is opened. New [versioned splitter](../../src/dmvmat_ecg/phase2c/splits.py):43 canonicalizes and audits groups.

1,784 age-eligible patients have multiple recordings. Across their eligible
records, **290 patients differ in observed CD annotation,437 in STTC,625 in
joint pattern** (overlapping counts, not additive). Ten patients have eligible
serial ages on both sides of60. The pre-existing draft policy retains the
canonical record's labels and age, keeping all other records excluded; patients
cannot migrate across roles. No diagnosis adjudication, imputation or OR occurs.
Different serial annotations may reflect changing physiology, readers or incomplete
features; they are not automatically errors. There is **no adopted clinical
adjudication policy**; that remains unresolved and does not change this explicitly
record-level annotation endpoint. Review/adopt the canonical policy before training.

All age-eligible records were checked for native `.dat` hash linkage, including
noncanonical recordings: zero repeated-native-hash groups/cross-person matches.
Known real patient IDs, canonical records and native hashes remain group-disjoint.
No near-duplicate waveform search or proof of unrecorded person identity is claimed.

## Complete eligible and fixed-domain counts

| Population/role | Patients | 00 | 10 | 01 | 11 | CD+ / CD− | STTC+ / STTC− | Unknown CD / STTC |
|---|---:|---:|---:|---:|---:|---|---|---|
| complete_eligible | 16,415 | 9,814 | 2,841 | 3,014 | 746 | 3,587 / 12,828 | 3,760 / 12,655 | 0 / 0 |
| source | 7,638 | 5,637 | 903 | 937 | 161 | 1,064 / 6,574 | 1,098 / 6,540 | 0 / 0 |
| target | 8,777 | 4,177 | 1,938 | 2,077 | 585 | 2,523 / 6,254 | 2,662 / 6,115 | 0 / 0 |

No source/target record or patient changes between candidates. Source joint
prevalence2.1079% and target6.6651% remain different; no cross-domain matching.
Every listed role has100% observed annotation coverage for each label.

## Original proposed roles — preserved audit history

| Population/role | Patients | 00 | 10 | 01 | 11 | CD+ / CD− | STTC+ / STTC− | Unknown CD / STTC |
|---|---:|---:|---:|---:|---:|---|---|---|
| A | 4,208 | 2,066 | 1,031 | 1,111 | 0 | 1,031 / 3,177 | 1,111 / 3,097 | 0 / 0 |
| C | 128 | 71 | 20 | 19 | 18 | 38 / 90 | 37 / 91 | 0 / 0 |
| C_reserve | 950 | 459 | 135 | 148 | 208 | 343 / 607 | 356 / 594 | 0 / 0 |
| E | 128 | 68 | 19 | 21 | 20 | 39 / 89 | 41 / 87 | 0 / 0 |
| E_reserve | 951 | 459 | 135 | 148 | 209 | 344 / 607 | 357 / 594 | 0 / 0 |
| O | 2,668 | 1,193 | 637 | 670 | 168 | 805 / 1,863 | 838 / 1,830 | 0 / 0 |
| d4101:n128:F | 128 | 63 | 25 | 40 | 0 | 25 / 103 | 40 / 88 | 0 / 0 |
| d4101:n128:G | 256 | 129 | 59 | 68 | 0 | 59 / 197 | 68 / 188 | 0 / 0 |
| d4101:n512:F | 512 | 248 | 126 | 138 | 0 | 126 / 386 | 138 / 374 | 0 / 0 |
| d4101:n512:G | 256 | 129 | 59 | 68 | 0 | 59 / 197 | 68 / 188 | 0 / 0 |
| d4102:n128:F | 128 | 63 | 24 | 41 | 0 | 24 / 104 | 41 / 87 | 0 / 0 |
| d4102:n128:G | 256 | 137 | 58 | 61 | 0 | 58 / 198 | 61 / 195 | 0 / 0 |
| d4102:n512:F | 512 | 262 | 109 | 141 | 0 | 109 / 403 | 141 / 371 | 0 / 0 |
| d4102:n512:G | 256 | 137 | 58 | 61 | 0 | 58 / 198 | 61 / 195 | 0 / 0 |
| source_stop | 1,172 | 919 | 52 | 59 | 142 | 194 / 978 | 201 / 971 | 0 / 0 |
| source_train | 6,466 | 4,718 | 851 | 878 | 19 | 870 / 5,596 | 897 / 5,569 | 0 / 0 |

The six base roles source_train/source_stop/O/E-reserve/C-reserve/A partition
16,415 patients. E/C are reserve subsets, G/F are A subsets, and G repeats across
tiers. Do not sum these aliases as additional people. Old O feasibility labels
were already inspected by the isolated cohort auditor. There were no old pilot
O predictions or performance results.

## Single proposed amended candidate

| Population/role | Patients | 00 | 10 | 01 | 11 | CD+ / CD− | STTC+ / STTC− | Unknown CD / STTC |
|---|---:|---:|---:|---:|---:|---|---|---|
| A | 4,388 | 2,088 | 969 | 1,038 | 293 | 1,262 / 3,126 | 1,331 / 3,057 | 0 / 0 |
| C | 128 | 60 | 25 | 36 | 7 | 32 / 96 | 43 / 85 | 0 / 0 |
| C_reserve | 878 | 418 | 194 | 208 | 58 | 252 / 626 | 266 / 612 | 0 / 0 |
| E | 128 | 56 | 29 | 37 | 6 | 35 / 93 | 43 / 85 | 0 / 0 |
| E_reserve | 878 | 418 | 194 | 208 | 58 | 252 / 626 | 266 / 612 | 0 / 0 |
| O | 2,633 | 1,253 | 581 | 623 | 176 | 757 / 1,876 | 799 / 1,834 | 0 / 0 |
| d4101:n128:F | 128 | 64 | 25 | 30 | 9 | 34 / 94 | 39 / 89 | 0 / 0 |
| d4101:n128:G | 256 | 141 | 48 | 50 | 17 | 65 / 191 | 67 / 189 | 0 / 0 |
| d4101:n512:F | 512 | 243 | 112 | 128 | 29 | 141 / 371 | 157 / 355 | 0 / 0 |
| d4101:n512:G | 256 | 141 | 48 | 50 | 17 | 65 / 191 | 67 / 189 | 0 / 0 |
| d4102:n128:F | 128 | 62 | 29 | 31 | 6 | 35 / 93 | 37 / 91 | 0 / 0 |
| d4102:n128:G | 256 | 122 | 51 | 62 | 21 | 72 / 184 | 83 / 173 | 0 / 0 |
| d4102:n512:F | 512 | 242 | 113 | 125 | 32 | 145 / 367 | 157 / 355 | 0 / 0 |
| d4102:n512:G | 256 | 122 | 51 | 62 | 21 | 72 / 184 | 83 / 173 | 0 / 0 |
| source_stop | 1,146 | 846 | 135 | 141 | 24 | 159 / 987 | 165 / 981 | 0 / 0 |
| source_train | 6,492 | 4,791 | 768 | 796 | 137 | 905 / 5,587 | 933 / 5,559 | 0 / 0 |


Every base pattern/role count differs by less than one patient from its natural
within-domain fractional quota. Maximum base-role joint-prevalence deviation is
**0.059225 percentage points**; maximum base-role marginal deviation is
**0.056003 percentage points**. A's joint-positive deviation is+0.012154pp;
O's+0.019243pp. Small random G/F/E/C samples can deviate more; their full
prevalences and supports are in the receipt. These samples were not enriched.

## Exact selector source-anchor support

Replicate the existing [analyze class masks](../../src/dmvmat_ecg/phase2/evidence.py):258
without features: at most256 source positives and256 source negatives for each
label, same patient hash sampling/geometry seed24011. B restores the other
label's observed status but retains the indexed label's own capped mask.
Consequently each label's source analysis has its **own** four-cell support:

| Population/role | Patients | 00 | 10 | 01 | 11 | CD+ / CD− | STTC+ / STTC− | Unknown CD / STTC |
|---|---:|---:|---:|---:|---:|---|---|---|
| source anchor for CD | 512 | 228 | 218 | 28 | 38 | 256 / 256 | 66 / 446 | 0 / 0 |
| source anchor for STTC | 512 | 227 | 29 | 218 | 38 | 67 / 445 | 256 / 256 | 0 / 0 |

Both labels have256/256 indexed-class support and all four cells≥10.
This is not a claim that the anchor distribution equals natural source prevalence:
class balancing is an existing geometry estimator convention, not resampling
source gradient training or target evaluation. Full patient vectors are retained.

## A and B feasibility by every registered draw/tier

A requires≥10 observed positive/negative patients per label **in both** source
analysis and target G. Both4101/4102 pass at F128 and F512, because the same G
is paired across tiers. Source-training≥200, source-stopping≥20, E≥10 and O≥50
positive/negative floors per label also pass. C has32/43 positives and96/85
negatives, meeting the20/class support floor for later calibration; no calibrator
was fitted and convergence is unknown.

B requires≥10 **in each joint cell** of source analysis and G. G4101 has
[141,48,50,17], G4102 [122,51,62,21]; both pass, for both tiers. F128's joint
counts9/6, E's6 and C's7 are **not B evidence roles**, so do not invalidate B
or justify enriching them. They remain limitations for operating-point precision.

For a new uniform256-patient G from A, expected joint counts are approximately
[121.816,56.532,60.558,17.094]. Exact hypergeometric per-cell tails give probability
bounds for **any cell below10** of [0.018716588828328648,0.018716588828328655],
about1.872%. This is a metadata support-risk calculation, not a fitted selector
abstention rate or predictive metric. The two actual fixed draws passed; no
alternative seed/draw was tried. Weak/constant learned separation, source D<0.1,
invalid bootstrap denominators or uncertainty can still make either A or B
abstain later, correctly defaulting to SHARE. Metadata cannot certify geometry.

The primary A−Full-cost contrast is feasible on these label counts even if
secondary B later abstains. Retain B and report that outcome; there is no
scientific need now for an A-only amendment or redistribution of its attempts.
