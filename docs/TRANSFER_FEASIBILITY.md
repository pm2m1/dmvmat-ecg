# Future cross-database transfer — feasibility only

**INDEPENDENT_CROSS_COHORT_TRANSFER_NOT_EXECUTED.** No Challenge corpus downloaded
or transfer model trained. Current ECG classification is an internal PTB-XL
benchmark, not prospective risk prediction or independent external validation.

## Verified overlap risk and licensing

The official [CinC2021 release1.0.2](https://physionet.org/content/challenge-2021/1.0.2/)
contains PTB-XL and PTB alongside CPSC, CPSC-extra, INCART, Georgia, Chapman and
Ningbo sources. Its PTB-XL component has21,837 recordings, predating the1.0.3
deduplication/label revisions used here. Thus **PTB-XL→entireChallenge2021 is not
independent transfer**. PTB itself is separately represented and cannot simply
be assumed unrelated to other clinical sources. Challenge terms/license are
CC-BY4.0; verify each original component's terms and citation obligations before
future acquisition/use. No claim that availability guarantees person linkage.

Source identity/record counts from the release: CPSC6,877, CPSC-extra3,453,
INCART74 (segments from Holter recordings), PTB516, PTB-XL21,837, Georgia10,344,
Chapman10,247, Ningbo34,905. These are recordings, not verified unique people;
segment and repeated-record structure needs source-specific grouping.

## Endpoint crosswalk is not established

The Challenge uses SNOMED labels and a different scored diagnostic task. Its
[official scored mapping](https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_scored.csv)
and [unscored mapping](https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_unscored.csv)
were inspected for feasibility only. Normal sinus rhythm426783006 is **not**
equivalent to an ECG with no diagnostic abnormality (NORM). A superclass absence
is not automatically a confirmed negative across heterogeneous coding systems.
CD/STTC require a medically reviewed code-level crosswalk; MI/HYP labels in
unscored mappings do not make all datasets comparable.

Illustrative mapping-file recording counts for code164865005(MI): CPSC0,
CPSC-extra376, INCART9, PTB368, PTB-XL5,261, Georgia7, Chapman40, Ningbo83.
For164873001(LVH):0,158,10,0,2,359,1,232,15,632 respectively. These are published
code counts, **not our audited eligible positives**, not harmonized HYP/MI
superclasses, not independent-patient counts and not guaranteed adaptation
support after exclusions. Broader old/acute infarction codes differ too.
The mutable mapping sources were read during this audit; a future protocol
must pin their exact revisions and re-audit rather than treat them as frozen data.

## Required future overlap-removal and role design

1. Exclude the entire PTB-XL Challenge component for external targets, not just
   filename matches; map retained1.0.3 IDs to historical releases where needed.
2. Fingerprint native waveform bytes, calibration-corrected physical signals,
   resampled canonical lead signals, and offsets/segments. Exact matching alone
   misses changed sample rate, amplitude scaling and cropped duplicate records.
3. Perform conservative near-duplicate correlation/segment searches using a
   prespecified threshold determined independently of outcome performance;
   audit manually uncertain pairs or exclude their connected components.
4. Recover true patient identifiers/repeated-Holter segment groups within each
   source where available. Cross-database unknown identity remains unresolved;
   waveform deduplication cannot prove no shared patient with different ECGs.
5. Harmonize five outcome definitions, annotation certainty and missing-label
   semantics with ECG expertise, document observation time and adjudication.
6. Count actual eligible positives/negatives after overlap removal. Separate
   source training, target fitting, selector calibration, checkpoint stopping
   and final target evaluation at patient/duplicate-group level.
7. Prespecify target-label costs for every role; compare source-only, target-only,
   head-only, full, parameter-matched random and selective adaptation with
   epoch0-eligible checkpoints and paired source initialization.

## Decision

Cross-database transfer is a plausible research direction but **independent
cross-cohort validation is currently BLOCKED** by unresolved person separation,
waveform/segment overlap and outcome harmonization. A future source pair may
be suitable after the above gate; this report does not authorize its execution.
The present model exposes seven representations and synthetic selector tests
only. A defensible next contribution is a registered, overlap-audited external
ECG adaptation study, not relabeling internal folds as independent cohorts.
