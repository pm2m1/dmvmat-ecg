# Phase3B corrected locked CPSC O evaluation V2

**Disclosure:** The first O read failed because the decoder did not recognise
`# Dx:` and returned zero observed labels, conveying no usable outcome information.
All predictions were sealed beforehand and remain unchanged. This is the corrected
second outcome read under explicit human authorization, not a new inference run.
The V1 report and blocker are unchanged and superseded by this completed V2 report.

**Pre-access revision:** Before the corrected read, the protocol owner replaced
the impossible all-R frozen-reference requirement with exact agreement on all711
frozen-reference R records plus live unmodified-adapter agreement on all3,859 R
records. Both had zero disagreements. The revision is transcribed privately in
STEP3_REVISION_AUTHORIZATION_V2.json and bound to DECODER_R_VALIDATION.json.
No split, endpoint, prediction, model, roster or statistical analysis changed.

The corrected validity gate requires at least95% observed labels and both classes
for each label. It passed before any metric calculation (see the aggregate gate receipt).

## Executive finding

Primary A−Full-cost at B128: Δ mean raw AP **-0.04702**;
95% paired record-group interval **[-0.05299, -0.04072]**.
Mean paired ΔAUROC: **-0.01552** (descriptive).
Full-cost is better than A here under the conditional comparison.
Evidence-guided selection beyond ordinary freezing requires A also to beat
Head-cost and Random-A. Both Bonferroni98.75% lower bounds exceed zero:
**False**. Beating Full-cost alone is a policy-level result.
No model, mask, checkpoint, threshold or seed was changed after O access.

## Population and pre-outcome signal audit

Frozen O membership3,860 records; evaluated **3860**;
short records excluded **0**; first10s truncation
**2436**. All exclusions were determined from the
physical WFDB header prefix before any score/outcome access. No padding or
windows. Inputs12 canonical leads,mV,100Hz; source or target-only frozen
normalization as registered. Diagnosis comments were not read during signal
metadata preparation; only the first13 unbuffered header lines were parsed.
Every MAT signal passed its publisher checksum. Full header checksums and
diagnoses were read only inside the authorized post-seal outcome accesses.
There are exactly two recorded outcome accesses: the failed V1 decode and this
single authorized corrected V2 decode. The first marker was never reset.

CD-proxy observed prevalence/random AP baseline **0.26632**;
STTC-proxy **0.24404**. An unlisted code means not annotated,
not clinically absent. Unknown labels remain masked; observed counts and
unknowns appear in every per-label metric record.

## Complete mean episode-wise performance

Source-only means three source models, reused across budgets/draws. Other
rows average nine source×draw episodes. These are means of metrics, not
probability ensembles, pooled AP or an outer-selected winning model.

|B|Procedure|Mean AP|Mean AUROC|Mean Brier|Mean log loss|Mean ECE15|
|---:|---|---:|---:|---:|---:|---:|
|64|source_only|0.54051|0.78203|0.17687|0.64609|0.13925|
|64|head_cost|0.54797|0.79343|0.16857|0.56636|0.12283|
|64|full_cost|0.61053|0.82212|0.16081|0.62131|0.13471|
|64|random_a|0.57419|0.80764|0.15925|0.53386|0.10901|
|64|selective_a|0.57091|0.80603|0.16029|0.54265|0.11103|
|64|target_only_cost|0.56222|0.73418|0.17229|0.67345|0.13842|
|128|source_only|0.54051|0.78203|0.17687|0.64609|0.13925|
|128|head_cost|0.54599|0.79195|0.16764|0.56188|0.11971|
|128|full_cost|0.65101|0.84417|0.13404|0.44908|0.07813|
|128|random_a|0.60835|0.82819|0.13935|0.44778|0.06639|
|128|selective_a|0.60399|0.82865|0.13943|0.44815|0.06580|
|128|target_only_cost|0.58000|0.74412|0.16907|0.64384|0.13677|
|256|source_only|0.54051|0.78203|0.17687|0.64609|0.13925|
|256|head_cost|0.54765|0.79384|0.16727|0.55550|0.11825|
|256|full_cost|0.67840|0.85477|0.12213|0.39185|0.04561|
|256|random_a|0.64519|0.84234|0.12806|0.40543|0.04307|
|256|selective_a|0.64299|0.84630|0.12682|0.40477|0.04310|
|256|target_only_cost|0.61627|0.77379|0.15950|0.57238|0.12302|

|B|Procedure|Label|AP|AUROC|Prevalence/AP baseline|Brier|Log loss|ECE15|Precision|Recall|Specificity|F1|Balanced accuracy|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|64|source_only|CD-proxy|0.74781|0.89173|0.26632|0.13012|0.49753|0.09496|0.68615|0.72309|0.87912|0.70315|0.80110|
|64|source_only|STTC-proxy|0.33322|0.67233|0.24404|0.22361|0.79465|0.18353|0.34007|0.26610|0.83687|0.29385|0.55149|
|64|head_cost|CD-proxy|0.74307|0.88905|0.26632|0.12621|0.44746|0.08478|0.69545|0.69909|0.88795|0.69630|0.79352|
|64|head_cost|STTC-proxy|0.35288|0.69781|0.24404|0.21093|0.68525|0.16089|0.37364|0.32720|0.82598|0.34503|0.57659|
|64|full_cost|CD-proxy|0.77658|0.89122|0.26632|0.12175|0.53339|0.10370|0.74002|0.71790|0.90576|0.72523|0.81183|
|64|full_cost|STTC-proxy|0.44447|0.75303|0.24404|0.19986|0.70923|0.16571|0.47037|0.47440|0.82572|0.46319|0.65006|
|64|random_a|CD-proxy|0.76209|0.89277|0.26632|0.12157|0.43380|0.08078|0.72430|0.67445|0.90443|0.69590|0.78944|
|64|random_a|STTC-proxy|0.38629|0.72251|0.24404|0.19692|0.63393|0.13724|0.41944|0.39266|0.82606|0.39918|0.60936|
|64|selective_a|CD-proxy|0.75417|0.88942|0.26632|0.12203|0.44410|0.08204|0.72609|0.67596|0.90494|0.69749|0.79045|
|64|selective_a|STTC-proxy|0.38765|0.72265|0.24404|0.19854|0.64119|0.14003|0.41820|0.40033|0.82107|0.40246|0.61070|
|64|target_only_cost|CD-proxy|0.70469|0.78630|0.26632|0.14269|0.59537|0.11014|0.81381|0.49125|0.95723|0.61003|0.72424|
|64|target_only_cost|STTC-proxy|0.41976|0.68206|0.24404|0.20189|0.75153|0.16671|0.49343|0.29476|0.89913|0.35639|0.59695|
|128|source_only|CD-proxy|0.74781|0.89173|0.26632|0.13012|0.49753|0.09496|0.68615|0.72309|0.87912|0.70315|0.80110|
|128|source_only|STTC-proxy|0.33322|0.67233|0.24404|0.22361|0.79465|0.18353|0.34007|0.26610|0.83687|0.29385|0.55149|
|128|head_cost|CD-proxy|0.74218|0.88835|0.26632|0.12605|0.44418|0.08160|0.69489|0.70104|0.88732|0.69696|0.79418|
|128|head_cost|STTC-proxy|0.34980|0.69556|0.24404|0.20922|0.67958|0.15781|0.36637|0.29936|0.83463|0.32662|0.56700|
|128|full_cost|CD-proxy|0.82527|0.91649|0.26632|0.10197|0.36535|0.06283|0.76812|0.72428|0.91882|0.74266|0.82155|
|128|full_cost|STTC-proxy|0.47674|0.77186|0.24404|0.16611|0.53280|0.09343|0.51741|0.42616|0.86852|0.45015|0.64734|
|128|random_a|CD-proxy|0.77940|0.90070|0.26632|0.11152|0.37677|0.05865|0.73459|0.68850|0.90631|0.70618|0.79740|
|128|random_a|STTC-proxy|0.43730|0.75569|0.24404|0.16718|0.51880|0.07414|0.48338|0.33699|0.88154|0.38232|0.60926|
|128|selective_a|CD-proxy|0.76759|0.89944|0.26632|0.11234|0.37782|0.05729|0.73042|0.68720|0.90466|0.70348|0.79593|
|128|selective_a|STTC-proxy|0.44039|0.75787|0.24404|0.16652|0.51848|0.07432|0.48422|0.35209|0.87712|0.39093|0.61461|
|128|target_only_cost|CD-proxy|0.72230|0.80090|0.26632|0.13811|0.55440|0.10708|0.81195|0.52291|0.95304|0.63172|0.73798|
|128|target_only_cost|STTC-proxy|0.43770|0.68735|0.24404|0.20003|0.73327|0.16647|0.52799|0.27707|0.91036|0.34244|0.59372|
|256|source_only|CD-proxy|0.74781|0.89173|0.26632|0.13012|0.49753|0.09496|0.68615|0.72309|0.87912|0.70315|0.80110|
|256|source_only|STTC-proxy|0.33322|0.67233|0.24404|0.22361|0.79465|0.18353|0.34007|0.26610|0.83687|0.29385|0.55149|
|256|head_cost|CD-proxy|0.74205|0.88858|0.26632|0.12495|0.43438|0.07855|0.69774|0.69585|0.88987|0.69615|0.79286|
|256|head_cost|STTC-proxy|0.35324|0.69910|0.24404|0.20959|0.67661|0.15794|0.37489|0.32142|0.82751|0.34380|0.57446|
|256|full_cost|CD-proxy|0.85342|0.92795|0.26632|0.08871|0.29536|0.02912|0.79615|0.72319|0.93271|0.75771|0.82795|
|256|full_cost|STTC-proxy|0.50338|0.78158|0.24404|0.15554|0.48834|0.06211|0.54114|0.47606|0.86939|0.50434|0.67272|
|256|random_a|CD-proxy|0.82134|0.91638|0.26632|0.09657|0.31717|0.03150|0.77267|0.68731|0.92655|0.72716|0.80693|
|256|random_a|STTC-proxy|0.46903|0.76830|0.24404|0.15955|0.49370|0.05465|0.51796|0.40823|0.87747|0.45501|0.64285|
|256|selective_a|CD-proxy|0.79937|0.91788|0.26632|0.09643|0.31805|0.02761|0.77442|0.68731|0.92722|0.72755|0.80727|
|256|selective_a|STTC-proxy|0.48661|0.77472|0.24404|0.15721|0.49150|0.05860|0.52944|0.44798|0.87122|0.48351|0.65960|
|256|target_only_cost|CD-proxy|0.74549|0.82613|0.26632|0.13387|0.50181|0.10220|0.77839|0.57015|0.93648|0.65274|0.75331|
|256|target_only_cost|STTC-proxy|0.48705|0.72144|0.24404|0.18513|0.64296|0.14383|0.54146|0.38051|0.89216|0.44366|0.63634|

All138 logical procedures, per-label values, unknown counts and confusion
counts are in PER_EPISODE_METRICS.json. No exclusions of abstaining episodes
or masks equal to controls. Precision etc use0.5; undefined denominators are
NA. Probabilities are raw, uncalibrated frozen sigmoid outputs. Only log loss
clips to[1e-7,1−1e-7]. AP/AUROC/Brier/ECE inputs are not clipped. ECE uses15
fixed equal-width bins; these are diagnostic statistics, not fitted calibrators.

## Primary paired episodes

|Source|Draw|ΔAP A−Full-cost|ΔAUROC|
|---:|---:|---:|---:|
|15101|16101|-0.03814|-0.00958|
|15101|16102|-0.07081|-0.02481|
|15101|16103|-0.03327|-0.01002|
|15102|16101|-0.06523|-0.03067|
|15102|16102|-0.05115|-0.01439|
|15102|16103|-0.06134|-0.01941|
|15103|16101|-0.02907|-0.00679|
|15103|16102|-0.03922|-0.01009|
|15103|16103|-0.03491|-0.01393|

Primary is the equal mean of all nine differences at B128. Budget64 and256
do not replace it. Practical AP margin+0.01 is an engineering criterion, not
a clinical threshold; distinguish a point estimate≥0.01 from a CI lower bound>0.01.

## Registered comparison family

|Contrast|Δ mean AP|Δ mean AUROC|95% AP interval|Bonferroni98.75% AP interval|
|---|---:|---:|---|---|
|B64/A-minus-full_cost|-0.03962|-0.01609|[-0.04509, -0.03414]|not estimable|
|B64/A-minus-head_cost|0.02294|0.01260|[0.02009, 0.02592]|[0.01936, 0.02673]|
|B64/A-minus-random_a|-0.00329|-0.00161|[-0.00468, -0.00187]|[-0.00508, -0.00152]|
|B64/A-minus-source_only|0.03039|0.02400|[0.02709, 0.03392]|[0.02621, 0.03486]|
|B64/A-minus-target_only_cost|0.00868|0.07185|[-0.00753, 0.02537]|[-0.01200, 0.02981]|
|B128/A-minus-full_cost|-0.04702|-0.01552|[-0.05299, -0.04072]|not estimable|
|B128/A-minus-head_cost|0.05800|0.03670|[0.05256, 0.06347]|[0.05125, 0.06514]|
|B128/A-minus-random_a|-0.00436|0.00046|[-0.00622, -0.00241]|[-0.00678, -0.00196]|
|B128/A-minus-source_only|0.06348|0.04662|[0.05758, 0.06955]|[0.05592, 0.07121]|
|B128/A-minus-target_only_cost|0.02399|0.08453|[0.00725, 0.04149]|[0.00238, 0.04592]|
|B256/A-minus-full_cost|-0.03541|-0.00846|[-0.04142, -0.02884]|not estimable|
|B256/A-minus-head_cost|0.09534|0.05246|[0.08679, 0.10386]|[0.08477, 0.10616]|
|B256/A-minus-random_a|-0.00220|0.00396|[-0.00643, 0.00202]|[-0.00751, 0.00301]|
|B256/A-minus-source_only|0.10247|0.06427|[0.09355, 0.11148]|[0.09113, 0.11393]|
|B256/A-minus-target_only_cost|0.02672|0.07251|[0.01001, 0.04419]|[0.00563, 0.04879]|

Only B128 A−Full-cost is primary. The four B128 secondary AP comparisons
form the registered Bonferroni family. Other budgets are exploratory; intervals
there do not establish simultaneous control across the entire result package.

Bootstrap10,000 draws,seed78701; shared record/known-duplicate-group weights
across all labels,methods,seeds,draws and budgets. Whole CD/STTC vectors are
preserved. Primary valid draws **10000**, undefined
**0**; no redraws. No primary interval when fewer
than9,500 replicates are valid. Training seeds/draws are not independent patients.
Intervals are conditional on the fitted models and realized design; they do not
capture model/selector refitting uncertainty, target selection from FEAS,
unknown within-patient correlation, source checkpoint selection or label error.

## Source transfer gap (fold9 reference versus O Source-only)

|Seed|Label|Fold9 AP|Fold9 AUROC|Fold9 AP baseline|O AP|O AUROC|O AP baseline|O−fold9 AUROC|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|15101|CD-proxy|0.84326|0.93455|0.22088|0.76189|0.90636|0.26632|-0.02819|
|15101|STTC-proxy|0.57729|0.87646|0.17102|0.31720|0.64936|0.24404|-0.22710|
|15102|CD-proxy|0.83572|0.93293|0.22088|0.74937|0.88757|0.26632|-0.04535|
|15102|STTC-proxy|0.56187|0.87770|0.17102|0.34349|0.67885|0.24404|-0.19884|
|15103|CD-proxy|0.83573|0.92428|0.22088|0.73217|0.88126|0.26632|-0.04301|
|15103|STTC-proxy|0.55902|0.87130|0.17102|0.33898|0.68878|0.24404|-0.18252|

Fold9 selected source checkpoints and is an optimistic development reference.
Source and target populations/prevalences and annotation practices differ.
AP changes cannot be interpreted as pure hospital signal-shift effects.

## Descriptive recovery contrasts fixed before O access

|Contrast|Mean ΔAP|Mean ΔAUROC|
|---|---:|---:|
|B64/head_cost-minus-source_only|0.00746|0.01140|
|B64/full_cost-minus-source_only|0.07001|0.04009|
|B128/head_cost-minus-source_only|0.00548|0.00992|
|B128/full_cost-minus-source_only|0.11049|0.06214|
|B256/head_cost-minus-source_only|0.00713|0.01181|
|B256/full_cost-minus-source_only|0.13788|0.07274|

These show how much adaptation changes ranking relative to Source-only;
they are not additional primary hypotheses or clinical recovery estimates.

## Negative transfer

|Procedure|AP worse than paired Source-only|Mean ΔAP|
|---|---:|---:|
|B64/head_cost|2/9|0.00746|
|B64/full_cost|0/9|0.07001|
|B64/random_a|0/9|0.03368|
|B64/selective_a|0/9|0.03039|
|B128/head_cost|0/9|0.00548|
|B128/full_cost|0/9|0.11049|
|B128/random_a|0/9|0.06783|
|B128/selective_a|0/9|0.06348|
|B256/head_cost|1/9|0.00713|
|B256/full_cost|0/9|0.13788|
|B256/random_a|0/9|0.10467|
|B256/selective_a|0/9|0.10247|

Counts are paired fitted episodes on the same O population, not independent
clinical events. Equality from aliases stays included. Increased ADAPT counts,
valid geometry, MMD significance and mask stability do not by themselves show
predictive benefit. Module masks/drift and label costs remain the frozen
development findings; all target arms received the same F∪G labels and200 updates.

## Execution integrity and limits

Human authorization was transcribed under private/phase3b_cpsc_v1/. Locked
study hash30a0c75ab9a159773bc87e3941ddfdee91c1860482d08e20a7b87ae601d25073.
Complete138-procedure prediction seal **b7c7e795b9641979858c5050bd6f7db9c2f1340a1f466b4828a0e4088eec5cde** preceded all O outcome
access. Aliases resolve to shared artifacts without disappearing from the roster.
Original V1 inference runs **121**; no new inference in V2.
Original summed forward time
**439.67s**, excluding preparation/load.
Predictor-fit ledger remains121/144; no training,fine-tuning,recalibration,
threshold selection,mask/seed/checkpoint changes or reserve use. Post-evaluation
tests and the final integrity receipt document re-entry refusal and unchanged
frozen artifacts. No Phase2 artifact,PTBXL fold10 or CPSC FEAS data were accessed.

This is **CPSC-only exploratory annotation-proxy evidence at record level**.
Target patient identifiers are unknown; exact native deduplication is not proof
of patient independence. PTBXL-versus-target waveform overlap was not audited
by explicit prior waiver. Phase3A FEAS outcomes chose the target and informed
the protocol; RESERVED annotations were previously used for stratification.
This history precludes a pristine independent confirmation claim. Labels are
recorded-code unions, not clinically adjudicated positives/negatives; label
definition/reporting mismatch can contribute to the apparent hospital gap.
First10s truncation can omit findings described by full-record annotations.
No external clinical validation, clinical utility, causal transfer mechanism
or future cardiovascular-event prediction is established. A CI crossing zero
is inconclusive, not proof of no effect. No further experiment is executed here.

## Final verification receipt

Post-correction full suite: **333 passed, seven existing warnings,54.44s**.
Ruff check passed; format check reports126 files already formatted.
All138 sealed prediction hashes and all295 frozen artifacts remain unchanged.
Exactly **two total O outcome accesses** are recorded: V1 failed decoding, then
one explicitly authorized V2 correction. Re-entry is rejected. No new inference,
model fitting, calibration, thresholding or source/target resplitting occurred.
The original121/144 attempt ledger and all138 logical job records match the
frozen snapshot. V1 reports/markers and the earlier blocked V2 receipt remain
unchanged; the blocked V2 narrative was archived privately before replacement.

Authoritative final receipt:
`results/phase3b_eval_v2/CORRECTED_FINAL_VERIFICATION_RECEIPT.json`.
Historical blocked `FINAL_VERIFICATION_RECEIPT.json` in this directory records
the prior stop and is superseded for current status by this corrected receipt.
The source-fold9 references are historical development scores, not new inference.
Both PNG/PDF figure pairs were generated from the completed corrected analysis
and their PNG rendering inspected. No patient-level public outputs were created.
Branch remains `phase3-cross-hospital`, HEAD
`7e93ed3a7a00b603d34d190843bcfbfc1753db1e`; no tracked-file modifications,
commits, pushes, merges or PRs.
