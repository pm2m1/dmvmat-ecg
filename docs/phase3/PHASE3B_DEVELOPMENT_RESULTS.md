# Phase3B development results — O NOT EVALUATED

Adopted v1; source code `d4f0dd6361cf115d79c8384905682d76a46ee2f9fb79446680dfadafc7cc7ebb`; real implementation `92654c006f94859b896b0c977c8e3b32c9a6d7490f342eb05ccc07e0eba9943c`.
Commit `7e93ed3a7a00b603d34d190843bcfbfc1753db1e`. All source validation numbers below are fold9
development scores for the restricted annotation proxies, not final test results.

|Source seed|Selected epoch|CD AP|CD AUROC|STTC AP|STTC AUROC|Elapsed fit seconds*|
|---:|---:|---:|---:|---:|---:|---:|
|15101|22|0.84326|0.93455|0.57729|0.87646|17264.7|
|15102|21|0.83572|0.93293|0.56187|0.87770|17178.2|
|15103|24|0.83573|0.92428|0.55902|0.87130|1877.9|

*The first two source fits were suspended and resumed in the same in-memory processes. Their elapsed seconds include this pause; uninterrupted active training time was not recorded. See IN_MEMORY_RESUME_RECEIPT.json. No fit was restarted or optimizer/RNG state reconstructed from disk.

Source train17,084 recordings/14,823 known patients; source selection2,146/1,917.
R3,859 records; O3,860 records (no O labels/signals loaded). The new adaptation
union comprises711 records/1,422 observed annotation entries. Target patient
linkage is unknown. Source pretraining labels/cost are separate from target budgets.

|Episode|Role|Records|CD positive annotations|STTC positive annotations|
|---|---|---:|---:|---:|
|16101/B128|F|64|16|15|
|16101/B128|G|64|15|17|
|16101/B256|F|128|36|25|
|16101/B256|G|128|34|33|
|16101/B64|F|32|7|9|
|16101/B64|G|32|8|10|
|16102/B128|F|64|17|14|
|16102/B128|G|64|23|11|
|16102/B256|F|128|26|34|
|16102/B256|G|128|41|31|
|16102/B64|F|32|7|10|
|16102/B64|G|32|13|5|
|16103/B128|F|64|23|13|
|16103/B128|G|64|10|22|
|16103/B256|F|128|50|27|
|16103/B256|G|128|27|42|
|16103/B64|F|32|11|5|
|16103/B64|G|32|5|11|

All target arms fit F∪G for200 updates. There is no E/C/P, target AP/AUROC or
calibration fitting. TARGET_TRAINING_DIAGNOSTICS.json reports last sampled
training-batch BCE and postfit in-sample F∪G BCE (eval-mode BN/dropout), solely
training diagnostics; neither selects a checkpoint or measures generalization.

Mask order: view1..5,fusion1,fusion2; A=ADAPT,S=SHARE. Invalid label geometry
abstains; all-SHARE is Head-cost and is retained in the eventual comparison.

|Source|Draw|B|Mask|
|---:|---:|---:|---|
|15101|16101|64|S S S A A A A|
|15101|16101|128|S S S A A A A|
|15101|16101|256|A S S A A A A|
|15101|16102|64|S S S S S S S|
|15101|16102|128|S S A S A A A|
|15101|16102|256|A S A A A A A|
|15101|16103|64|S S S S A S S|
|15101|16103|128|S S A S A A A|
|15101|16103|256|S S A A A A A|
|15102|16101|64|S S S S S A A|
|15102|16101|128|S S A S S A A|
|15102|16101|256|S S A S A A A|
|15102|16102|64|S S S S S S S|
|15102|16102|128|A S S S A A A|
|15102|16102|256|A S A A A A A|
|15102|16103|64|S S S S A A A|
|15102|16103|128|S S A S A A A|
|15102|16103|256|A S A S A A A|
|15103|16101|64|S S A A A A A|
|15103|16101|128|S S A A A A A|
|15103|16101|256|A S A A A A A|
|15103|16102|64|S S S S S S S|
|15103|16102|128|S S A A A A A|
|15103|16102|256|A S A A A A A|
|15103|16103|64|S S A S A A S|
|15103|16103|128|S S A S A A A|
|15103|16103|256|A S A A A A A|

Actual attempts **121/144**; target artifact aliases **17/135**.
Budget mask/abstention summaries, all135 procedures, random candidate counts,
equality probabilities, parameter drift, losses, runtimes and artifacts are in
the aggregate JSONs in results/phase3b/. Alias rows reference their original
fit runtime and incur zero additional gradient-fitting cost. No alias savings
funded extra work. Exact SHARE parameters and BN buffers were checked unchanged.

This is a development-selected cross-dataset annotation-proxy study, not
independent clinical validation. No evidence of selective superiority is
available without separately authorized locked evaluation. Prior FEAS outcomes
informed the design; public annotations and historical stratification exposure
are disclosed in the adopted protocol. Source-only's task-aligned transfer gap
is not yet known. Absence of a code is annotation absence, not clinical absence.
