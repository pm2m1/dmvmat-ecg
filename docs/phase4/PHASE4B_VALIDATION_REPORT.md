# Phase 4B validation and preservation report

Completed UTC: 2026-11-21 13:03:54

Status: **PASS for completed validation-only experiment and retained-artifact verification.** The orchestration incident below is part of this record; success does not mean every invocation succeeded.

[Protocol](PHASE4B_PROTOCOL.md) · [Results](PHASE4B_RESULTS.md) · [Execution audit](PHASE4B_EXECUTION_AUDIT.json) · [Numeric results](PHASE4B_AGGREGATES.json) · [Control ranks](PHASE4B_CONTROL_RANKS.json)

## Completed computations

| Item | Verified count |
|---|---:|
| 64-label fixed-mask fits (128 masks × 9 episodes) | 1,152 |
| Other original-source baseline evaluations | 159 |
| Aligned-source target/candidate/zero-update evaluations | 66 |
| Retained target-gradient fits (200 updates each) | 1,368 |
| Retained zero-update evaluations | 9 |
| Retained A/B probability archives and matching complete reports | 1,377 each |
| Unique new 50-epoch MK-MMD source candidates | 6 |
| Selected lambda=0.1 source models | 3 |
| Permitted aligned-source checkpoints | 6 |
| Target/AdaBN checkpoints | 0 |
| Planned bitwise target/zero-update replay checks | 10 |
| Planned complete source-training replay check | 1 |
| Numerical reproducibility failures | 0 |

The B64 all-one oracle fits supply the nine B64 full-fine-tuning comparators; they were not retrained as duplicate baseline jobs. Target zero-update vectors represent three distinct source seeds per method, reused across draw cells/budgets. Aligned lambda candidates include both sets of nine B128 full adaptations; only the selected candidate enters the main aligned-method table.

## Integrity and exact-recipe checks

- All 1,377 NPZ files have only A/B probability arrays, each shape (500, 2), finite and within [0,1]. Every SHA-256 matches its completed report. Every source file/state binding matches the source actually registered for that fit. No participant IDs are exported in these public aggregate reports.
- Every target-gradient report contains exactly 200 finite recorded losses. Frozen-module parameter/buffer checks were active on every update; the prediction head remained trainable in gradient arms. The LP-FT transition was 50 head-only + 150 full updates, with head optimizer moments retained.
- All nine original-source B128 full fits reconcile exactly to retained Experiment 4 final-state hashes, drift norms and final losses. Old test predictions were not needed or opened.
- The source-data role identities match all three original source checkpoints. Original fold9 AP/AUROC values were recomputed and reconciled. All six new source checkpoint file hashes, decoded state hashes and fold9 probability hashes were verified. Checkpoints are source models only; candidate selection used the original fold9 AP checkpoint rule after all 50 epochs.
- The full source-training repeat matched state, complete history, selected epoch and fold9 probabilities. Each new target method has a bitwise A/B, state and loss-history replay. The replay checks are same-host CPU checks; cross-hardware bitwise portability is not claimed.
- All nine B128 original-source selector/random pairs replayed exactly from permitted source/G inputs. B64 controls were reconstructed with unchanged production numerics (999 permutations/4,999 geometry resamples), not obtained from original B64 receipts. Those receipts are unavailable within the allowed bundle. No A/B data selected the reconstructed masks.

## Data isolation

| Role | Records | CD positives | STTC positives |
|---|---:|---:|---:|
| A (selection) | 500 | 126 | 135 |
| B (evaluation) | 500 | 127 | 135 |

- CPSC adaptation pool: 3,859. All-budget labeled-draw union: 711. Eligible validation-draw population after excluding that union: 3,148. A/B mutual overlap and overlap with every original label draw: zero.
- U: 2,859 adaptation signals, excluding all A/B records. U labels were never requested by AdaBN or alignment; the label getter rejects that role. Integrity hashing of the immutable bundle is not supervision. F/G labels are used only for the prespecified supervised fits/selector.
- The bundle runtime guard checked every loaded CPSC ID against adaptation membership and test-ID exclusion hashes. CPSC test-half signals, labels, predictions, metrics, derived evaluation files and seals were never opened. Exclusion hashes were used only to reject out-of-scope IDs.
- Source inputs: PTB-XL folds1–8 train (17,084), fold9 validation (2,146), with patient separation. A byte-level source metadata projector checks the fold tag first and skips out-of-scope rows before decoding record IDs/annotations. No PTB fold10 identifier, label or waveform payload was decoded/loaded. Only the explicitly projected source-train/source-stop cache metadata were used.
- Runtime OS-open guards restrict fitting to the isolated adaptation/source bundles and new outputs. No CPSC test provider, old evaluation executor, ledger or reserved/sealed payload was invoked. Source/G control replay uses the same boundary.
- All per-record inputs/caches/predictions and six source checkpoints remain under ignored private Phase4B directories; no target checkpoint, network upload, commit, push or merge occurred.

## Runtime

| Recorded stage | Wall time |
|---|---|
| 64-label oracle including its gate | 2 h 55 min 42.58 s |
| Successful baseline invocation including replay gates | 0 h 32 min 26.09 s |
| Successful source candidate-grid invocation (initial source repeat excluded) | 2 h 41 min 48.46 s |
| Primary analysis, 10,000 common record resamples | 0 h 02 min 41.72 s |

Retained target/zero-update evaluations sum to 37713.66 worker wall-seconds; their planned replays sum to 182.06. Retained source fits sum to 29948.94 worker wall-seconds; the verified extra source repeat took 4496.73 seconds. These are concurrent worker elapsed durations, **not total elapsed study time or CPU time**. Stage durations overlap. Source/G evidence extraction and redundant/interrupted computations are not included in these per-fit sums; a complete worker-time total for the incident was not retained.


Filesystem-derived elapsed durations ([timing metadata](PHASE4B_WALL_TIMING.json)): aligned-target stage **11.56 minutes**; oracle contract to primary result **4.365 hours**. These are explicitly timestamp-derived elapsed windows, include overlap/incident delay, and exclude coding before the oracle contract and subsequent report QA.

## Orchestration incident and correction

An earlier asynchronous controller was still active when orchestration was restarted. This duplicated baseline and source-grid computation. Exclusive file creation rejected a duplicate baseline invocation at the AdaBN prediction destination (exit 1), preserving the successful original invocation, which completed all 159 evaluations and six replay gates. Process inspection confirmed two source-grid parents. Only the later parent/workers were terminated (exit 143); the earlier grid completed all six candidates and its completion manifest. No prior result/checkpoint was removed, reset, overwritten or selected according to its score.

Beyond the intended counts above, three full redundant source-candidate training computations completed, two redundant source computations were interrupted, and at least two redundant zero-update evaluations occurred in the refused baseline invocation. Their discarded model states were not retained or compared; they are not additional verified replays and are not treated as additional experimental replicates. Terminating the duplicate also emitted a multiprocessing resource-tracker warning about five semaphore objects; every retained checkpoint/prediction subsequently passed integrity verification. The report does not claim zero execution failures or conceal this extra computation.

A per-stage OS advisory mutex now refuses a competing controller before expensive stage work. Three new synthetic interprocess/exception/symlink regressions pass. Numerical fitting code, seeds, data, masks, update counts, losses, grids and selection rules were unchanged. Original V1/V2 identity records remain intact; explicit V3 records only the new orchestration wrapper/lock. All subsequent stages completed under this wrapper. This is an execution-only correction, not an experimental-policy change.

## Statistical result and limits

- Only MK-MMD lambda=0.1 + full adaptation meets the prespecified B128 pointwise AP rule: delta +0.021584, 95% interval [+0.004333,+0.038866]. Its seven-method Bonferroni sensitivity interval [-0.002344,+0.046415] includes zero. Macro AUROC delta +0.007316 has interval [-0.001327,+0.016053]; AUROC superiority is not established.
- The global A-selected B64 mask does not establish improvement; the per-episode A oracle also has an interval including zero. One B-ranked nonfull mask has a slightly higher descriptive mean; it is explicitly post-hoc, not a validated winner. Neither result proves universal impossibility of useful freezing.
- The same 10,000 B-record weights are used across models, episodes and labels; supported draws are 10,000/10,000 for reported comparisons. Intervals condition on fitted models and A-only choices; they do not include A-selection, source-training or unknown patient-cluster uncertainty. Module effects/control-rank analyses are secondary and exploratory.
- B was already examined in Phase4A. These are reused validation findings, not a new untouched test result. Optimization budgets exclude 500 A selection labels and 500 B evaluation labels. Aligned zero-shot has zero target-gradient labels but its lambda selection uses B128 adaptations/A; that selection is not label-free and also conditions its B64/B256 arms.
- Lambda=0.1 preserves source mean fold9 AUROC (0.905576 versus original 0.902868); lambda=1.0 (0.788438) fails the prespecified screen. Both-domain BN updates and added U exposure accompany alignment, so an MMD-specific causal claim requires a matched future ablation. No such extra experiment or test evaluation was run.

## Commands and actual checks

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest tests/phase4 tests/phase4b -q` | 52 passed, 0 failed; two WFDB/Pandas deprecations |
| `.venv/bin/ruff check src/dmvmat_ecg/phase4b tests/phase4b` | PASS |
| `.venv/bin/ruff format --check src/dmvmat_ecg/phase4b tests/phase4b` | PASS; 21 files formatted |
| `.venv/bin/python -m dmvmat_ecg.phase4b.verify_outputs` | PASS; all retained hashes/arrays/source states checked |
| `.venv/bin/python -m dmvmat_ecg.phase4b.controls_report` | Completed once; all B64 mask-score means re-reconciled to primary results |
| Three new SVGs parsed/rendered at 1,500 px and visually inspected | PASS; readable, no clipped text or missing axes |
| Relative report links / table structure / SVG XML | 14 links valid; 0 broken; 27 method-budget rows; 128 masks; 3 SVGs valid |
| `git diff --exit-code HEAD` / staged diff | Empty; original tracked files unchanged, nothing staged |

The plotted selector/random rank markers coincide for several episodes because the corresponding masks/ranks coincide; exact values are in the table. Figure renderings used only the new aggregate SVGs and were saved in ignored private visual-QA storage. No extra dependency installation or historical environment change was needed.

## Preservation and final branch

Local branch: `phase4b-adaptation`. HEAD remains `48b18b5447b672d8f5d4212d8045b13c3db84865`, the preexisting Phase4A parent. Earlier experiments, original source weights/normalization, Phase4A split/bundle/results, configs, ledgers and reports were not modified. All final tracked-file and staged diffs are empty; additions are only new Phase4B code/tests/docs/figures and ignored new input/run storage. No commit, push, merge, CPSC test evaluation or new oracle at 128/256 labels was performed.

The count/error distinction and exact method counts are retained in [the machine-readable execution audit](PHASE4B_EXECUTION_AUDIT.json). All original scientific outputs remain intact.
