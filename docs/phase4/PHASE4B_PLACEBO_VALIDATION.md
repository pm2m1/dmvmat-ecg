# Placebo-alignment amendment — validation verification

Branch `phase4c-confirm`; no commit, push or merge. This amendment completed only the validation experiment. It did not run a Phase4C confirmatory replay or access CPSC test material, PTB-XL fold10, old test predictions/results or reserved/sealed inputs.

## Actual computation

- Three source/source placebo models: seeds 15101/15102/15103, lambda0.1, 50 epochs each, identical source architecture, folds, optimizer/schedule and seven-module/four-kernel MMD recipe to Phase4B. The second stream was PTB training data rather than CPSC. Source training's filesystem guard allowed only the PTB-only source bundle and this new run's storage.
- One additional complete 50-epoch source repeat passed exact state/history/selected-epoch/fold9-prediction authentication. The repeat saved no extra checkpoint.
- Exactly 27 distinct target fits: nine episodes × budgets64/128/256, 200 updates each. One additional 200-update repeat passed **byte-for-byte** A/B prediction equality, state, normalization, source, seed, history and drift checks. No target checkpoint was saved.
- All 27 A/B probability archives, three source checkpoints, source states/histories/fold9 predictions and future replay reference hashes passed independent verification. Private arrays and checkpoints are ignored by Git. Original source normalization matched exactly before target fitting.
- All six fixed paired contrasts used10,000 common split-B record resamples. Every reported macro/per-label AP/AUROC interval had10,000 valid draws and zero unsupported draws. A/B are reused validation data, not test data; intervals are pointwise and conditional on the fixed fitted models.

Detailed metrics and intervals: [new aggregates](PHASE4B_PLACEBO_AGGREGATES.json). Checkpoint SHA-256 values and runtime totals: [verification record](PHASE4B_PLACEBO_VALIDATION.json). Source epoch-selection and target metrics: [appended report](PHASE4B_RESULTS.md#placebo-alignment-control).

## Public-code verification

Executed from the project root:

```sh
.venv/bin/python -m pytest tests/phase4 tests/phase4b tests/phase4c_placebo -q
.venv/bin/ruff check src/dmvmat_ecg/phase4c_placebo tests/phase4c_placebo
.venv/bin/ruff format --check src/dmvmat_ecg/phase4c_placebo tests/phase4c_placebo
.venv/bin/python -m dmvmat_ecg.phase4c_placebo.verify
```

Results: **88 tests passed, zero failed**; two upstream WFDB/Pandas deprecation warnings. Ruff lint and format checks passed. The real-run integrity verification returned PASS.

Invented fixtures establish numerical parity with both the frozen 50-epoch alignment loop and 200-update full-FT loop. Tests cover source-only file/role restrictions, independent stream sampling, normalization, wrong model/episode/source bindings, altered hashes, malformed A/B arrays, and strict replay rejection. The byte-level replay test rejects even signed-zero differences that ordinary numeric array equality would accept. Synthetic tests do not establish clinical performance.

## Preservation and remaining gate

The pre-amendment Phase4B report is preserved as an identical byte prefix; only a new `Placebo-alignment control` section was appended. Existing Phase4A/B numerical code, configurations and results were not changed. The primary Phase4C contrast remains M3−M2 at128 labels; only the specified M4 substitution and deterministic-replay authentication amendment were made. The 95-secondary Holm family remains fixed. AdaBN ranking-result equality is distinguished from universal bitwise state/probability equality.

The author must commit the amended [confirmatory protocol](PHASE4C_CONFIRMATORY_PROTOCOL.md) and then explicitly say `protocol committed, continue`. All 81 future target replays must authenticate both A/B arrays bitwise and all recorded state/recipe bindings **before any test signal or prediction**. A mismatch stops the operation. This amendment has not started that future operation.
