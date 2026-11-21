# Phase 4A validation and preservation report

## Actual checks

- `python -m pytest -q tests/phase4 -p no:cacheprovider`: **33 passed, zero failed**. Two WFDB/Pandas deprecation warnings from imported dependency tables; no failed numerical check.
- `python -m ruff check src/dmvmat_ecg/phase4 tests/phase4`: PASS.
- `python -m ruff format --check src/dmvmat_ecg/phase4 tests/phase4`: checked at final handoff.
- Original source-code identity: `source_code_hash() == BASE_HASH` is true. Its fixed source/configuration list excludes the new Phase 4 directory; no old numerical behavior or frozen guard changed.
- Final integrity pass validated all **1,152 fit reports and prediction-file hashes** using only the bundle/new output directory. All **36 original method-reference comparisons**, spanning **31 distinct matching mask/episode fits**, match state hash, parameter drift and final loss exactly.
- Repeated full-mask fit: bitwise-identical A/B probabilities, final states and 200-entry loss histories. Three seeded random parallel fits were repeated serially: all prediction/state/history/drift comparisons passed.
- Original full target-checkpoint loss-history payloads were not opened; original-reference checks used allowlisted completion state/drift/final-loss fields. Repeat-to-repeat training histories were compared in full. No original test evaluation artifact was used as a reference.
- New numerical runtime/study/safety file hashes match the code identities captured before the big run. A post-run builder-only path-traversal hardening and invented traversal test were added; this changed no fitted numerical code, original input, existing bundle or result. The builder was not rerun.
- All 711 original draw records match original development-only waveform caches and verified V2 outcomes; all 3,859 pool records decode with observed two-label targets. Zero test IDs in included record inventories; only exclusion hashes retained.
- Split records: A=500, B=500, disjoint from each other and all 711 original labeled records. SHA-256 values are in VALIDATION_SPLIT_SUMMARY.json; actual IDs remain gitignored.
- Tie-aware weighted bootstrap arithmetic was tested against literal duplicated-record sklearn AP, including equal scores, zero-weight prefixes and absent-class replicates. All 10,000 actual replicates are supported; same record-count resamples used for every arm/mask/episode/label.
- Factorial effect coefficients match the seven-bit-plus-episode-fixed-effects OLS regression exactly within floating-point tolerance. Record-bootstrap intervals preserve shared-participant dependence.
- Both SVGs were rendered from their saved vector files into private QA PNGs and visually inspected: labels, axes, intervals and policy/mask markers are legible. Full and A-selected lines intentionally coincide.

## Results and interpretation

All 128 fixed masks were evaluated in all nine episodes. Full fine-tuning is rank 1 on A and B; head-only is rank 128 on A and B. The A-selected mask is full itself, so the primary difference and CI are [0,0] by identity; this is not an equivalence test or proof of universal nonexistence. No non-full fixed mask exceeded full's observed mean B macro AP. The separate per-episode A-selected oracle also underperformed full on B. The report preserves the requested rule (b) label with these limitations and explicitly counts additional validation labels.

## Access boundary

Only the builder read the permitted original source/model/membership/evidence/development fields. Mixed JSON metadata were projected, with other subtrees structurally skipped instead of materialized. Only adaptation IDs can resolve to raw header/signal reads. The old V2 `validate_r`/integrity functions were deliberately NOT called because they would verify held-out seals/predictions. The pure V2 decoder was reused instead. Runtime splitting/training/analysis use an OS-open audit guard and per-record exclusion guard; no direct read of the original private experiment, old result/evaluation directories or raw dataset is permitted.

No test signal, label, prediction, metric or derived evaluation file was read. No PTB-XL fold 10 or sealed/reserved payload was read. Test role IDs were used only to reject excluded records. No target checkpoint was loaded or saved. The three source checkpoint copies retain original bytes, and no source training occurred.

## Git and preservation

Branch: `phase4-oracle-masks`. Original/main commit: `c93939a54f62399b781fd5cb353f149d27fcfb52`. Original tracked files and staged index are unchanged. Only new docs/phase4, src/dmvmat_ecg/phase4, tests/phase4 and figures/phase4 files are visible as untracked additions. Bundle, record lists, arrays, progress logs, reports and source-copy artifacts under private/phase4_oracle_* are ignored by the original .gitignore. The new run root is owner-only. No commit, push, merge, PR or network operation occurred.
