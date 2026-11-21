# Phase 4A: completed validation-only oracle study

Read [protocol](ORACLE_MASK_PROTOCOL.md), [results](ORACLE_MASK_RESULTS.md), [aggregates](ORACLE_MASK_AGGREGATES.json), [validation](VALIDATION_REPORT.md), [split hashes](VALIDATION_SPLIT_SUMMARY.json) and [bundle summary](BUNDLE_MANIFEST_SUMMARY.json).

The completed run uses 128 masks × nine original 128-label episodes. No new source training, test evaluation, 64-/256-label fitting, clinical threshold selection or checkpoint saving is part of this study. A-selected full fine-tuning ranks first on both A and B. See the report for the degenerate primary identity interval and the non-universal interpretation.

## Local commands

Use the existing project environment from the repository root:

```sh
.venv/bin/python -m pytest -q tests/phase4 -p no:cacheprovider
.venv/bin/python -m ruff check src/dmvmat_ecg/phase4 tests/phase4
```

Original execution order (already completed; do not recreate old destinations):

```sh
.venv/bin/python -m dmvmat_ecg.phase4.build_bundle
.venv/bin/python -m dmvmat_ecg.phase4.runtime --reproduce
.venv/bin/python -m dmvmat_ecg.phase4.study
.venv/bin/python -m dmvmat_ecg.phase4.analysis
```

The builder refuses an existing bundle. The reproducibility gate and analysis refuse to overwrite completed artifacts. The study resumes only exact compatible completed fits and skips them; it refuses incompatible or orphaned partial artifacts. Never delete old files to force success. Source checkpoint copies are byte-identical copies only, not retrained model saves. All Phase 4 numerical code/data bindings are recorded in the new run identity.

## Locations and contents

- `private/phase4_oracle_bundle/MANIFEST.json`: hashed adaptation-only source bundle; three source checkpoint copies/normalization, original episode memberships/seeds, recorded masks/evidence, development-reference projections, 3,859 allowed signals/labels and hashed test exclusions. No test record ID is stored in its record lists.
- `private/phase4_oracle_run/split.json`: private A/B IDs and fixed seeds.
- `private/phase4_oracle_run/predictions/`: 1,152 small A/B-only probability arrays; no IDs or weights.
- `private/phase4_oracle_run/fits/`: mask/episode/configuration identity, aggregate training loss/state/drift/runtime and A/B AP. No model state tensor saved.
- `private/phase4_oracle_run/progress.jsonl`: append-only progress; completion and reproducibility summaries are in the same new run directory.
- `figures/phase4/`: only the new aggregate distribution and module-effect SVGs.

The original .gitignore already excludes both new private paths. No old config, result, ledger, report, source or ignore file was modified; nothing is committed. This is an oracle diagnostic using 500 additional A labels for selection and 500 B labels for evaluation, not a 128-total-label deployment algorithm.
