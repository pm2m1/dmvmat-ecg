# Internal pilot integrity report — Phase 2E

**Development completed; evaluation remains unauthorized.**

## Identity and authority

```text
canonical_path = /Users/kiyo/Desktop/dmvmat-ecg/private/phase2d_joint44301_v1/study.sqlite
code_hash = ed812868495e91aac456f955315635932031a67e250c798fad5127a831bc3afa
contract_hash = dbc0afe9bee29e7323a3a353b37660b8dff20733e3fe62fa94958d95400ecb28
dependency_hash = 195cd5ae983b6a3e196bc52ac0cf0997b26f60f45fb801e53256b8b57f881424
ledger_id = d6f784e8-5073-4fad-a804-1011820f92ef
manifest_hash = 65cf652084396403077918e0e9be560eaedd009ab4fc8587260c833ddcf4f816
namespace = real:phase2d-joint44301-v1
lock_hash = 1f31e3835fa3ee270d4866313af91107ac2d654b3832469a369bd08326a8d609
source_revision = c7b795553ee9806caeb1835e7a829dc0469872eb
```

Adoption semantic hash: `1eb08c623d74456f3390c1c5426d6f450ad1e361608f7893118e3a79d2474c7d`. Development receipt semantic hash: `301af02185d259570ab28920bdff9db2ffb3a57cb21b1fd40d90bd2b1745d1cd`. Both were compared with the existing private human-supplied records; neither was modified or inferred. No evaluation approval was created.

## Actual checks

- Trusted full inventory remains69 paths and matches the authorized code hash; dependency versions and contract hash unchanged. No inventoried file edited.
- Seed44301 worker hash matches the adopted candidate and prior disjointness/support receipts. Non-O source/target groups and each active fit/evidence/stopping/C role are disjoint. Patient/record IDs, canonical policy and label ordering unchanged.
- Publisher SHA256SUMS file matched its pinned release hash; all non-O native hashes matched publisher entries; consumed WFDB/header bytes, dimensions,100Hz, mV and lead ordering were verified by the frozen role loader. No global label cache, fold10 data or O vault was reopened for model development.
- O isolation was checked through unchanged signed audit/role hashes and denied evaluation authority; O identities/outcomes were not reopened to repeat the earlier cohort audit.
- Exactly33 atomic reservations,33 successful outcomes,zero failed/interrupted/retried/active fits,zero technical reserves used; hard ceiling36. Fresh source initialization and target-only resets recorded; no exact mid-training continuation claimed.
- All33 job specifications/seeds, checkpoint AP/BCE/earlier-tie selection, scratch epoch-zero ineligibility and pretrained epoch-zero eligibility reproduced from histories.
- All111 checkpoint/lineage/evidence/calibration artifacts validated by SHA-256. Source checkpoint remained `f406a8d01c39f6a0c81367239af610a60561bed9416bef5a368f661084bae68b`; source preprocessing hash `4651a624c594642636bc53c5e46024da3ce8eb2500897b59a52d68a5158589dd`.
- All pretrained initial states equal the newly trained source, all SHARE owner drift exactly0;1560 frozen parameter/BatchNorm-buffer tensor comparisons equal source. Every retained epoch-zero checkpoint equals source in all state tensors.
- Random-A parameter and module counts exactly match A in every episode; same-mask status recorded without resampling.
- Every data-access event checked against its registered operation and role; no O event. All label unions/cells match384/768 or640/1024, and source C128.
- All33 C artifacts present;66 converged mappings. No recalibration or model reselection after development feedback.
- Historical classification ledger remains evaluated,11 fits,one historical evaluation access. Only its administrative counters were read; no historical signal/prediction/checkpoint was used. Tracked historical files remain unchanged against commitc7b7955.
- Lock completed using the existing ledger API only after all checks and tests; phase now locked,33/36 attempts,zero new evaluation accesses, evaluation grant absent.

## Tests and execution evidence

`.venv/bin/python -B -m pytest -q -p no:cacheprovider`: **206 passed,7 existing warnings**,38.38s. Ruff and formatting checks passed for the repository and all three private execution/reporting/audit helpers. Tests used fictional fixtures and consumed no real attempts. Read-only completed-artifact audit passed.

Platform: Apple M5 Pro,48GiB; CPU float32,4 threads/fit,maximum2 fit workers. AC power100% throughout sampled checks,709GiB initially available. Environment versions:

```json
{
  "device": "cpu",
  "dtype": "float32",
  "max_workers": 2,
  "python": "3.12.14",
  "threads_per_fit": 4,
  "uv_lock_sha256": "a69f264030cf30ee4392cabc05e72a2184a9441ff5fd50e92092015f65b4bed3",
  "versions": {
    "httpx": "0.28.1",
    "matplotlib": "3.11.2",
    "numpy": "2.5.4",
    "pandas": "3.1.1",
    "pytest": "9.2.0",
    "ruff": "0.17.4",
    "scikit-learn": "1.9.1",
    "scipy": "1.18.1",
    "torch": "2.15.0",
    "wfdb": "4.3.1"
  }
}
```

Full receipts: [pre-execution](../../results/phase2e/PRE_EXECUTION_RECEIPT.json), [completed audit](../../results/phase2e/COMPLETED_DEVELOPMENT_AUDIT.json), [configuration/artifact index](../../results/phase2e/CONFIGURATION_AND_ARTIFACT_INDEX.json), [development lock](../../results/phase2e/DEVELOPMENT_LOCK_RECEIPT.json), [frozen code inventory](../../results/phase2e/FROZEN_CODE_INVENTORY.json). Private launcher hashes are additionally bound in per-attempt execution-context events and the lock receipt; these helpers invoked unchanged core fitting code and added reporting only.

## Files and repository state

New Git-visible outputs only: these three `docs/phase2/INTERNAL_PILOT_*.md` reports; `results/phase2e/` preflight/support/reference/audit/analysis/configuration/lock/inventory JSONs;33 per-attempt aggregate JSONs;three PNG/PDF figure pairs. No participant rows, row predictions or weights in those outputs.

New private outputs: three versioned execution/reporting/audit helpers and four worker/reference logs under `private/phase2e_run_v1/`, plus unsigned evaluation request. Official API writes added33 attempts,78 predictor/evidence artifacts and33 calibration files under the existing pilot namespace; source/target lineages include fresh preprocessing and effective drift. The existing pilot ledger legitimately advanced development→locked and recorded only authorized execution/calibration/lock activity. Existing approvals and old manifests were preserved.

Branch `main`, HEAD `c7b795553ee9806caeb1835e7a829dc0469872eb`. No tracked changes, commits, pushes, merges or PRs in Phase2E. Only the new reports/results are untracked. The prior repositories and original classification conclusions remain unchanged.

No scientific protocol deviation or unresolved integrity blocker was found. Descriptive metric limitations are recorded in the results report; they are not hidden by the passing engineering checks.
