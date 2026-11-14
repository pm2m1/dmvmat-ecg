# Phase3B development integrity

- Authorized commit: `7e93ed3a7a00b603d34d190843bcfbfc1753db1e`; exact requested message, no trailers.
- Step0 git add -A also versioned pre-existing public Phase2 aggregate outputs;
  their contents were not changed. Protected Phase2 ledgers/approvals/checkpoints/O
  were never opened or modified. No commits followed Step0; no push/merge/PR.
- Frozen base code: `d4f0dd6361cf115d79c8384905682d76a46ee2f9fb79446680dfadafc7cc7ebb`; real implementation: `92654c006f94859b896b0c977c8e3b32c9a6d7490f342eb05ccc07e0eba9943c`.
- Mapping and original CPSC parent manifest match the adopted hashes.
- New memberships hash: `ec2db8d89cca158320e5842550be87914860f82ddd00f598c095a41452df6c5e`; O record count only3,860.
- Source train/selection known patients disjoint; target known-record groups
  disjoint across prohibited roles; FEAS excluded. Patient-level target separation
  and cross-source person independence remain unverified.
- Source-only normalization fitted afresh on source train. All source models
  initialized from fresh random weights. Target-only fits its own F∪G statistics.
- All3 source fits and135 target logical procedures completed. Attempts
  121/144; retries0; no in-flight attempts.
- Complete alias/attempt table and all checkpoint/normalization hashes are frozen
  privately. Checkpoints verified; SHARE parameters/buffers bitwise unchanged.
- Training preflight:291 tests passed,7 existing warnings; Ruff and formatting
  passed. Post-development verification results are recorded in a separate receipt.
- CPU float32,4 threads per fit,at most2 workers,caffeinate. Shell-detached cache
  launches did not persist; retained logged sessions were used. Those were not
  predictor fits and consumed no attempts. No exact optimizer/RNG resume claimed.
- Development target448/711 recordings used deterministic first10s truncation,
  identical for every arm; annotations may refer to later signal. No short-record
  replacement, seed search, support enrichment, threshold or protocol change.
- O data accesses0. There is no O cache, decoder or descriptor in the real
  development provider; evaluation authorization remains absent. Fold10 excluded.

Ledger and frozen artifacts: private/phase3b_cpsc_v1/. Aggregate-only release:
results/phase3b/. New implementation/tests are uncommitted additions in
src/dmvmat_ecg/phase3b_real/ and tests/phase3b_real/; old tracked files unchanged.
