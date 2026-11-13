# H1/H2 minimal integrity fix

Scope: H1 and H2 only. H3, the protocol-bundle proposal and all scientific
settings remain unchanged. This is implementation verification, not protocol
adoption or permission to train.

## Changes

- Modified `src/dmvmat_ecg/phase2d/access.py`: every real role read verifies the
  ledger and the complete trusted inventory before accessing protected data.
  Supplied inventories must exactly match the full inventory; omitting the
  argument computes the trusted inventory internally.
- Modified `src/dmvmat_ecg/phase2d/ledger.py`: real construction, reservation and
  active-fit entry verify the authoritative identity and current code/dependency
  hashes. Real construction requires the existing canonical database and rejects
  empty/unbound databases. Development/evaluation receipts must exactly match the
  identity, now including `ledger_id` and `canonical_path`.
- Created `src/dmvmat_ecg/phase2d/runtime.py`: fixed project/ledger locations and
  full-inventory verification using the **same `execution_inventory` rule as
  `scripts/phase2d_prepare.py`** (69 paths). Missing, extra or changed pins/source
  and dependency drift fail closed. Alternate locations and symlink aliases are
  rejected; use the direct resolved canonical path.
- Created `tests/phase2d/test_h1_h2_integrity.py`: 20 synthetic regression cases.
- Created this report.

The only real database changes were the explicitly required identity migration:
added `meta.ledger_id` and `meta.canonical_path` once; updated `meta.identity` with
those fields and the new code hash. The explicit migration accepts only the exact
old identity in an unsigned, zero-attempt draft; constructors never migrate or
create a real ledger. Repeating the migration is rejected.

## Signing/authorization identity — human action only

```text
code_hash = ed812868495e91aac456f955315635932031a67e250c798fad5127a831bc3afa
ledger_id = d6f784e8-5073-4fad-a804-1011820f92ef
canonical_path = /Users/kiyo/Desktop/dmvmat-ecg/private/phase2d_joint44301_v1/study.sqlite
dependency_hash = 195cd5ae983b6a3e196bc52ac0cf0997b26f60f45fb801e53256b8b57f881424
```

The dependency hash is unchanged. Existing historical inventory/preflight receipts
were preserved; their old code hash is not the current signing identity. A caller
must supply the full current inventory or let the boundary compute it internally.
Any subsequent inventoried-file change invalidates the current identity.

## Verification

Commands executed with the project environment:

- `.venv/bin/python -B -m pytest tests/phase2d/test_h1_h2_integrity.py -q -p no:cacheprovider`:
  **20 passed**.
- `.venv/bin/python -B -m pytest -q -p no:cacheprovider`:
  **206 passed, 7 existing warnings**, 37.03 seconds.
- `.venv/bin/ruff check .`: **passed**.
- `.venv/bin/ruff format --check .`: **passed**, 108 files already formatted in
  the final check.

Regression coverage: partial/missing/extra/empty/forged inventories; modified,
deleted and added source at read/reserve/active-fit boundaries; dependency drift;
existing canonical ledger reopen; alternate directories and directory/file
symlink aliases; absent/empty/unbound databases; UUID persistence and one-time
migration; development/evaluation receipt rejection without UUID/path fields.
All test projects, databases, authorizations and training fixtures were fictional
and temporary. The actual canonical ledger also opened successfully after its
metadata-only migration, without requesting data access or fitting.

## Preservation and stopping state

- Real ledger: **draft; 0/36 attempts; 0 evaluation accesses**; no protocol
  adoption, development authorization or evaluation authorization.
- Before/after comparison: `attempts`, `artifacts`, `events` and `access` tables
  identical (all empty); every pre-existing metadata value except `identity`
  identical. Manifest, contract and dependency hashes in the identity unchanged.
- All 95 tracked baseline files unchanged (`git diff --exit-code` passed).
  Public-file hash comparison found only the two modified source files listed
  above; all prior configs, reports, results and manifests remained unchanged.
- No real patient data, O vault, checkpoint, normalization or prediction artifact
  was opened or modified. No real training, inference, fitting or evaluation.
- Repository remains `main` at
  `1f0ff9031e9e1e2adc97962922fd97705570baf2`. Existing Phase 2 additions remain
  untracked alongside these changes; nothing staged, committed, pushed or merged.

H1/H2 regression checks pass. H3 and the unsigned scientific-review decisions
remain outside this fix; this report does not supersede the pre-signature audit.
