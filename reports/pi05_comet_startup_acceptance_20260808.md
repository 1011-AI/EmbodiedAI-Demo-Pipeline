# PI0.5 Comet startup acceptance — 2026-08-08

## Scope

This acceptance covers the complete path used by the Baige entry point: prepared
backend verification, run-directory coordination, topology validation, JAX model
initialization, real Behavior1K data, optimizer updates, validation, both Orbax
checkpoint layouts, and exact resume.

## Corrected failure modes

1. Runtime startup no longer invokes `git` or applies patches. It verifies the
   immutable prepared-source manifest and source-tree hashes on the shared disk.
2. Only node rank 0 decides whether a new run directory is occupied. Other nodes
   join through a launch-session marker bound to the current run ID, coordinator,
   port, and world size.
3. All JAX processes create and synchronize the `weights`, `state`, and
   `state_manifests` directories before Orbax 0.11.13 constructs either manager.
4. Baige `pilot`, `formal`, and `formal_decay` require exactly 6 nodes × 8 GPUs.
   A 4 × 8 allocation is rejected before model allocation.
## Dynamic acceptance

### Four-GPU cold start

- Run: `pi05-coldstart-e2e-20260808-v1`
- Started with absent run, log, and checkpoint roots.
- Loaded all 51 parameter leaves with no missing, unexpected, or shape-mismatch
  entries; 3,353,433,872 parameters were trainable and none frozen.
- Completed two real optimizer steps and validation.
- Step 1: loss `0.0227872413`, grad norm `1.56442`.
- Step 2: loss `0.0217480790`, grad norm `1.20774`.
- Step 2 validation loss: `0.0189704224`.
- Both `weights/2` and `state/2` passed the independent checkpoint verifier:
  51 leaves, 204 available chunks, no missing chunks or metadata.

### Four-GPU stop and exact resume

- Run: `pi05-formal-resume-e2e-20260808-v1`
- First invocation saved exact state at step 1.
- The second invocation used the same run ID with exact resume, restored step 1,
  continued the sampler at global counter 32, and completed step 2 at counter 64.
- Step 1: loss `0.0151702957`, grad norm `0.945497`.
- Step 2 after restore: loss `0.0120132091`, grad norm `0.430591`.
- `state/2` passed the independent verifier with 51 leaves, 204 chunks, and no
  missing files. No temporary checkpoint directories remained.

### Environment and topology

- `scripts/pi05/doctor.py --require-gpus 4`: passed with JAX 0.5.3, Flax 0.10.2,
  Orbax 0.11.13, four visible GPUs, complete 5,893-chunk base checkpoint, complete
  data contracts, and matching prepared backend hashes.
- Simulated Baige formal preflight with 6 × 8: passed and resolved to 48 global
  devices, FSDP 8, and global batch 384.
- The same formal preflight with 4 × 8: rejected with
  `nodes=4, expected_nodes=6`.
- The failed Baige v2 allocation supplied only four ranks (32 GPUs). Its four
  collective manifests nevertheless confirmed `transport=ib`, GDR observed,
  and no socket fallback on every supplied rank.
- `python -m pytest -q tests/pi05`: 30 passed.

## Remaining external acceptance

No single-node machine can reproduce a six-host Orbax write. Multi-host startup
acceptance belongs to the separate `pilot` profile; the `formal` production path
contains no step-specific test actions and only validates and checkpoints at its
documented training cadence.
