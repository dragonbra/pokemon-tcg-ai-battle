"""0044 V1 test boundary.

The project was frozen from the 0043 tree for implementation provenance, but
0043-only promotion and V2--V6 restart tests are not executable 0044 contracts.
They remain in the source snapshot for audit and are deliberately not collected.
"""

collect_ignore = [
    "test_cuda_engine_2_inference.py",       # old 0809/G1 two-policy smoke
    "test_cuda_engine_2_policy_pool.py",     # old focal-seed/three-policy pool
    "test_focal_seed_initialization.py",     # old G1 -> expanded G1 seed
    "test_focal_seed_parity.py",             # old generated seed evidence
    "test_focal_seed_runtime.py",            # old three-policy registry
    "test_frozen_schedule.py",               # old FrozenMeta256 / 0809 eval
    "test_g2_candidate_gate.py",              # 0043 G2 promotion experiment
    "test_g2_candidate_full67.py",            # 0043 G2 promotion experiment
    "test_initial_run_acceptance.py",         # old 67 x 2 opponent contract
    "test_phase0_contract.py",                # 0043 initial-run documents
    "test_policy_identity.py",                # old 0809/G1 pool assertions
    "test_promote.py",                        # champion promotion, not V1 training
    "test_runtime.py",                        # old 0809/G1 runtime assertions
    "test_training_regression.py",            # old approved 0043 config snapshot
    "test_v3_generalist.py",                  # historical 0043 run
    "test_v4_restart.py",                     # historical 0043 run
    "test_v5_memory_safe.py",                 # historical 0043 run
    "test_v6_restart.py",                     # historical 0043 run
]
