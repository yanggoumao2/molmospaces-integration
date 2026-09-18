"""MimicGen bridge for FloorPlan1 bimanual YAM pick-and-place.

Design contract:
- Follows official MimicGen (Mandlekar et al. CoRL 2023) subtask/waypoint pipeline.
- Bimanual coordination follows DexMimicGen sequential mode: right arm executes
  subtasks 0..4 first, left arm executes subtasks 5..9 after the right arm has
  finished. There is a single shared MuJoCo env; the two arms are two
  MG_EnvInterface instances over the same env.
- Single source demonstration (no demo pool). MimicGen selection_strategy is
  forced to random and demo_keys always contains one entry.
"""
