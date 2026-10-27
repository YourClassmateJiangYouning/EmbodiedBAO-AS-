# Stage 2 memory schema

Each line in `stage2_memory_cards_top7_15.jsonl` is one successful narrow-pass
memory.  A test uses exactly that memory card; there is no 15-card memory bank.

Fields:
- identity: `memory_id`, `model`, `scene`, `marker_slot`
- source condition: `source_level`, `source_as_ratio`, `channel_width_m`
- geometry: `required_turns`, `theta_req_deg`
- outcome: `outcome`, `total_steps`, `max_rotation_deg`, `look_down_count`,
  `final_position_x`, `final_position_z`, `final_torso_rotation`,
  `wall_collision_count`
- behaviour: `actions` (ordered action sequence)
- reasoning: `steps` (per-step action, feedback, reasoning, pose, collision,
  scene_description)
- `memory_text`: a prompt-ready rendering of the same information

The test plan (`stage2_top7_15_test_plan.csv`) pairs each memory with the SAME
scene and marker slot at Level 0 / A/S 2.0.  `wide_episode_id` is TBD because
Stage 2 has not been run.

Raw camera frames are not included: the Stage 1 runner did not save them.
