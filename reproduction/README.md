# Reproduction code

Every figure PDF and every number in the paper. This directory exists because both sets of scripts used to
live under `analysis/`, which `.gitignore` excludes, so a clone could not redraw a figure or recompute a
statistic.

## figures/

| script | writes |
| :--- | :--- |
| `make_main_v2.py` | main figures: c01_threshold, c05_mechanism_full, c07_strategy, c13_stage2, c15_body_pass, c16_body_rot, c17_abs_width, c22_rot_heat, c47_mask |
| `make_v3.py` | v3 redraws: policy violins, body-swap panels, feasibility mask |
| `make_v4.py` | v4 palette refresh for main Figs 2, 4, 7, 8 |
| `make_supp.py` | supplementary figures s01 .. s10 |
| `make_supp_tables.py` | supplementary tables |
| `make_figs.py` | fig3_threshold, fig4_mechanism, fig5_strategy, fig6_transfer |
| `make_fig4_v5.py` | fancy_traj.pdf |
| `make_fig6_tall.py` | c13a_turns_tall.pdf |
| `make_geom.py` | fig_geometry.pdf |
| `where.py` | the OLD Figure 2 (where10.pdf); superseded, kept for comparison |
| `build_supp.py` | builds the supplementary document |
| `make_paper_figures.py` | tutorial-style paper figures |
| `render_fig2_compare.py` | renders the Figure 2 comparison |
| `plot_style.py` | shared plotting style, imported by `make_paper_figures.py` |
| `make_scene_dumbbell.py` | per-scene wide and narrow pass rates with Wilson 95% intervals (the Sec. 3 validity check); `python make_scene_dumbbell.py data/v8/S1_episodes.csv out.pdf` |

**Font requirement.** Seven of these scripts load Liberation Serif from `/usr/share/fonts/truetype/liberation/`, the path the figures were drawn on. Off that machine the `addfont` call raises before anything is drawn: install Liberation Serif at that path, or delete the loop and let matplotlib fall back to its default serif. The numbers do not depend on the font.

**These must stay in one directory**: `make_v3.py` imports `make_main_v2` and `make_v4`, and
`make_paper_figures.py` imports `plot_style`.

## stats/

| file | what it is |
| :--- | :--- |
| `v8_final_analysis.py` | the analysis: `pi_hat_bootstrap`, `interp_crossing`, the rotation profile, the Early metric |
| `pi_hat.csv` | pi-hat and its 95% interval per model and pooled |
| `s1_model_profile.csv` | the per-model profile Table 2 is built from |
| `s1_level_pooled.csv` | pooled per level |
| `s1_timing.csv` | early/late x met-required crosstab |
| `required_angles.csv` | theta*(A) for H1, G1, R1 |

The bootstrap is fully specified in the code: `boot=1500` per model and 2000 pooled, `seed=20261107`,
percentile 2.5/97.5, resampling episodes **within level**, `rot30 = max_rotation_deg >= 30`. Re-running it
reproduces Table 2's pi-hat column exactly, including the pooled 0.917 [0.906, 0.927].

## Still missing from the repository

These were produced on the machine that built the paper and are not in this checkout:

* the PPTX sources for Fig. 1 and Fig. 3 (`Fig1_v24.pptx`, `Fig2_v15.pptx`) -- both were in the download
  folder on 2026-11-07 and are gone now;
* whatever draws `fig/main/body_a.pdf`, `body_b.pdf`, `body_c.pdf` (Fig. 7), `fig/main/fig8_real.pdf`
  (Fig. 8) and `fig/main/policy_rows.pdf` (Fig. 5). No script in this checkout writes those filenames;
  `make_v3.py` and `make_v4.py` write `c15_body_pass.pdf`, `c16_body_rot.pdf`, `c17_abs_width.pdf` instead,
  which are structurally similar but not the same files.

## A known inconsistency in `make_v4.py`

Its Fig. 7(a) annotation reads `n=10/bin`, but the body-swap experiment ran **15, 15, 10, 10** episodes per
level (A/S 1.3, 1.1, 1.0, 0.9 -- verified in `论文材料/03_Stage3/S3_episodes_100.csv`: G1 and R1 each have
10 at 0.9, 10 at 1.0, 15 at 1.1, 15 at 1.3). The paper's version is correct; the script's label is stale.
