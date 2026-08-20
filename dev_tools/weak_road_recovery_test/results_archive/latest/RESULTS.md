# Relative skeleton topology normalization — latest result

Source run: `outputs/relative_roadness_ab/junction_normalized_v2/`

This archive contains the PNG, CSV, and JSON artifacts from the final
4096×4096 recovery-only validation run. Model checkpoints, environments,
TIFF inputs, cache files, and pickle graphs are intentionally excluded.

## Key results

- `RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX` remained `48.0`.
- Narrow target corridor chains: `387 → 21`; too-short chains: `387 → 17`.
- Full requested crop chains: `523 → 106`; too-short chains: `516 → 95`.
- Whole-scene chains: `9823 → 2890`; too-short chains: `9553 → 2568`.
- Structure-rescued length: `23600.472984313965 px`.
- Relative Auto length: `21758.645646334164 → 34924.8054717484 px`.
- Final graph length: `43342.65234375 → 56508.81640625 px`.
- T/X junction, short real branch, roof-grid, parallel-road, and ring
  regressions passed.

Use `relative_junction_debug.png`, `relative_chain_debug.png`, and
`relative_skeleton_normalization.json` for the primary visual and numerical
audit.
