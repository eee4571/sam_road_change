# Cross-sensor change test

This independent developer tool validates the complete chain:

`sensor style -> period extraction -> two-stage change detection -> auto/review products`.

It does not change or sweep any single-period extraction threshold.

## 1. Generate a no-change degraded pair

```powershell
python dev_tools/cross_sensor_change_test/generate_degraded_pair.py `
  --input path/to/source.tif `
  --output-dir dev_tools/cross_sensor_change_test/outputs/no_change
```

The default degradation combines down/up sampling, Gaussian blur, brightness,
contrast, gamma, RGB response scaling and light noise without changing road geometry.

Use `--change added` or `--change removed` to also write a polygon truth layer.

## 2. Run the real extraction and change pipeline

```powershell
python dev_tools/cross_sensor_change_test/run_pair_test.py `
  --before-image .../A_original.tif `
  --after-image .../B_degraded.tif `
  --output-dir dev_tools/cross_sensor_change_test/outputs/no_change_run `
  --device cuda
```

Existing `latest_result.json` files can be reused with `--before-result` and
`--after-result`; no SAMRoad rerun occurs in that mode.

Outputs include `cross_sensor_report.json`, `cross_sensor_report.md`,
`cross_sensor_change_preview.png`, and `sensor_disagreement_preview.png`.

The official layers contain only `qa_state=auto`. Orange review candidates remain
separate and never become formal Added/Removed changes.

## 3. Full production pipeline no-change regression bench

`run_full_pipeline_pair_test.py` is the reusable end-to-end bench. It discovers
the newest existing no-change `A_original.tif` / `B_degraded.tif` pair, verifies
identical shape/CRS/transform/bounds, then invokes the production CLI only:

`user_pipeline.py prepare -> extract -> change`

Run both complete extractions and change detection from clean workspaces:

```powershell
python dev_tools/cross_sensor_change_test/run_full_pipeline_pair_test.py --device cuda --fresh
```

Reuse complete A/B extraction results and rerun only formal change detection:

```powershell
python dev_tools/cross_sensor_change_test/run_full_pipeline_pair_test.py --change-only
```

Explicit inputs take priority over discovery. Existing extraction results may
also seed the stable cache with `--before-result` and `--after-result`.

The default output is `outputs/full_pipeline_pair_test/`. It contains seven PNG
diagnostics, per-feature line/surface and width CSV audits, false-change JSON,
and `full_pipeline_report.json/.md`. Observed surface means the independent
`*_molra_clean_mask.png` (falling back to the source SAM-MoLRA mask); the final
Buffer-derived `road_surfaces.shp` is reported separately as product surface.

Exit semantics: `PASS` has no auto changes or major width disagreement; `WARN`
has review/suppressed or diagnostic width disagreements but no auto change;
`FAIL` contains at least one official auto Added/Removed/Widened/Narrowed.
