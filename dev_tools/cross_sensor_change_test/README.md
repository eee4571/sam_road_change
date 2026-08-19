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
