# Relative context embedding fix

This audit reproduces the GUI failure with a cached 4096×4096 road-probability
map. The Regularized Skeleton context contains variable-length vector paths,
which must remain a list rather than being converted into one raster array.

The fixed embedding step:

- pads only two-dimensional raster arrays;
- preserves vector paths and dictionary audits unchanged;
- reuses raster arrays when the target shape is already identical.

Artifacts:

- `relative_context_embed_4096_result.png`: full-image candidate/final skeleton comparison.
- `relative_context_embed_4096_audit.json`: shape, path-count, timing, and reuse audit.
- `regression_test.log`: focused production/Relative/weak/GUI/pipeline tests.
