# Relative skeleton hole/cycle cleanup — problem crop

- Source image: `dev_tools/weak_road_recovery_test/1.tif`
- Source crop `(x0, y0, x1, y1)`: `(2418, 1421, 3179, 2722)`
- Junction collapse: disabled (default A variant)
- Raw cycles: 27
- Removed small cycles: 23
- Preserved large cycles: 4
- Final cycles: 4
- Candidate holes: 146
- Filled holes: 122
- Preserved holes: 24
- Narrow cracks filled: 13
- Small-cycle pixels near a detected hole/crack: 98.0%
- Small-cycle pixels near a filled hole/crack: 67.7%
- Regularized-skeleton pipeline: 0.838 s
- Full crop recovery/visualization: 7.208 s
- Verification: 81 unit/synthetic/dev-tool tests passed

`relative_hole_cycle_debug.png` is the primary seven-stage diagnostic.
The JSON files contain the complete hole/cycle and performance audits.
