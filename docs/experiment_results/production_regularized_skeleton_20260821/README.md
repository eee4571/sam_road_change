# Production Regularized Skeleton audit

This lightweight synthetic audit loads `config/samroad_inference.yaml`, resolves
the production `auto` threshold profile, and runs `compute_relative_roadness()`.
It does not load a model, use a GPU, or run a 4096 image.

Regenerate from the repository root:

```powershell
env\samroad_env\python.exe dev_tools\production_relative_path_audit.py
```

Expected route: Relative Roadness → Regularized Skeleton, with Continuous Trace,
Junction Collapse, and Endpoint→Segment Recovery disabled.

Artifacts:

- `production_relative_path_result.png`: four-stage synthetic result image.
- `production_relative_path_audit.json`: selected route and stage timing audit.
- `regression_test.log`: 172 lightweight regression/unit tests; all passed.
