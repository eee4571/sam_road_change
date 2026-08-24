---
name: samroadchange-incremental
description: Use for routine SamRoadChange code changes, Tkinter GUI adjustments, debugging, and focused refactors that should preserve the existing architecture and build on the current working-tree context without rescanning the whole repository.
---

# SamRoadChange Incremental Development

Work incrementally from the repository's current state. Keep the task boundary narrow, preserve local edits, and avoid rebuilding repository context that is already documented or visible in the diff.

## Start with repository context

Before reading implementation files:

1. Use the applicable `AGENTS.md` instructions already loaded by Codex. Do not reread the root file by default; read it only when instruction scope is unclear, the file has changed, or the applicable instructions are unavailable.
2. Treat `docs/CODEMAP.md` as a navigation index. Search it for task-specific headings or keywords and read only the matching sections; do not load the whole file unless the task genuinely spans it.
3. Run `git status`, `git diff --stat`, and inspect the current diff relevant to the requested files.

When work enters a nested scope, load its applicable nested `AGENTS.md` only if those instructions are not already available. If a needed instruction or index file is absent, use the closest clearly equivalent repository document without recursively scanning the repository.

Treat all pre-existing tracked and untracked changes as user work. Do not overwrite, revert, reformat, or fold them into the task unless the request explicitly requires it.

## Select the smallest useful reading set

Use `AGENTS.md`, the code map, the request, and targeted search to identify the minimum relevant files. Read only:

- the files likely to change;
- their necessary direct callers or direct dependencies;
- focused tests that exercise the affected behavior.

Expand this set only when a concrete unresolved dependency or behavior requires it. Prefer targeted `rg` searches and bounded file sections over directory-wide traversal.

For GUI-only layout, wording, styling, or widget changes, default to the relevant files under `code/gui/`. Do not read `code/user_pipeline.py` or `code/engine/` unless the requested behavior demonstrably crosses into pipeline or algorithm execution.

Before changing anything under `code/gui/`, follow the applicable `code/gui/AGENTS.md` instructions, loading that file only when they are not already available. If it is absent, follow the applicable root instructions.

## Preserve project boundaries

Keep the existing dependency direction:

```text
GUI → app → backend → user_pipeline → engine
```

- GUI code handles presentation and interaction, not algorithms.
- Application managers coordinate GUI intent and backend operations.
- Backend and pipeline code orchestrate processing.
- Engine modules contain algorithm implementations.

Reuse existing interfaces and managers before introducing new paths between layers. Do not add `GUI → user_pipeline` or `GUI → engine` coupling. Avoid unrelated refactors, interface changes, file moves, or formatting churn.

## Make focused changes

- Base edits on the current diff and nearby code conventions.
- Change only what is required for the requested outcome.
- Preserve backward compatibility, task recovery, and existing behavior outside the task boundary.
- When debugging, establish a focused reproduction or evidence trail before editing.
- If the necessary fix expands into another layer, explain why and inspect only the relevant entry point in that layer.

## Verify proportionally

After editing:

1. Run syntax or import checks for the changed modules when applicable.
2. Run the smallest relevant test file or focused test cases.
3. Expand testing only when the observed impact justifies it.
4. Inspect `git diff --stat` and the final diff for unintended edits, formatting churn, and overlap with user changes.

Do not default to the full test suite, model inference, or end-to-end remote-sensing workflows for a local GUI or narrowly scoped code change.

## Report the result

Briefly state:

- which files changed and why;
- whether public interfaces or architecture changed;
- which checks or tests ran and their results;
- any relevant behavior that remains unverified.

Keep the report scoped to this task rather than restating the repository architecture.
