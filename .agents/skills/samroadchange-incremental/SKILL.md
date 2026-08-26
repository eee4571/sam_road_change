---
name: samroadchange-development
description: Use for SamRoadChange code changes, debugging, refactors, module rewrites, and architecture adjustments that should acquire only the necessary repository context while choosing an implementation strategy based on the requested outcome rather than preserving legacy code by default.
---

# SamRoadChange Development Context

Use the smallest necessary repository context to understand the task, then choose the implementation strategy that best fits the requested outcome and the quality of the current design.

Minimize context acquisition, not implementation quality. The current codebase is context, not a constraint. Do not assume that preserving the existing implementation is desirable.

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

Do not recursively scan `runtime/`, `project/`, model directories, output directories, or unrelated engine modules merely to build general context.

## Choose the modification strategy

After locating the relevant implementation and necessary call chain, select the strategy according to the problem rather than defaulting to a local patch.

### Local patch

Use a local patch when the current design is sound, the problem is genuinely local, and the change does not introduce extra branches or duplicated responsibilities.

### Focused refactor

Use a focused refactor when the relevant code has clear duplication, excessive conditionals, misplaced responsibilities, or an internal interface that obstructs the requested behavior. Update direct callers together when that produces a clearer result.

### Module rewrite

Rewrite a module when its core approach no longer fits the requirement, when fallback or heuristic layers have made it harder to change than replace, or when the old implementation is itself the source of recurring complexity. It is acceptable to remove the obsolete implementation and replace it with a clearer one.

### Architectural change

Make an architectural change when the user requests redesign, the current data flow materially blocks the outcome, or preserving it would require adapters, bridges, duplicate pipelines, or parallel execution paths. Keep changes scoped to the affected architecture and preserve genuine project responsibility boundaries.

## Prevent compatibility-layer accumulation

Historical implementation is not a requirement. Unless the user explicitly requires backward compatibility or the behavior belongs to a stable external interface, do not preserve legacy internal behavior by adding:

- compatibility branches or legacy fallbacks;
- wrappers, adapters, or bridges around an obsolete design;
- parallel or duplicated processing pipelines;
- special-mode patches;
- additional heuristic layers.

Prefer removing, simplifying, merging, refactoring, or replacing obsolete logic over stacking another compatibility layer. If an existing internal interface or implementation causes unnecessary complexity, refactor it and its direct callers instead of building another abstraction around it.

Preserve backward compatibility for stable external interfaces and user-requested contracts. Internal interfaces may change when they obstruct a clear implementation, provided their affected callers and tests are updated consistently.

When the user says “重新设计”, “重新实现”, “重构”, “推翻当前实现”, “不要兼容旧逻辑”, “重新考虑这个模块”, “现在的方案不合理”, “不要继续打补丁”, or an equivalent phrase, treat it as explicit authorization to delete or replace the relevant internal implementation. Do not respond by retaining it behind another fallback or compatibility path.

## Preserve responsibility boundaries

Keep the existing dependency direction:

```text
GUI → app → backend → user_pipeline → engine
```

- GUI code handles presentation and interaction, not algorithms.
- Application managers coordinate GUI intent and backend operations.
- Backend and pipeline code orchestrate processing.
- Engine modules contain algorithm implementations.

Do not add `GUI → user_pipeline` or `GUI → engine` coupling. Within these layers, files, functions, internal interfaces, and implementations may be reorganized, consolidated, split, or rewritten when justified by the selected strategy.

## Verify proportionally

After editing:

1. Run syntax or import checks for the changed modules when applicable.
2. Run the smallest relevant test file or focused test cases.
3. Expand testing according to the actual impact of a refactor, rewrite, or interface change.
4. Inspect `git diff --stat` and the final diff for unintended edits, formatting churn, and overlap with user changes.

Do not default to the full test suite, model inference, or end-to-end remote-sensing workflows when they are unrelated to the affected behavior.

## Report the result

Briefly state:

- which files changed and why;
- whether public interfaces or architecture changed;
- which checks or tests ran and their results;
- any relevant behavior that remains unverified.

Keep the report scoped to this task rather than restating the repository architecture.
