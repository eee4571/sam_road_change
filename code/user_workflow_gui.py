"""Backward-compatible launcher and public GUI API.

The implementation lives under :mod:`gui`; existing launch scripts and tests may
continue importing this module.
"""

from gui.main_window import *  # noqa: F401,F403 - intentional compatibility API
from gui.main_window import main
from app.backend_client import *  # noqa: F401,F403 - legacy helper exports
from app.editor_manager import *  # noqa: F401,F403 - legacy helper exports
from app.project_manager import *  # noqa: F401,F403 - legacy helper exports
from app.task_manager import *  # noqa: F401,F403 - legacy helper exports
from app.project_manager import _display_scope  # legacy underscored helper


if __name__ == "__main__":
    raise SystemExit(main())
