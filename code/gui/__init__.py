"""Tkinter presentation layer for the SAMRoad desktop workflow."""

__all__ = ["UserApp", "main"]


def __getattr__(name: str):
    if name in __all__:
        from .main_window import UserApp, main
        return {"UserApp": UserApp, "main": main}[name]
    raise AttributeError(name)
