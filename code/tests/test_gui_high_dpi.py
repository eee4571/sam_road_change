from __future__ import annotations

import unittest
from tkinter import TclError, Tk, font as tkfont
from tkinter import ttk

from gui.common_widgets import treeview_metrics


class HighDpiLayoutSmokeTests(unittest.TestCase):
    def test_tree_rows_and_standard_controls_fit_font_at_common_windows_scaling(self) -> None:
        try:
            root = Tk()
        except TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        root.withdraw()
        try:
            style = ttk.Style(root)
            for scale in (1.0, 1.25, 1.5):
                root.tk.call("tk", "scaling", 96.0 * scale / 72.0)
                metrics = treeview_metrics(root, ("Microsoft YaHei UI", 9))
                line = tkfont.Font(root=root, font=metrics["font"]).metrics("linespace")
                self.assertGreaterEqual(metrics["rowheight"], line + 4)
                style.configure(
                    "Smoke.Data.Treeview", font=metrics["font"],
                    rowheight=metrics["rowheight"],
                )
                frame = ttk.LabelFrame(root, text="区域数据配置")
                button = ttk.Button(frame, text="重新扫描")
                combo = ttk.Combobox(frame, values=("中文期次",), state="readonly")
                tree = ttk.Treeview(frame, columns=("path",), show="headings", style="Smoke.Data.Treeview")
                for widget in (frame, button, combo, tree):
                    widget.pack()
                root.update_idletasks()
                self.assertGreaterEqual(tree.winfo_reqheight(), metrics["rowheight"])
                self.assertGreater(button.winfo_reqheight(), line)
                self.assertGreater(combo.winfo_reqheight(), line)
                frame.destroy()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
