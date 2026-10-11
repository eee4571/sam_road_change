"""Paired-image visual review. No GIS or inference imports in the host."""
import base64
import re
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QUrl, QTimer, QSize
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QListWidget, QListWidgetItem, QSplitter, QWidget,
    QGraphicsView, QGraphicsScene, QGraphicsPathItem, QMessageBox, QMenu,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)

STATUS_TEXT = {"pending": "待复核", "keep": "已保留", "reject": "已剔除"}
STATUS_COLOR = {'pending': '#8b6214', 'keep': '#137c51', 'reject': '#687280'}
PAIR_GRID_ROLE = int(Qt.ItemDataRole.UserRole) + 1
CHANGE_TEXT = {"added": "道路新增", "removed": "道路消失", "widened": "道路变宽",
               "narrowed": "道路变窄", "width_changed": "道路宽度变化"}


def change_text(value):
    return CHANGE_TEXT.get(str(value).strip().lower(), "道路变化")


def geometry_path(geometry):
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)

    def line(coords, closed=False):
        if not coords:
            return
        path.moveTo(*coords[0][:2])
        for point in coords[1:]:
            path.lineTo(*point[:2])
        if closed:
            path.closeSubpath()

    kind, coords = geometry["type"], geometry.get("coordinates", [])
    if kind == "Polygon":
        for ring in coords:
            line(ring, True)
    elif kind == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                line(ring, True)
    elif kind == "LineString":
        line(coords)
    elif kind == "MultiLineString":
        for points in coords:
            line(points)
    elif kind == "Point":
        path.addEllipse(coords[0] - 4, coords[1] - 4, 8, 8)
    elif kind == "MultiPoint":
        for point in coords:
            path.addEllipse(point[0] - 4, point[1] - 4, 8, 8)
    elif kind == "GeometryCollection":
        for child in geometry.get("geometries", []):
            path.addPath(geometry_path(child))
    return path


class ImageView(QGraphicsView):
    moved = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#313740"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(180, 200)
        self.overlay = None
        self.horizontalScrollBar().valueChanged.connect(self.moved)
        self.verticalScrollBar().valueChanged.connect(self.moved)

    def clear_image(self):
        self.overlay = None
        self.scene().clear()

    def load(self, encoded, geometry):
        self.clear_image()
        image = QImage.fromData(base64.b64decode(encoded), "PNG")
        if image.isNull():
            raise ValueError("影像预览解码失败")
        self.scene().addPixmap(QPixmap.fromImage(image))
        self.overlay = QGraphicsPathItem(geometry_path(geometry))
        pen = QPen(QColor("#ffdd36"), 2)
        pen.setCosmetic(True)
        self.overlay.setPen(pen)
        self.scene().addItem(self.overlay)
        self.scene().setSceneRect(0, 0, image.width(), image.height())
        self.fit()

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else .8
        if .05 <= self.transform().m11() * factor <= 24:
            self.scale(factor, factor)
            self.moved.emit()
        event.accept()


class ChangeReviewDialog(QDialog):
    def __init__(self, service, store, parent=None):
        super().__init__(parent)
        self.service, self.store = service, store
        self.features, self.undo_stack = [], []
        self.dirty = self.waiting = self.preview_valid = self._syncing = False
        self.pair_index = -1
        self.saved_path = ""
        self.saved_decisions = {}
        self.drafts = {}
        self._signature = None
        self._save_queue = None
        self._auto_advance = False
        self._pending_undo = None
        self._save_origin = -1
        self.pair_states = {}
        self.record_unsaved = False
        self._record_after = None
        self._editing_complete = False
        self._application_feedback = ''
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._preview)
        self.setWindowTitle("人工复核变化")
        self.resize(1280, 820)
        self.setMinimumSize(800, 580)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 14)
        layout.setSpacing(12)
        self.setStyleSheet('''
            QDialog { background: #fafbfc; color: #202b36; font-size: 14px; }
            QLabel, QPushButton, QComboBox, QListWidget { font-size: 14px; }
            QLabel { color: #202b36; }
            QPushButton { padding: 7px 14px; border: 1px solid #ccd4dc;
                border-radius: 6px; background: white; color: #253442; }
            QPushButton:hover { background: #edf3f8; }
            QPushButton:disabled { color: #9aa5af; background: #f1f3f5; }
            QPushButton#apply { background: #183d55; color: white; border: none; }
            QPushButton#apply:disabled { background: #dce3e8; color: #82909a; }
            QPushButton#keep { color: #137c51; border: 1px solid #329369; font-size: 16px; }
            QPushButton#reject { color: #754744; font-size: 16px; }
            QListWidget { background: white; border: 1px solid #e0e5eb; border-radius: 6px; }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #eef1f4; }
            QListWidget::item:selected { background: #e5eff7; color: #183d55; }
            QComboBox { padding: 5px 8px; min-height: 22px; }
            QLabel#muted { color: #667581; font-size: 12px; }
            QLabel#heading { font-size: 22px; font-weight: 600; }
        ''')
        heading = QHBoxLayout()
        title = QLabel('人工复核变化')
        title.setObjectName('heading')
        heading.addWidget(title)
        heading.addStretch()
        self.summary_button = self.button('复核汇总', self._summary)
        heading.addWidget(self.summary_button)
        self.save_button = self.button('应用复核结果', self._save)
        self.save_button.setObjectName('apply')
        self.save_button.setToolTip('更新所有已修改组的正式变化、长时序和可用的精度评价（Ctrl+S）')
        heading.addWidget(self.save_button)
        layout.addLayout(heading)
        self.pairs = QComboBox()
        self.pairs.setMinimumContentsLength(24)
        self.pairs.currentIndexChanged.connect(self._choose_pair)
        top = QHBoxLayout()
        top.addWidget(QLabel("区域与期次"))
        top.addWidget(self.pairs, 1)
        self.counts = QLabel()
        top.addWidget(self.counts)
        layout.addLayout(top)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        sidebar = QWidget()
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        self.list_title = QLabel('变化候选')
        side.addWidget(self.list_title)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel('复核状态'))
        self.filter = QComboBox()
        for text, value in (("全部", "all"), ("待复核", "pending"), ("已保留", "keep"), ("已剔除", "reject")):
            self.filter.addItem(text, value)
        self.filter.currentIndexChanged.connect(self._filter)
        filter_row.addWidget(self.filter, 1)
        side.addLayout(filter_row)
        self.items = QListWidget()
        self.items.currentRowChanged.connect(self._selection)
        self.items.itemClicked.connect(self._edit_completed)
        side.addWidget(self.items, 2)
        self.details = QLabel()
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        splitter.addWidget(sidebar)
        compare = QWidget()
        compare_layout = QVBoxLayout(compare)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.details, 1)
        self.overlay_check = QCheckBox("显示轮廓")
        self.overlay_check.setToolTip('空格键显示或隐藏黄色候选轮廓')
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(self._overlay)
        toolbar.addWidget(self.overlay_check)
        self.context = QComboBox()
        for text, value in (("近景", 1.5), ("中景", 3), ("远景", 6)):
            self.context.addItem(text, value)
        self.context.currentIndexChanged.connect(self._preview)
        toolbar.addWidget(self.context)
        toolbar.addWidget(self.button("居中", self._fit))
        compare_layout.addLayout(toolbar)
        panels = QHBoxLayout()
        self.views = [ImageView(), ImageView()]
        self.titles = [QLabel("前期"), QLabel("后期")]
        for title, view in zip(self.titles, self.views):
            panel = QVBoxLayout()
            panel.addWidget(title)
            panel.addWidget(view, 1)
            panels.addLayout(panel, 1)
            view.moved.connect(lambda v=view: self._sync(v))
        compare_layout.addLayout(panels, 1)
        self.view_note = QLabel("滚轮缩放 · 拖动平移 · 两期同步")
        self.view_note.setObjectName('muted')
        self.view_note.setWordWrap(True)
        compare_layout.addWidget(self.view_note)
        self.completion = QWidget()
        complete_row = QHBoxLayout(self.completion)
        complete_row.setContentsMargins(0, 0, 0, 0)
        complete_row.addWidget(QLabel('本组已完成复核'), 1)
        complete_row.addWidget(self.button('查看复核汇总', self._summary))
        complete_row.addWidget(self.button('修改当前判断', self._edit_completed))
        compare_layout.addWidget(self.completion)
        self.decision_bar = QWidget()
        decisions = QHBoxLayout(self.decision_bar)
        decisions.setContentsMargins(0, 0, 0, 0)
        self.keep = self.button("有真实变化 · 保留（1）", lambda: self._mark("keep"))
        self.reject_button = self.button("无真实变化 · 剔除（2）", lambda: self._mark("reject"))
        self.keep.setObjectName('keep')
        self.reject_button.setObjectName('reject')
        for button in (self.keep, self.reject_button):
            button.setMinimumHeight(40)
            decisions.addWidget(button)
        compare_layout.addWidget(self.decision_bar)
        controls = QHBoxLayout()
        self.previous = self.button("‹ 上一条", lambda: self._step(-1))
        self.next = self.button("下一条 ›", lambda: self._step(1))
        self.undo = self.button('撤销', self._undo)
        self.undo.setToolTip('撤销上一次判断（Ctrl+Z）')
        for button in (self.previous, self.next, self.undo):
            controls.addWidget(button)
        more = self.button('更多', lambda: None)
        menu = QMenu(more)
        self.next_pending = menu.addAction('暂时跳过（3）', self._next_pending)
        self.reset = menu.addAction('设为待复核', lambda: self._mark('pending'))
        self.open_button = menu.addAction('查看正式成果文件', self._open_output)
        menu.addAction('操作说明', self._help)
        more.setMenu(menu)
        controls.addWidget(more)
        controls.addStretch()
        self.record_label = QLabel('判断后自动保存')
        self.record_label.setObjectName('muted')
        controls.addWidget(self.record_label)
        compare_layout.addLayout(controls)
        splitter.addWidget(compare)
        splitter.setSizes([240, 1000])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.message = QLabel("正在准备影像，请稍候…")
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        footer = QHBoxLayout()
        self.message.setObjectName('muted')
        footer.addWidget(self.message, 1)
        self.retry_record = self.button('重试保存判断', lambda: self._persist(self._record_after))
        self.retry_record.hide()
        footer.addWidget(self.retry_record)
        footer.addWidget(self.button("关闭", self.close))
        layout.addLayout(footer)
        for key, callback in (("1", self.keep.click), ("2", self.reject_button.click),
                              ("3", self.next_pending.trigger), ("Ctrl+S", self.save_button.click),
                              ("Ctrl+Z", self.undo.click), ("Space", self.overlay_check.toggle)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback)
        service.replied.connect(self._reply)
        service.failed.connect(self._failed)
        service.stopped.connect(self._stopped)
        service.progress.connect(self._save_progress)
        self._controls(True)

    @staticmethod
    def button(text, callback):
        button = QPushButton(text)
        button.setAutoDefault(False)
        button.clicked.connect(callback)
        return button

    def start(self):
        try:
            self.service.start(self.store)
        except (OSError, ValueError) as exc:
            self._stopped(str(exc))

    def dispose(self):
        self.service.replied.disconnect(self._reply)
        self.service.failed.disconnect(self._failed)
        self.service.stopped.disconnect(self._stopped)
        self.service.progress.disconnect(self._save_progress)
        self.service.close()

    def _controls(self, waiting=None):
        if waiting is not None:
            self.waiting = waiting
        ready = not self.waiting and self.service.active and self._save_queue is None and not self.record_unsaved
        selected = self.items.currentRow() >= 0 and bool(self.features)
        for control in (self.pairs, self.filter, self.items, self.context):
            control.setEnabled(ready)
        for control in (self.previous, self.next, self.next_pending, self.reset):
            control.setEnabled(ready and selected)
        for control in (self.keep, self.reject_button):
            control.setEnabled(ready and selected and self.preview_valid)
        self.save_button.setEnabled(ready and self.dirty)
        self.summary_button.setEnabled(ready and self.pairs.count() > 0)
        self.retry_record.setVisible(self.record_unsaved and not self.waiting)
        self.retry_record.setEnabled(self.service.active and not self.waiting)
        self.next_pending.setEnabled(ready and any(f['status'] == 'pending' for f in self.features))
        self.reset.setEnabled(ready and selected and self.features[self.items.currentRow()]['status'] != 'pending')
        self.open_button.setEnabled(bool(self.saved_path))
        self.undo.setEnabled(ready and bool(self.undo_stack))
        complete = self.pair_index >= 0 and not any(f['status'] == 'pending' for f in self.features)
        self.completion.setVisible(complete)
        self.completion.setEnabled(ready)
        self.decision_bar.setVisible(not complete or self._editing_complete)
        if complete and not self._editing_complete:
            self.keep.setEnabled(False)
            self.reject_button.setEnabled(False)

    def _persist(self, after=None):
        self._record_after = after
        self.record_unsaved = True
        self.preview_timer.stop()
        self.record_label.setText('正在自动保存判断…')
        self._request('record', decisions={f['id']: f['status'] for f in self.features})

    def _edit_completed(self, *_):
        self._editing_complete = True
        self._controls()

    def _help(self):
        QMessageBox.information(self, '复核操作说明',
            '1. 查看前后期影像，按 1 保留真实变化，按 2 剔除非变化。每次判断自动保存，随后进入下一条。\n'
            '2. 本组完成后自动进入下一期次或区域。可随时在复核汇总中返回修改。\n'
            '3. 点击“应用复核结果”才更新正式变化、长时序及可用的精度评价。未复核项暂时保留。\n\n'
            '滚轮缩放、拖动平移，两期同步；空格显示/隐藏轮廓；Ctrl+Z 撤销；3 暂时跳过。')

    def _summary(self):
        if self.waiting or self.record_unsaved or self._save_queue is not None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('复核汇总')
        dialog.resize(850, 460)
        layout = QVBoxLayout(dialog)
        note = QLabel('判断记录自动保存。应用后才更新正式成果；未复核项暂时保留。')
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QTableWidget(self.pairs.count(), 5, dialog)
        table.setHorizontalHeaderLabels(['区域与期次', '待复核', '已保留', '已剔除', '成果状态'])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(self.pairs.count()):
            state = self.pair_states.get(i, {})
            counts = state.get('counts', {})
            values = [self.pairs.itemText(i)] + [str(counts.get(s, '—')) if state.get('known') else '—'
                                               for s in ('pending', 'keep', 'reject')]
            values.append('待应用' if state.get('modified') else '与正式成果一致' if state.get('known') else '尚未打开')
            for col, value in enumerate(values):
                table.setItem(i, col, QTableWidgetItem(value))
        table.selectRow(max(0, self.pair_index))
        layout.addWidget(table, 1)
        actions = QHBoxLayout()
        def review():
            index = table.currentRow()
            if index < 0:
                return
            dialog.accept()
            self._editing_complete = True
            if index != self.pair_index:
                self._load_pair(index)
                self._editing_complete = True
            else:
                self._controls()
        actions.addWidget(self.button('返回所选组复核', review))
        actions.addStretch()
        apply = self.button('应用复核结果', lambda: (dialog.accept(), self._save()))
        apply.setEnabled(self.dirty)
        actions.addWidget(apply)
        layout.addLayout(actions)
        table.cellDoubleClicked.connect(lambda *_: review())
        dialog.exec()

    def _request(self, op, **data):
        self._controls(True)
        try:
            self.service.request(op, **data)
        except ValueError as exc:
            self._failed(op, str(exc))

    def _reply(self, op, data):
        self._controls(False)
        if op == "init":
            self.pairs.blockSignals(True)
            self.pairs.clear()
            def order(item):
                return tuple(tuple((0, int(v)) if v.isdigit() else (1, v.casefold())
                                   for v in re.split(r'(\d+)', str(item[1][key])))
                             for key in ('grid', 'before_period', 'after_period'))
            for index, pair in sorted(enumerate(data['pairs']), key=order):
                self.pairs.addItem(f'{pair["grid"]} / {pair["before_period"]} → {pair["after_period"]}', index)
                self.pairs.setItemData(self.pairs.count() - 1, pair['grid'], PAIR_GRID_ROLE)
                if data.get('states'):
                    self.pair_states[self.pairs.count() - 1] = data['states'][index]
            self.pairs.blockSignals(False)
            if self.pairs.count():
                self._load_pair(0)
            else:
                self.message.setText("当前项目没有可复核的变化成果")
        elif op == "open":
            index = self.pairs.currentIndex()
            if data['warning'] and self._save_queue is not None:
                self._failed('open', data['warning'])
                return
            signature = (data['source_signature'], data['published_signature'])
            draft = self.drafts.get(index)
            if draft and draft['signature'] != signature and (draft['decisions'] != draft['saved'] or self._pending_undo):
                self._failed('open', '这组检测结果已改变，未套用原判断记录。请关闭并重新打开复核。')
                return
            if draft and draft['signature'] != signature:
                self.undo_stack = [action for action in self.undo_stack if action[0] != index]
            self.pair_index = self.pairs.currentIndex()
            self.features = data["features"]
            self.saved_decisions = data.get('applied_decisions', {f['id']: f['status'] for f in self.features})
            self._signature = signature
            if draft and draft['signature'] == signature and draft['decisions'] != draft['saved']:
                for feature in self.features:
                    feature['status'] = draft['decisions'][feature['id']]
            self.saved_path = data["saved_path"]
            self._remember_pair()
            self.items.blockSignals(True)
            self.items.clear()
            for i, feature in enumerate(self.features):
                self.items.addItem(QListWidgetItem(self._item_text(i)))
                self._style_item(i)
            self.items.blockSignals(False)
            self.titles[0].setText("前期 · " + str(data["pair"]["before_period"]))
            self.titles[1].setText("后期 · " + str(data["pair"]["after_period"]))
            self.message.setText(data["warning"] or self._application_feedback or
                                 '判断记录自动保存；点击“应用复核结果”才更新正式成果。')
            self._application_feedback = ''
            self.record_label.setText('判断记录已保存' if any(f['status'] != 'pending' for f in self.features) else '判断后自动保存')
            if not self.features:
                self.message.setText("这组影像没有检测到变化，无需确认。")
            self._counts()
            if self._save_queue is not None:
                self._save_current()
                return
            if self._pending_undo is not None:
                _, row, state = self._pending_undo
                self._pending_undo = None
                self._set_status(row, state)
                self._show_item(row)
                self._persist(lambda: self._return_to_item(row))
                return
            if self._auto_advance and not any(f['status'] == 'pending' for f in self.features):
                self._advance_pair()
                return
            if self._auto_advance:
                self.message.setText('已自动进入下一组，之前的判断记录已保存。')
            self._auto_advance = False
            first = next((i for i, f in enumerate(self.features) if f['status'] == 'pending'), 0)
            self._show_item(first)
        elif op == "preview":
            if self.items.currentRow() < 0 or data['id'] != self.features[self.items.currentRow()]['id']:
                self.preview_timer.start(0)
                return
            try:
                self._syncing = True
                for view, image in zip(self.views, data["images"]):
                    view.load(image, data["geometry"])
                self._overlay(self.overlay_check.isChecked())
                self.preview_valid = data["has_images"]
                self.view_note.setText('；'.join(data['warnings']) or
                    f'滚轮缩放 · 拖动平移 · 两期同步   |   当前范围约 {data["span_m"]:.0f} 米')
            except ValueError as exc:
                self._failed(op, str(exc))
            finally:
                self._syncing = False
        elif op == 'record':
            self.record_unsaved = False
            self.record_label.setText('判断记录已自动保存')
            after, self._record_after = self._record_after, None
            self._controls()
            if after:
                after()
            return
        elif op == "save":
            self.saved_decisions = {f['id']: f['status'] for f in self.features}
            self._signature = (self._signature[0], data['published_signature'])
            self.saved_path = data["path"]
            self._remember_pair()
            evaluation = '精度评价已重新计算。' if data.get('evaluation_state') == 'completed' else '这组未配置真值数据，未计算精度。'
            self.message.setText('已应用复核结果，正式变化、长时序和统计已更新。' + evaluation)
            self._application_feedback = self.message.text()
            self.open_button.setToolTip(self.saved_path)
            self._counts()
            self._save_next()
            return
        self._controls()

    def _failed(self, op, error):
        saving = self._save_queue is not None
        self._save_queue = None
        self._auto_advance = False
        self._application_feedback = ''
        if self._pending_undo is not None:
            self.undo_stack.append(self._pending_undo)
            self._pending_undo = None
        if op == "preview":
            self.preview_valid = False
            for view in self.views:
                view.clear_image()
            self.view_note.setText("影像未能显示，请检查影像文件；也可以暂时跳过这一条。")
        if op == "open":
            self.pair_index = -1
        if op == 'record':
            self.record_label.setText('判断尚未保存，请重试')
        self.message.setText(("应用未全部完成，判断记录已保留，可重试：" if saving else "操作失败：") + error)
        self._controls(False)

    def _save_progress(self, message):
        self.message.setText(message + ' 请等待更新完成后再关闭。')

    def _stopped(self, error):
        self._failed('stopped', error)

    def _confirm_close(self, continuation):
        if not self.record_unsaved:
            continuation()
            return
        prompt = QMessageBox(self)
        prompt.setWindowTitle("复核修改尚未保存")
        prompt.setText("最近一次判断尚未自动保存成功。")
        prompt.setInformativeText('直接退出会丢失这次未保存的修改，之前的判断记录仍保留。')
        prompt.setIcon(QMessageBox.Icon.Warning)
        prompt.setStandardButtons(QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        for button, title in ((QMessageBox.StandardButton.Save, "保存并退出"),
                              (QMessageBox.StandardButton.Discard, "不保存退出"),
                              (QMessageBox.StandardButton.Cancel, "继续复核")):
            prompt.button(button).setText(title)
        prompt.setDefaultButton(QMessageBox.StandardButton.Save)
        prompt.setEscapeButton(QMessageBox.StandardButton.Cancel)
        choice = prompt.exec()
        if choice == QMessageBox.StandardButton.Save:
            self._persist(continuation)
        elif choice == QMessageBox.StandardButton.Discard:
            self.dirty = False
            self.drafts.clear()
            continuation()

    def _choose_pair(self, index):
        if index < 0 or index == self.pair_index:
            return
        self._auto_advance = False
        self._load_pair(index)

    def _load_pair(self, index):
        self._editing_complete = False
        self._remember_pair()
        self.preview_timer.stop()
        self.pairs.blockSignals(True)
        self.pairs.setCurrentIndex(index)
        self.pairs.blockSignals(False)
        self.items.blockSignals(True)
        self.items.clear()
        self.items.blockSignals(False)
        self.features = []
        self.saved_decisions = {}
        self.pair_index = -1
        self.saved_path = ""
        self.preview_valid = False
        self.details.clear()
        self._counts()
        for view in self.views:
            view.clear_image()
        self.message.setText("正在准备这一组的变化与影像…")
        self._request("open", index=self.pairs.itemData(index))

    def _remember_pair(self):
        if self.pair_index >= 0:
            self.drafts[self.pair_index] = {
                'decisions': {f['id']: f['status'] for f in self.features},
                'saved': dict(self.saved_decisions), 'signature': self._signature}
            self.pair_states[self.pair_index] = {'known': True,
                'modified': self.drafts[self.pair_index]['decisions'] != self.saved_decisions,
                'counts': {s: sum(f['status'] == s for f in self.features) for s in STATUS_TEXT}}
        self.dirty = any(d.get('modified') for d in self.pair_states.values())

    def _item_text(self, i):
        feature = self.features[i]
        return f'{i + 1:04d} · {change_text(feature["change_type"])}\n{STATUS_TEXT[feature["status"]]}'

    def _style_item(self, i):
        state = self.features[i]['status']
        color = QColor(STATUS_COLOR[state])
        icon = QPixmap(16, 16)
        icon.fill(Qt.GlobalColor.transparent)
        painter = QPainter(icon)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(color, 2))
        if state == 'keep':
            painter.drawLine(2, 8, 6, 12)
            painter.drawLine(6, 12, 14, 3)
        elif state == 'reject':
            painter.drawLine(3, 3, 13, 13)
            painter.drawLine(3, 13, 13, 3)
        else:
            painter.drawEllipse(3, 3, 10, 10)
        painter.end()
        self.items.item(i).setIcon(QIcon(icon))
        self.items.item(i).setForeground(color)
        self.items.item(i).setSizeHint(QSize(180, 64))

    def _counts(self):
        counts = {state: sum(f["status"] == state for f in self.features) for state in STATUS_TEXT}
        self.counts.setText(f'已复核 {counts["keep"] + counts["reject"]}/{len(self.features)}  ·  保留 {counts["keep"]}  ·  剔除 {counts["reject"]}')
        self.list_title.setText(f'变化候选（{len(self.features)}）')
        pending = sum(bool(d.get('modified')) for d in self.pair_states.values())
        self.save_button.setText(f'应用复核结果（{pending} 组）' if pending else '应用复核结果')

    def _filter(self, *_, preferred=None):
        if not self.features:
            self._controls()
            return
        value = self.filter.currentData()
        visible = []
        for i, feature in enumerate(self.features):
            hidden = value != "all" and feature["status"] != value
            self.items.item(i).setHidden(hidden)
            if not hidden:
                visible.append(i)
        if preferred in visible:
            self.items.setCurrentRow(preferred)
        elif self.items.currentRow() not in visible:
            self.items.setCurrentRow(visible[0] if visible else -1)
        self._controls()

    def _show_item(self, index):
        if not 0 <= index < len(self.features):
            self._controls()
            return
        if self.filter.currentData() not in {'all', self.features[index]['status']}:
            self.filter.blockSignals(True)
            self.filter.setCurrentIndex(0)
            self.filter.blockSignals(False)
        self._filter(preferred=index)
        self._details()

    def _details(self):
        index = self.items.currentRow()
        self.details.setText('' if index < 0 else
            f'候选 {index + 1:04d} · {change_text(self.features[index]["change_type"])}')
        if index >= 0:
            self.record_label.setText(f'{STATUS_TEXT[self.features[index]["status"]]} · 判断自动保存')

    def _selection(self, index):
        self.preview_valid = False
        self.details.clear()
        self._syncing = True
        for view in self.views:
            view.clear_image()
        self._syncing = False
        if index < 0 or index >= len(self.features):
            self._controls()
            return
        self._details()
        self._controls()
        self.preview_timer.start(0)

    def _preview(self, *_):
        index = self.items.currentRow()
        if index < 0 or not self.features or self.waiting:
            return
        self.preview_valid = False
        self.view_note.setText("正在读取该条变化附近的前后期影像…")
        self._request("preview", id=self.features[index]["id"], context=self.context.currentData())

    def _mark(self, state):
        index = self.items.currentRow()
        if self.waiting or self.record_unsaved or index < 0 or (state != "pending" and not self.preview_valid):
            return
        old = self.features[index]["status"]
        if state != old:
            self.undo_stack.append((self.pair_index, index, old))
            self._set_status(index, state)
            self._persist(lambda: self._after_mark(index, state))
            return
        self._after_mark(index, state)

    def _after_mark(self, index, state):
        if state == 'pending':
            self._show_item(index)
            self.message.setText(f'第 {index + 1} 条已设为待复核，记录已自动保存。')
        else:
            self._advance(index, f'第 {index + 1} 条{STATUS_TEXT[state]}。')

    def _set_status(self, index, state):
        self.features[index]["status"] = state
        self.items.item(index).setText(self._item_text(index))
        self._style_item(index)
        self._remember_pair()
        self._counts()

    def _undo(self):
        if self.waiting or self.record_unsaved or not self.undo_stack:
            return
        pair, index, state = self.undo_stack.pop()
        if pair != self.pair_index:
            self._pending_undo = (pair, index, state)
            self._auto_advance = False
            self._load_pair(pair)
            return
        self._set_status(index, state)
        self._show_item(index)
        self._persist(lambda: self._return_to_item(index))

    def _return_to_item(self, index):
        self._show_item(index)
        self.message.setText('已撤销上一次判断，记录已自动保存。')
        if not self.preview_valid:
            self.preview_timer.start(0)

    def _step(self, direction):
        self._editing_complete = True
        index = self.items.currentRow() + direction
        while 0 <= index < len(self.features):
            if not self.items.item(index).isHidden():
                self.items.setCurrentRow(index)
                return
            index += direction

    def _next_pending(self):
        if not self.waiting:
            self._advance(self.items.currentRow(), '已暂时跳过，稍后仍可回来确认。')

    def _advance(self, start, feedback):
        for offset in range(1, len(self.features) + 1):
            index = (start + offset) % len(self.features)
            if self.features[index]["status"] == "pending":
                self._show_item(index)
                self.message.setText(feedback + '继续确认下一条。' if index != start else
                                     '只剩这一条待复核，拿不准可以稍后继续。')
                return
        self._advance_pair()

    def _advance_pair(self):
        self._remember_pair()
        candidates = [(self.pair_index + offset) % self.pairs.count()
                      for offset in range(1, self.pairs.count())]
        grid = self.pairs.itemData(self.pair_index, PAIR_GRID_ROLE)
        candidates.sort(key=lambda i: self.pairs.itemData(i, PAIR_GRID_ROLE) != grid)
        for index in candidates:
            draft = self.drafts.get(index)
            if draft is None or 'pending' in draft['decisions'].values():
                self._auto_advance = True
                self._load_pair(index)
                return
        self._auto_advance = False
        self._editing_complete = False
        self.message.setText('所有区域与期次已完成复核，判断记录已保存。可查看汇总或应用复核结果。')
        self._controls()
        self.save_button.setFocus(Qt.FocusReason.OtherFocusReason)

    def _save(self):
        if self.waiting or self.record_unsaved or self._save_queue is not None:
            return
        self._auto_advance = False
        self.preview_timer.stop()
        self._remember_pair()
        self._save_origin = self.pair_index
        changed = [i for i, d in sorted(self.pair_states.items()) if d.get('modified')]
        self._save_queue = [i for i in changed if i != self.pair_index]
        if self.pair_index in changed:
            self._save_queue.append(self.pair_index)
        self._save_next()

    def _save_next(self):
        if self._save_queue:
            index = self._save_queue.pop(0)
            if index != self.pair_index:
                self._load_pair(index)
            else:
                self._save_current()
            return
        self._save_queue = None
        if self._save_origin >= 0 and self.pair_index != self._save_origin:
            self._load_pair(self._save_origin)
        else:
            self._application_feedback = ''
            if not self.preview_valid:
                first = next((i for i, f in enumerate(self.features) if f['status'] == 'pending'), 0)
                self._show_item(first)
        self._controls()

    def _save_current(self):
        self.message.setText("正在应用复核结果，更新正式成果…")
        self._request("save", decisions={f["id"]: f["status"] for f in self.features})

    def _open_output(self):
        if self.saved_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.saved_path).parent)))

    def _overlay(self, visible):
        for view in self.views:
            if view.overlay:
                view.overlay.setVisible(visible)

    def _fit(self):
        self._syncing = True
        for view in self.views:
            view.fit()
        self._syncing = False

    def _sync(self, source):
        if self._syncing:
            return
        self._syncing = True
        try:
            for view in self.views:
                if view is not source:
                    view.setTransform(source.transform())
                    view.centerOn(source.mapToScene(source.viewport().rect().center()))
        finally:
            self._syncing = False

    def closeEvent(self, event):
        if self.service.pending == 'record':
            event.ignore()
            self._record_after = lambda: self.done(QDialog.DialogCode.Rejected)
            self.message.setText('正在保存最后一次判断，完成后自动关闭；正式成果尚未应用。')
            return
        if self.service.pending == "save" or self._save_queue is not None:
            event.ignore()
            prompt = QMessageBox(QMessageBox.Icon.Warning, "正在应用复核结果",
                                 "正在更新正式成果，请等待完成后再关闭。",
                                 QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Abort, self)
            prompt.setInformativeText("判断记录已保存。仍然退出可能中断成果更新；再次进入复核时，会恢复上次完整成果。")
            prompt.button(QMessageBox.StandardButton.Ok).setText("继续等待")
            prompt.button(QMessageBox.StandardButton.Abort).setText("仍然退出")
            prompt.setDefaultButton(QMessageBox.StandardButton.Ok)
            prompt.setEscapeButton(QMessageBox.StandardButton.Ok)
            prompt.finished.connect(self._save_close_choice)
            prompt.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            prompt.open()
            return
        if self.record_unsaved:
            event.ignore()
            self._confirm_close(lambda: self.done(QDialog.DialogCode.Rejected))
            return
        self.done(QDialog.DialogCode.Rejected)
        event.accept()

    def _save_close_choice(self, choice):
        if choice == QMessageBox.StandardButton.Abort:
            # The modal caller disposes the worker on exit. Its publication
            # journal restores an interrupted update when review opens again.
            self._save_queue = None
            self._auto_advance = False
            self.done(QDialog.DialogCode.Rejected)

    def reject(self):
        # Escape and the title-bar close follow the same unsaved-edit guard.
        self.close()
