from __future__ import annotations

import json
import inspect
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import user_workflow_gui as gui


class UserGuiInputCommandTests(unittest.TestCase):
    def test_harmless_tiff_warning_is_hidden_but_preserved_in_log_file(self) -> None:
        raw = "cv::TIFF_Warning TIFFReadDirectory: Unknown field with tag 33550\n"
        log_file = io.StringIO()
        self.assertIsNone(gui.write_and_filter_gui_log(raw, log_file))
        self.assertEqual(log_file.getvalue(), raw)
        important = "RuntimeError: model inference failed\n"
        self.assertEqual(gui.write_and_filter_gui_log(important), important.rstrip())

    def test_structured_stage_status_contains_area_period_and_step_number(self) -> None:
        status = gui.structured_task_status({
            "kind": "stage", "grid": "area_03", "period": "2021",
            "stage": "道路面提取", "stage_index": 2, "stage_total": 5,
            "status": "running",
        })
        self.assertEqual(
            status,
            "验证区：area_03\n期次：2021\n当前步骤：道路面提取\n步骤进度：2 / 5",
        )

    def test_cancel_marks_state_without_deleting_completed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw); job = output / "run_a"; job.mkdir()
            product = job / "grids" / "area_03" / "centerline.done"
            product.parent.mkdir(parents=True); product.write_text("kept", encoding="utf-8")
            period_state = job / "grids" / "area_03" / "period_state.json"
            period_state.write_text(json.dumps({"status": "running", "stages": {"centerline": "completed", "width": "running"}}), encoding="utf-8")
            state = {
                "run_id": "run_a", "status": "running", "current_grid": "area_03",
                "current_period": "2021", "current_stage_label": "道路宽度计算",
                "period_state": str(period_state),
            }
            (job / "job_state.json").write_text(json.dumps(state), encoding="utf-8")

            cancelled = gui.mark_task_cancelled(output, "run_a")

            self.assertEqual(cancelled["status"], "cancelled")
            self.assertTrue(product.is_file())
            self.assertEqual(json.loads(period_state.read_text(encoding="utf-8"))["status"], "cancelled")
            self.assertTrue((output / "latest_pipeline.json").is_file())

    def test_unfinished_task_notice_reports_saved_position(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw); job = output / "run_b"; job.mkdir()
            state_path = job / "job_state.json"
            state_path.write_text(json.dumps({
                "run_id": "run_b", "status": "cancelled", "current_grid": "area_03",
                "current_period": "2021", "current_stage_label": "道路宽度计算",
            }, ensure_ascii=False), encoding="utf-8")
            state = gui.unfinished_task_state(output, {"run_id": "run_b", "state": str(state_path)})
            message = gui.unfinished_task_message(state)
            self.assertIn("area_03 · 2021 · 道路宽度计算", message)
            self.assertIn("从未完成步骤继续", message)

    def test_control_metrics_are_limited_to_three_size_tiers(self) -> None:
        self.assertEqual(set(gui.CONTROL_METRICS), {"primary", "regular", "compact"})
        self.assertEqual(len({tuple(value["padding"]) for value in gui.CONTROL_METRICS.values()}), 3)
        self.assertGreater(
            sum(gui.CONTROL_METRICS["primary"]["padding"]),
            sum(gui.CONTROL_METRICS["regular"]["padding"]),
        )
        self.assertGreater(
            sum(gui.CONTROL_METRICS["regular"]["padding"]),
            sum(gui.CONTROL_METRICS["compact"]["padding"]),
        )

    def test_semantic_button_styles_reuse_shared_size_metrics(self) -> None:
        source = inspect.getsource(gui.UserApp._style)
        for style_name, tier in (
            ("Hero.TButton", "primary"),
            ("Primary.TButton", "primary"),
            ("Secondary.TButton", "regular"),
            ("Compact.TButton", "compact"),
            ("Quiet.TButton", "compact"),
            ("Danger.TButton", "compact"),
        ):
            self.assertIn(
                f'style.configure("{style_name}", **CONTROL_METRICS["{tier}"]',
                source,
            )

    def test_result_buttons_share_one_visual_height(self) -> None:
        source = inspect.getsource(gui.UserApp._style)
        self.assertIn('style.configure("ResultPrimary.TButton", **CONTROL_METRICS["regular"]', source)
        self.assertIn('style.configure("ResultSecondary.TButton", **CONTROL_METRICS["regular"]', source)

    def test_display_percentage_does_not_change_stored_metrics(self) -> None:
        self.assertEqual(gui.format_percentage(0.788), "78.8%")
        self.assertEqual(gui.format_percentage("0.999"), "99.9%")
        self.assertEqual(gui.format_percentage(None), "--")

    def test_result_copy_is_user_facing_and_keeps_internal_codes_out_of_intro(self) -> None:
        source = inspect.getsource(gui.UserApp._build_result_page)
        self.assertIn("根据真值数据中的变化类型", source)
        self.assertNotIn("BHBM=2", source)
        self.assertIn('text="待评价结果"', source)
        self.assertIn('text="项目真值"', source)
        self.assertIn('text="变化类型字段"', source)
        self.assertIn('text="中心线匹配容差（米）"', source)
        self.assertIn("self.result_review_count", source)

    def test_layout_metrics_cover_page_card_form_and_wrap_spacing(self) -> None:
        self.assertEqual(
            set(gui.LAYOUT_METRICS),
            {"page_padding", "card_padding", "section_gap", "module_gap", "form_gap", "form_label_width", "content_wrap"},
        )
        self.assertGreaterEqual(min(gui.LAYOUT_METRICS["page_padding"]), 8)
        self.assertGreaterEqual(min(gui.LAYOUT_METRICS["card_padding"]), 8)
        self.assertGreaterEqual(gui.LAYOUT_METRICS["form_gap"], 3)

    def test_project_path_row_reserves_flexible_space_for_long_paths(self) -> None:
        source = inspect.getsource(gui.UserApp._build_data_page)
        self.assertIn("project_meta.grid_columnconfigure(1, weight=1)", source)
        self.assertIn('textvariable=self.project_path_display, width=1', source)
        self.assertIn('text="连接数据源"', source)
        self.assertEqual(source.count('text="重新扫描"'), 1)

    def test_result_metrics_are_compact_inline_labels(self) -> None:
        source = inspect.getsource(gui.UserApp._build_result_page)
        self.assertIn('(\"可人工编辑：\", self.result_review_count)', source)
        self.assertIn('ttk.Label(metrics, textvariable=variable).pack(side=LEFT', source)
        self.assertNotIn('style="MetricValue.TLabel"', source)

    def test_workflow_pages_use_native_label_frames_without_card_layouts(self) -> None:
        for builder in (
            gui.UserApp._build_data_page,
            gui.UserApp._build_run_page,
            gui.UserApp._build_review_page,
            gui.UserApp._build_result_page,
        ):
            source = inspect.getsource(builder)
            self.assertIn("ttk.LabelFrame", source)
            self.assertNotIn('style="Card.TFrame"', source)
            self.assertNotIn('style="Soft.TFrame"', source)

    def test_main_window_has_fixed_sidebar_log_and_plain_step_labels(self) -> None:
        build_source = inspect.getsource(gui.UserApp._build)
        stepper_source = inspect.getsource(gui.UserApp._draw_stepper)
        self.assertIn('self.content_shell.grid_columnconfigure(0, weight=55', build_source)
        self.assertIn('self.content_shell.grid_columnconfigure(1, weight=45', build_source)
        self.assertIn('self.shared_log_shell.grid(row=1, column=0, sticky="nsew")', inspect.getsource(gui.UserApp._build_shared_log_panel))
        self.assertNotIn("ROAD CHANGE", build_source)
        self.assertIn('text=f"{index + 1}. {label}"', stepper_source)

    def test_gui_source_contains_no_decorative_unicode_controls(self) -> None:
        gui_root = Path(gui.__file__).resolve().parent / "gui"
        source = "\n".join(path.read_text(encoding="utf-8") for path in gui_root.glob("*.py"))
        for character in "▣▤⚙＋●○✓›▶▼⌄":
            self.assertNotIn(character, source)

    def test_unicode_paths_and_config_text_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="SAMRoad_unicode_") as raw:
            root = Path(raw) / "验证区_𠀀"
            root.mkdir()
            area = root / "范围_武汉.shp"
            area.touch()
            periods = []
            for year in ("2021", "2022"):
                source = root / f"影像 清单_{year}_道路.txt"
                source.write_text(f"C:/数据/遥感影像_{year}_𠀀.tif\n", encoding="utf-8-sig")
                periods.append((year, str(source)))
            config = root / "任务 配置_中文.json"
            payload = {"项目": "道路变化", "路径": str(root), "字符": "𠀀✓→"}
            config.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(json.loads(config.read_text(encoding="utf-8")), payload)
            command = gui.build_pipeline_command(
                mode="validation", validation_area=str(area), periods=periods,
                truths=[], evaluate=False, **self.common(root),
            )
            self.assertIn(str(area), command)
            for _period, source in periods:
                self.assertIn(source, command)

    def test_optional_manual_review_is_a_generation_step_before_results(self) -> None:
        self.assertEqual(
            gui.WORKFLOW_STEPS,
            ("数据准备", "自动处理", "人工编辑（可选）", "成果与评价"),
        )

    def test_project_config_is_atomic_and_separate_from_external_sources(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            project = Path(raw) / "project"
            source = Path(raw) / "external-source"
            project.mkdir(); source.mkdir()
            payload = {"project_root": str(project), "external_data_sources": [str(source)]}
            path = gui.atomic_write_json(gui.project_config_path(project), payload)
            self.assertEqual(path.name, "project_config.json")
            self.assertEqual(gui.read_project_config(project), payload)
            self.assertNotEqual(path.parent, source)
            self.assertEqual(list(project.glob(".*.tmp")), [])

    def test_selected_period_reports_both_adjacent_dependencies(self) -> None:
        self.assertEqual(
            gui.affected_change_pairs(["2024", "2021", "2022"], "2022"),
            [("2021", "2022"), ("2022", "2024")],
        )

    def test_cancelled_active_task_is_automatically_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw); state = output / "run_cancelled" / "job_state.json"
            state.parent.mkdir(); state.write_text('{"status":"cancelled"}', encoding="utf-8")
            run_id, resume, resolved = gui.resolve_automatic_run(
                output, active_task={"run_id": "run_cancelled"}, generated_run_id="new_run",
            )
            self.assertEqual(run_id, "run_cancelled")
            self.assertTrue(resume)
            self.assertEqual(resolved, state)

    def test_data_check_and_runtime_preflight_are_separate_commands(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); area = root / "area.shp"; area.touch()
            periods = []
            for year in ("2021", "2022"):
                source = root / f"{year}.txt"; source.touch(); periods.append((year, str(source)))
            common = dict(mode="validation", validation_area=str(area), periods=periods, truths=[], evaluate=False, **self.common(root))
            data_check = gui.build_pipeline_command(data_check_only=True, **common)
            full = gui.build_pipeline_command(**common)
            self.assertIn("--data-check-only", data_check)
            self.assertNotIn("--runtime-preflight", data_check)
            self.assertIn("--runtime-preflight", full)

    def test_result_browser_represents_missing_products_without_opening_them(self) -> None:
        items = gui.collect_result_tree_items({
            "period_results": [{"grid": "north", "period": "2022", "centerlines": "missing.shp"}],
            "change_results": [],
        })
        centerline = next(item for item in items if item["label"] == "道路中心线")
        self.assertEqual(centerline["status"], "未生成")
        self.assertIn("长时序道路成果", {item["label"] for item in items})

    def test_accuracy_evaluation_is_a_runnable_result_step(self) -> None:
        data_source = inspect.getsource(gui.UserApp._build_data_page)
        result_source = inspect.getsource(gui.UserApp._build_result_page)
        self.assertNotIn('text="精度评价"', data_source)
        self.assertIn('text="精度评价"', result_source)
        self.assertIn('command=self.run_result_evaluation', result_source)

    def test_builds_evaluation_command_for_existing_change_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth = root / "truth.shp"
            truth.touch()
            command = gui.build_evaluate_existing_command(
                {"grid": "validation", "before_period": "2021", "after_period": "2022"},
                root / "pipeline_result.json", str(truth), truth_type_field="change_typ",
                validation_area=str(root / "area.shp"), evaluation_tolerance="5.0",
            )
            self.assertEqual(command[0], "evaluate-existing")
            self.assertIn("--pipeline-manifest", command)
            self.assertIn("--truth-type-field", command)

    def test_entering_run_step_does_not_auto_start_preflight(self) -> None:
        source = inspect.getsource(gui.UserApp._show_step)
        self.assertNotIn("preflight_inputs", source)

    def test_manual_review_step_is_not_blocked_without_results(self) -> None:
        app = object.__new__(gui.UserApp)
        app.step_pages = [mock.Mock() for _ in range(4)]
        app.process = None
        app.current_step = 0
        app.results_available = False
        app.preflight_passed = False
        app.status = mock.Mock()
        app.root = mock.Mock()
        app.footer_back = mock.Mock()
        app.footer_next = mock.Mock()
        app._populate_review_step = mock.Mock()

        app._show_step(2)

        self.assertEqual(app.current_step, 2)
        app.step_pages[2].pack.assert_called_once()
        app._populate_review_step.assert_called_once()

    def test_shared_log_widget_supports_selection_copy_and_file_opening(self) -> None:
        source = inspect.getsource(gui.UserApp._build_shared_log_panel)
        build_source = inspect.getsource(gui.UserApp._build)
        self.assertIn("Text(", source)
        self.assertIn("全流程日志", source)
        self.assertIn("复制全部", source)
        self.assertIn("打开日志文件", source)
        self.assertIn("self._build_shared_log_panel()", build_source)
        self.assertNotIn("self.log = Text(", inspect.getsource(gui.UserApp._build_run_page))

    def test_backend_command_is_an_instance_method(self) -> None:
        descriptor = inspect.getattr_static(gui.UserApp, "_command")
        self.assertFalse(isinstance(descriptor, staticmethod))
        self.assertEqual(list(inspect.signature(descriptor).parameters)[:2], ["self", "args"])

    def test_review_page_uses_shared_log_and_keeps_progress_and_stop_controls(self) -> None:
        source = inspect.getsource(gui.UserApp._build_review_page)
        self.assertIn("编辑后增量重建", source)
        self.assertNotIn("self.review_log", source)
        self.assertIn("全流程日志", source)
        self.assertIn("self.review_progress = ttk.Progressbar(", source)
        self.assertIn('text="停止重建"', source)
        self.assertIn("command=self.cancel_task", source)

    def test_apply_edits_writes_a_dedicated_log_file(self) -> None:
        source = inspect.getsource(gui.UserApp._command)
        self.assertIn('args[0] == "apply-edits"', source)
        self.assertIn('人工编辑重建_', source)

    @staticmethod
    def common(root: Path) -> dict[str, str]:
        return {
            "output_root": str(root / "out"), "checkpoint": "model.ckpt", "config": "config.yaml",
            "device": "cpu", "pixel_size": "0", "rescale": "off",
            "absolute": "1", "ratio": "0.1", "tolerance": "3",
        }

    def test_default_validation_command_naturally_orders_periods_and_truths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area = root / "area.shp"; area.touch()
            images = []
            for name in ("20210", "2022", "2021"):
                path = root / f"{name}.txt"; path.touch(); images.append((name, str(path)))
            truth_12 = root / "truth_2021_2022.shp"; truth_12.touch()
            truth_2_10 = root / "truth_2022_20210.shp"; truth_2_10.touch()
            command = gui.build_pipeline_command(
                mode="validation", validation_area=str(area), periods=images,
                truths=[("2022", "20210", str(truth_2_10)), ("2021", "2022", str(truth_12))],
                truth_type_field="change_kind", **self.common(root),
            )
            period_args = [command[index + 1:index + 3] for index, value in enumerate(command) if value == "--period"]
            truth_args = [command[index + 1:index + 4] for index, value in enumerate(command) if value == "--truth"]
            self.assertEqual([entry[0] for entry in period_args], ["2021", "2022", "20210"])
            self.assertEqual([entry[:2] for entry in truth_args], [["2021", "2022"], ["2022", "20210"]])
            self.assertIn("--validation-area", command)
            self.assertIn("--truth-type-field", command)

    def test_validation_command_rejects_missing_adjacent_truth(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area = root / "area.shp"; area.touch()
            first = root / "one.txt"; first.touch()
            second = root / "two.txt"; second.touch()
            with self.assertRaisesRegex(ValueError, "每个相邻期次"):
                gui.build_pipeline_command(
                    mode="validation", validation_area=str(area),
                    periods=[("2021", str(first)), ("2022", str(second))], truths=[],
                    **self.common(root),
                )

    def test_production_command_allows_validation_without_truth(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area = root / "area.shp"; area.touch()
            first = root / "one.txt"; first.touch()
            second = root / "two.txt"; second.touch()
            command = gui.build_pipeline_command(
                mode="validation", validation_area=str(area),
                periods=[("2021", str(first)), ("2022", str(second))], truths=[],
                evaluate=False, **self.common(root),
            )
            self.assertIn("--no-evaluation", command)
            self.assertNotIn("--truth", command)

    def test_resume_requires_named_task(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); grid_root = root / "grids"; grid_root.mkdir()
            with self.assertRaisesRegex(ValueError, "原任务名称"):
                gui.build_pipeline_command(
                    mode="grid", source_root=str(grid_root), resume=True,
                    **self.common(root),
                )

    def test_documented_project_folder_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area_dir = root / "01_验证区"; area_dir.mkdir()
            imagery_dir = root / "02_影像"; imagery_dir.mkdir()
            truth_dir = root / "03_变化真值"; truth_dir.mkdir()
            (area_dir / "area.shp").touch()
            for period in ("2021", "2022"):
                (imagery_dir / f"{period}.txt").touch()
            (truth_dir / "2021_to_2022.shp").touch()
            project = gui.discover_validation_project(root)
            self.assertEqual([period for period, _source in project["periods"]], ["2021", "2022"])
            self.assertEqual(project["truths"][("2021", "2022")], str(truth_dir / "2021_to_2022.shp"))
            self.assertEqual(Path(project["output_root"]), root / "04_成果输出")

    def test_nested_area_folders_keep_independent_periods_and_truths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            expected = {"north": ("2021", "2022"), "south": ("2020", "2023", "2025")}
            for area, periods in expected.items():
                area_root = root / area
                boundary = area_root / "01_验证区"; boundary.mkdir(parents=True)
                imagery = area_root / "02_影像"; imagery.mkdir()
                truths = area_root / "03_变化真值"; truths.mkdir()
                (boundary / "boundary.shp").touch()
                for period in periods:
                    (imagery / f"{period}.txt").touch()
                for before, after in zip(periods, periods[1:]):
                    (truths / f"{before}_to_{after}.shp").touch()
            project = gui.discover_validation_project(root)
            self.assertEqual(set(project["area_periods"]), set(expected))
            self.assertEqual(
                {area: tuple(period for period, _source in rows) for area, rows in project["area_periods"].items()},
                expected,
            )
            self.assertEqual(len(project["area_truths"]), 3)
            command = gui.build_pipeline_command(
                mode="validation", validation_area="", validation_areas=project["validation_areas"],
                periods=project["periods"], area_periods=project["area_periods"],
                area_truths=[(*key, value) for key, value in project["area_truths"].items()],
                evaluate=False, **self.common(root),
            )
            self.assertEqual(command.count("--validation-area"), 2)
            self.assertEqual(command.count("--period"), 5)
            self.assertEqual(command.count("--truth"), 3)

    def test_multi_area_command_expands_every_area_and_period(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            areas = []
            for name in ("north", "south"):
                path = root / f"{name}.shp"; path.touch(); areas.append((name, str(path)))
            periods = []
            for period in ("2021", "2022", "2023"):
                path = root / f"{period}.txt"; path.touch(); periods.append((period, str(path)))
            command = gui.build_pipeline_command(
                mode="validation", validation_area="", validation_areas=areas,
                periods=periods, evaluate=False, **self.common(root),
            )
            self.assertEqual(command.count("--validation-area"), 2)
            self.assertEqual(command.count("--period"), 6)
            self.assertIn(["--validation-area", "north", str(root / "north.shp")], [command[i:i + 3] for i in range(len(command) - 2)])

    def test_production_command_keeps_multi_area_truths_for_result_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area = root / "north.shp"; area.touch()
            truth = root / "truth.shp"; truth.touch()
            periods = []
            for period in ("2021", "2022"):
                source = root / f"{period}.txt"; source.touch(); periods.append((period, str(source)))
            command = gui.build_pipeline_command(
                mode="validation", validation_area="", validation_areas=[("north", str(area))],
                periods=periods, area_truths=[("north", "2021", "2022", str(truth))],
                evaluate=False, truth_type_field="BHBM", **self.common(root),
            )
            self.assertIn("--no-evaluation", command)
            self.assertIn(["--truth", "north", "2021", "2022", str(truth)], [command[i:i + 5] for i in range(len(command) - 4)])

    def test_result_page_contains_no_image_preview_decoder(self) -> None:
        source = inspect.getsource(gui.UserApp)
        self.assertNotIn("open_preview_window", source)
        self.assertNotIn("_refresh_result_thumbnails", source)
        self.assertNotIn("PILImage.open", source)

    def test_validation_command_rejects_non_shp_or_non_txt_user_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area = root / "area.geojson"; area.touch()
            first = root / "one.tif"; first.touch()
            second = root / "two.txt"; second.touch()
            with self.assertRaisesRegex(ValueError, "验证区必须使用 SHP"):
                gui.build_pipeline_command(
                    mode="validation", validation_area=str(area),
                    periods=[("2021", str(first)), ("2022", str(second))],
                    evaluate=False, **self.common(root),
                )
            area = root / "area.shp"; area.touch()
            with self.assertRaisesRegex(ValueError, "影像必须使用内含影像路径的 TXT"):
                gui.build_pipeline_command(
                    mode="validation", validation_area=str(area),
                    periods=[("2021", str(first)), ("2022", str(second))],
                    evaluate=False, **self.common(root),
                )

    def test_grid_mode_is_an_explicit_backup_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); grid_root = root / "grids"; grid_root.mkdir()
            command = gui.build_pipeline_command(mode="grid", source_root=str(grid_root), **self.common(root))
            self.assertEqual(command[:5], ["all", "--mode", "grid", "--source-root", str(grid_root)])
            self.assertNotIn("--validation-area", command)

    def test_apply_edits_command_keeps_manifest_and_reports_change_reruns(self) -> None:
        command = gui.build_apply_edits_command(
            {"result": "period.json", "edited_directory": "edited"}, "latest_pipeline.json",
        )
        self.assertEqual(command[-2:], ["--pipeline-manifest", "latest_pipeline.json"])
        message = gui.UserApp._friendly({
            "kind": "complete", "stage": "apply-edits", "change_rerun_count": 2,
        })
        self.assertIn("重跑 2 个", message)


class UserGuiArtifactTests(unittest.TestCase):
    def test_collects_fusion_counts_and_centerline_edit_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            preview = root / "fusion.png"
            preview.touch()
            review = root / "review"
            review.mkdir()
            edited = root / "edited"
            result = root / "latest_result.json"
            result.touch()
            final_centerlines = root / "road_centerlines.shp"
            final_centerlines.touch()
            manifest = {
                "period_results": [{
                    "grid": "g", "period": "2026", "result": str(result),
                    "centerlines": str(final_centerlines),
                    "previews": {"fusion": str(preview)},
                    "fusion": {
                        "original_edge_count": 10, "optimized_edge_count": 13,
                        "auto_gap_count": 1, "auto_surface_count": 2,
                        "geometry_edited_tile_count": 0,
                    },
                    "review": {
                        "available": True, "directory": str(review),
                        "edited_directory": str(edited), "manual_item_count": 3,
                    },
                }],
                "change_results": [],
            }
            previews = gui.collect_preview_items(manifest, root)
            reviews = gui.collect_review_items(manifest, root)
            self.assertEqual(len(previews), 1)
            self.assertIn("道路面骨架新增 2 条", previews[0]["detail"])
            self.assertEqual(reviews[0]["edited_directory"], str(edited.resolve()))
            self.assertEqual(reviews[0]["result"], str(result.resolve()))
            self.assertEqual(reviews[0]["final_centerlines"], str(final_centerlines.resolve()))

    def test_result_preview_omits_intermediate_extraction_images(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = {}
            for key in ("centerline", "surface", "fusion", "width"):
                paths[key] = root / f"{key}.png"
                paths[key].touch()
            items = gui.collect_preview_items({
                "period_results": [{"grid": "g", "period": "2026", "previews": {key: str(value) for key, value in paths.items()}}],
                "change_results": [],
            }, root)
            self.assertEqual([item["category"] for item in items], ["融合", "重新测宽"])

    def test_change_and_review_previews_are_separate_gui_items(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            formal = root / "change_preview.png"
            review = root / "review_preview.png"
            formal.touch()
            review.touch()
            manifest = {"period_results": [], "change_results": [{
                "grid": "g", "before_period": "2021", "after_period": "2022",
                "previews": {"change": str(formal), "review_change": str(review)},
            }]}

            items = gui.collect_preview_items(manifest, root)
            tree = gui.collect_result_tree_items(manifest, root)

            self.assertEqual(
                [item["category"] for item in items],
                ["最终变化结果", "待复核变化"],
            )
            preview_nodes = {
                item["label"]: item for item in tree
                if item["label"] in {"最终变化结果", "待复核变化"}
            }
            self.assertEqual(preview_nodes["最终变化结果"]["path"], str(formal.resolve()))
            self.assertEqual(preview_nodes["待复核变化"]["path"], str(review.resolve()))

    def test_collects_temporal_life_shp_for_attribute_table(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            life = root / "road_life.shp"
            life.touch()
            items = gui.collect_temporal_items({"temporal_results": [{
                "grid": "g", "life_shp": str(life), "period_count": 8, "road_count": 12,
            }]}, root)
            self.assertEqual(items[0]["path"], str(life.resolve()))
            self.assertEqual(items[0]["period_count"], "8")

    def test_validation_manifest_uses_user_facing_scope_label(self) -> None:
        self.assertEqual(gui._display_scope({"grid": "validation", "period": "2026"}), ("验证区项目", "2026"))

    def test_root_geometry_editor_is_available_as_packaged_fallback(self) -> None:
        app = object.__new__(gui.UserApp)
        script = app._geometry_editor_script()
        self.assertIsNotNone(script)
        self.assertEqual(script.name, "geometry_editor.py")

    def test_packaged_geometry_editor_is_self_contained(self) -> None:
        app = object.__new__(gui.UserApp)
        script = app._geometry_editor_script()
        self.assertIsNotNone(script)
        source = script.read_text(encoding="utf-8")
        self.assertNotIn("runpy", source)
        self.assertNotIn("sam_width_experiment", source)
        self.assertIn("class GeometryEditorApp", source)

    def test_geometry_editor_preflight_and_command_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            review = root / "review"
            review.mkdir()
            image = root / "2022.tif"
            graph = root / "2022_prepared_graph.p"
            image.touch()
            graph.touch()
            (review / "2022_summary.json").write_text(
                json.dumps({"image": str(image), "prepared_graph": str(graph)}),
                encoding="utf-8",
            )
            item = {
                "directory": str(review),
                "edited_directory": str(root / "centerline_edit"),
                "final_centerlines": str(root / "road_centerlines.shp"),
            }
            Path(item["final_centerlines"]).touch()
            inputs = gui.collect_geometry_editor_inputs(review)
            self.assertEqual(len(inputs), 1)
            self.assertTrue(inputs[0]["image_exists"])
            self.assertTrue(inputs[0]["prepared_graph_exists"])
            self.assertEqual(gui.geometry_editor_diagnostics(review), [])
            ready_file = root / "editor_ready.json"
            command = gui.build_geometry_editor_command(
                root / "geometry_editor.py", item, ready_file,
            )
            self.assertEqual(Path(command[command.index("--review-dir") + 1]), review.resolve())
            self.assertEqual(
                Path(command[command.index("--edited-dir") + 1]),
                (root / "centerline_edit").resolve(),
            )
            self.assertEqual(
                Path(command[command.index("--final-centerlines") + 1]),
                Path(item["final_centerlines"]).resolve(),
            )
            self.assertEqual(
                Path(command[command.index("--ready-file") + 1]), ready_file.resolve(),
            )

    def test_geometry_editor_process_stays_starting_until_ready_file_exists(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as raw:
            ready_file = Path(raw) / "ready.json"
            state, _detail = gui.geometry_editor_process_state(
                process, ready_file, started_monotonic=10.0, now=20.0,
            )
            self.assertEqual(state, "starting")

            ready_file.write_text(
                json.dumps({"status": "ready", "pid": 123}), encoding="utf-8",
            )
            state, detail = gui.geometry_editor_process_state(
                process, ready_file, started_monotonic=10.0, now=20.0,
            )
            self.assertEqual(state, "ready")
            self.assertEqual(detail["pid"], 123)

    def test_geometry_editor_process_reports_exit_before_ready(self) -> None:
        process = mock.Mock()
        process.poll.return_value = 7
        state, detail = gui.geometry_editor_process_state(
            process, None, started_monotonic=10.0, now=11.0,
        )
        self.assertEqual(state, "failed")
        self.assertEqual(detail["returncode"], 7)

    def test_editor_poll_sets_opened_only_after_ready_signal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ready_file = Path(raw) / "ready.json"
            ready_file.write_text(
                json.dumps({"status": "ready", "pid": 123}), encoding="utf-8",
            )
            app = object.__new__(gui.UserApp)
            app.editor_process = mock.Mock(pid=123)
            app.editor_process.poll.return_value = None
            app.editor_ready_file = ready_file
            app.editor_started_monotonic = time.monotonic()
            app.editor_launch_state = "starting"
            app.editor_timeout_reported = False
            app.editor_stdout_lines = []
            app.editor_stderr_lines = []
            app.status = mock.Mock()
            app.review_status = mock.Mock()
            app._append_log = mock.Mock()

            app._poll_geometry_editor()

            self.assertEqual(app.editor_launch_state, "ready")
            app.status.set.assert_called_with(
                "人工编辑器已打开。完成编辑并保存后，可返回此处更新相关结果。"
            )
            self.assertFalse(ready_file.exists())

    def test_editor_poll_reports_early_exit_and_keeps_stderr(self) -> None:
        app = object.__new__(gui.UserApp)
        app.editor_process = mock.Mock(pid=999)
        app.editor_process.poll.return_value = 9
        app.editor_ready_file = None
        app.editor_started_monotonic = time.monotonic()
        app.editor_launch_state = "starting"
        app.editor_timeout_reported = False
        app.editor_stdout_lines = []
        app.editor_stderr_lines = ["Traceback line", "RuntimeError: boom"]
        app.status = mock.Mock()
        app.review_status = mock.Mock()
        app._append_log = mock.Mock()
        app._show_log = mock.Mock()
        app.root = mock.Mock()
        app.review_items = [{}]
        app.launch_review_button = mock.Mock()

        with mock.patch.object(gui.messagebox, "showerror") as showerror:
            app._poll_geometry_editor()

        self.assertEqual(app.editor_launch_state, "idle")
        error_message = showerror.call_args.args[1]
        self.assertIn("退出码：9", error_message)
        self.assertIn("RuntimeError: boom", error_message)

    def test_launching_editor_reports_starting_not_opened_after_popen(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            review = root / "review"
            review.mkdir()
            image = root / "tile.tif"
            graph = root / "tile_graph.p"
            centerlines = root / "road_centerlines.shp"
            surfaces = root / "road_surfaces.shp"
            script = root / "geometry_editor.py"
            for path in (image, graph, centerlines, surfaces, script):
                path.touch()
            (review / "tile_summary.json").write_text(
                json.dumps({"image": str(image), "prepared_graph": str(graph)}),
                encoding="utf-8",
            )
            item = {
                "directory": str(review),
                "edited_directory": str(root / "edited"),
                "final_centerlines": str(centerlines),
                "final_surfaces": str(surfaces),
            }
            app = object.__new__(gui.UserApp)
            app.root = mock.Mock()
            app.editor_process = None
            app.review_items = [item]
            app.review_edit_directory = mock.Mock()
            app.launch_review_button = mock.Mock()
            app.status = mock.Mock()
            app.review_status = mock.Mock()
            app._append_log = mock.Mock()
            app._selected_review_item = lambda: item
            app._geometry_editor_script = lambda: script
            fake_process = mock.Mock()
            fake_process.stdout = io.StringIO("")
            fake_process.stderr = io.StringIO("")
            fake_process.pid = 321
            fake_process.poll.return_value = None

            with mock.patch.object(gui.subprocess, "Popen", return_value=fake_process):
                app.launch_selected_review_editor()

            self.assertEqual(app.editor_launch_state, "starting")
            app.status.set.assert_called_with("正在启动人工编辑器，请稍候…")
            self.assertNotIn(
                "已打开",
                " ".join(str(call) for call in app.status.set.call_args_list),
            )
            app._clear_geometry_editor_state()

    def test_actual_run_2022_geometry_editor_paths_are_resolved(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "project" / "04_成果输出" / "latest_pipeline.json"
        )
        if not manifest_path.is_file():
            self.skipTest(f"actual run manifest is unavailable: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = gui.collect_review_items(manifest, manifest_path.parent)
        target = next((item for item in items if item.get("scope") == "2022"), None)
        if target is None:
            self.skipTest("actual run has no available 2022 review item")
        review = Path(target["directory"])
        inputs = gui.collect_geometry_editor_inputs(review)
        tile = next(
            (
                entry for entry in inputs
                if entry.get("image_exists") and entry.get("prepared_graph_exists")
            ),
            None,
        )
        self.assertIsNotNone(tile)
        self.assertTrue(tile["image_exists"], tile)
        self.assertTrue(tile["prepared_graph_exists"], tile)
        self.assertTrue(Path(target["result"]).is_file())
        command = gui.build_geometry_editor_command(gui.ROOT.parent / "sam_width_experiment" / "geometry_editor.py", target)
        self.assertEqual(Path(command[command.index("--review-dir") + 1]), review.resolve())
        self.assertEqual(
            Path(command[command.index("--edited-dir") + 1]),
            Path(target["edited_directory"]).resolve(),
        )


if __name__ == "__main__":
    unittest.main()
