import importlib.util
import os
import sys
import tempfile
import unittest


SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "check_csrfaith_ready.py",
)

spec = importlib.util.spec_from_file_location("check_csrfaith_ready", SCRIPT_PATH)
check_csrfaith_ready = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = check_csrfaith_ready
spec.loader.exec_module(check_csrfaith_ready)


class CheckCSRFaithReadyTest(unittest.TestCase):
    def test_check_modules_reports_available_and_missing(self):
        checks = check_csrfaith_ready.check_modules(
            (
                ("sys", "stdlib module"),
                ("definitely_missing_csrfaith_module", "fake module"),
            )
        )

        by_name = {check.name: check for check in checks}
        self.assertTrue(by_name["sys"].available)
        self.assertFalse(by_name["definitely_missing_csrfaith_module"].available)

    def test_check_files_uses_repo_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "scripts"))
            existing_path = os.path.join(tmpdir, "scripts", "config.yaml")
            with open(existing_path, "w", encoding="utf-8") as f:
                f.write("algorithm:\n  enable_csrfaith: false\n")

            checks = check_csrfaith_ready.check_files(
                tmpdir,
                ("scripts/config.yaml", "scripts/missing.sh"),
            )
            by_path = {check.path: check for check in checks}
            self.assertTrue(by_path["scripts/config.yaml"].exists)
            self.assertFalse(by_path["scripts/missing.sh"].exists)

    def test_config_text_check_lists_missing_csr_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "verl", "trainer"))
            with open(os.path.join(tmpdir, "verl", "trainer", "config.py"), "w", encoding="utf-8") as f:
                f.write("enable_csrfaith: bool = False\n")

            checks = check_csrfaith_ready.check_config_text(tmpdir)
            self.assertEqual(checks[0].name, "csr_config_fields")
            self.assertFalse(checks[0].passed)
            self.assertIn("csr_target_max_relations", checks[0].details)

    def test_rollout_token_budget_check_rejects_small_max_num_batched_tokens(self):
        check = check_csrfaith_ready._check_rollout_token_budget(
            "unit",
            """
            data.max_prompt_length=6144
            data.max_response_length=512
            worker.rollout.max_num_batched_tokens=4096
            """,
        )

        self.assertEqual(check.name, "unit_rollout_token_budget")
        self.assertFalse(check.passed)
        self.assertIn("4096", check.details)
        self.assertIn("6656", check.details)

    def test_build_report_has_stable_shape(self):
        report = check_csrfaith_ready.build_report(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        self.assertIn("ready", report)
        self.assertIn("required_modules", report)
        self.assertIn("files", report)
        self.assertIn("text_checks", report)
        self.assertIn("next_step", report)
        text_check_names = {check["name"] for check in report["text_checks"]}
        self.assertIn("csr_train_passes_cli_overrides", text_check_names)
        self.assertIn("csr_smoke_passes_cli_overrides", text_check_names)
        self.assertIn("csr_train_gpu_count_overridable", text_check_names)
        self.assertIn("csr_train_cuda_visible_devices_overridable", text_check_names)
        self.assertIn("csr_smoke_gpu_count_overridable", text_check_names)
        self.assertIn("csr_train_keeps_intermediate_checkpoints", text_check_names)
        self.assertIn("csr_smoke_keeps_checkpoint_smoke_steps", text_check_names)
        self.assertIn("csr_smoke_disables_kl_reference_policy", text_check_names)
        self.assertIn("csr_train_uses_scene_graph_answer_key", text_check_names)
        self.assertIn("csr_smoke_uses_scene_graph_answer_key", text_check_names)
        self.assertIn("csr_train_rollout_token_budget", text_check_names)
        self.assertIn("csr_smoke_rollout_token_budget", text_check_names)


if __name__ == "__main__":
    unittest.main()
