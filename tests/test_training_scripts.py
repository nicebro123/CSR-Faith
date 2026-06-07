import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _trainer_scripts():
    paths = []
    for root, _, names in os.walk(SCRIPTS_DIR):
        for name in sorted(names):
            if not name.endswith(".sh"):
                continue
            path = os.path.join(root, name)
            text = _read(path)
            if "python3 -m verl.trainer.main" in text:
                paths.append(path)
    return paths


class TrainingScriptTest(unittest.TestCase):
    def test_all_trainer_scripts_forward_cli_overrides(self):
        for path in _trainer_scripts():
            with self.subTest(script=os.path.basename(path)):
                self.assertIn('"$@"', _read(path))

    def test_all_trainer_scripts_allow_gpu_count_override(self):
        for path in _trainer_scripts():
            text = _read(path)
            with self.subTest(script=os.path.basename(path)):
                self.assertIn("N_GPUS=", text)
                self.assertIn("trainer.n_gpus_per_node=${N_GPUS}", text)

    def test_csr_script_cuda_visible_devices_is_overridable(self):
        text = _read(os.path.join(SCRIPTS_DIR, "csrfaith_7b_grpo.sh"))
        self.assertIn("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}", text)
        self.assertIn("N_GPUS=${N_GPUS:-2}", text)

    def test_csr_scripts_keep_expected_checkpoints(self):
        train_text = _read(os.path.join(SCRIPTS_DIR, "csrfaith_7b_grpo.sh"))
        smoke_text = _read(os.path.join(SCRIPTS_DIR, "csrfaith_smoke.sh"))
        self.assertIn("trainer.save_freq=25", train_text)
        self.assertIn("trainer.save_limit=3", train_text)
        self.assertIn("trainer.save_limit=3", smoke_text)

    def test_csr_smoke_rollout_token_budget_matches_lengths(self):
        smoke_text = _read(os.path.join(SCRIPTS_DIR, "csrfaith_smoke.sh"))
        self.assertIn("data.max_prompt_length=6144", smoke_text)
        self.assertIn("data.max_response_length=512", smoke_text)
        self.assertIn("worker.rollout.max_num_batched_tokens=6656", smoke_text)

    def test_csr_smoke_disables_kl_reference_policy(self):
        smoke_text = _read(os.path.join(SCRIPTS_DIR, "csrfaith_smoke.sh"))
        self.assertIn("algorithm.disable_kl=True", smoke_text)
        self.assertIn("algorithm.use_kl_loss=False", smoke_text)

    def test_spatial_sgg_scripts_use_scene_graph_answer_key(self):
        for path in _trainer_scripts():
            text = _read(path)
            if "worker.reward.score_function=spatial_sgg" not in text:
                continue
            with self.subTest(script=os.path.relpath(path, REPO_ROOT)):
                self.assertIn('data.answer_key="answer"', text)
                self.assertNotIn('data.answer_key="answer_option_text"', text)

    def test_run_guide_matches_current_csr_checkpoint_defaults(self):
        guide = _read(os.path.join(REPO_ROOT, "docs", "csrfaith_run_guide.md"))
        self.assertIn("trainer.save_freq=25", guide)
        self.assertIn("trainer.save_limit=3", guide)
        self.assertNotIn("中间 checkpoint 默认不存", guide)

    def test_logger_list_overrides_are_shell_quoted(self):
        for path in _trainer_scripts():
            text = _read(path)
            with self.subTest(script=os.path.relpath(path, REPO_ROOT)):
                self.assertNotIn("trainer.logger=[", text)


if __name__ == "__main__":
    unittest.main()
