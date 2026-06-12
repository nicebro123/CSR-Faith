import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAY_TRAINER_PATH = os.path.join(REPO_ROOT, "verl", "trainer", "ray_trainer.py")
CONFIG_PATH = os.path.join(REPO_ROOT, "verl", "trainer", "config.py")


class RayTrainerStaticTest(unittest.TestCase):
    def _trainer_text(self):
        with open(RAY_TRAINER_PATH, encoding="utf-8") as f:
            return f.read()

    def test_final_validation_obeys_val_freq_gate(self):
        text = self._trainer_text()
        self.assertIn("if self.val_reward_fn is not None and self.config.trainer.val_freq > 0:", text)
        self.assertNotIn("# perform validation after training\n        if self.val_reward_fn is not None:", text)

    def test_training_loop_checks_max_steps_before_incrementing_global_step(self):
        text = self._trainer_text()
        pre_increment = "if self.global_step >= self.training_steps:\n                    break\n                self.global_step += 1"
        outer_break = "self.logger.log(data=metrics, step=self.global_step)\n\n            if self.global_step >= self.training_steps:\n                break"
        self.assertIn(pre_increment, text)
        self.assertIn(outer_break, text)

    def test_advantage_estimator_string_is_normalized_before_branching(self):
        text = self._trainer_text()
        self.assertIn("def _normalize_advantage_estimator(adv_estimator: Any) -> AdvantageEstimator:", text)
        self.assertIn("self.adv_estimator = _normalize_advantage_estimator(config.algorithm.adv_estimator)", text)
        self.assertIn("if self.adv_estimator == AdvantageEstimator.GAE:", text)
        self.assertIn("if self.adv_estimator == AdvantageEstimator.REMAX:", text)
        self.assertIn("adv_estimator=self.adv_estimator,", text)
        self.assertNotIn("config.algorithm.adv_estimator not in list(AdvantageEstimator)", text)
        self.assertNotIn("if self.config.algorithm.adv_estimator == \"remax\":", text)

    def test_validation_dataloader_is_created_only_when_validation_enabled(self):
        text = self._trainer_text()
        self.assertIn("def _validation_enabled(self) -> bool:", text)
        self.assertIn("self.val_dataset = None\n        self.val_dataloader = None", text)
        self.assertIn("if self._validation_enabled():\n            self.val_dataset = RLHFDataset", text)
        self.assertIn("Validation dataloader is disabled.", text)
        self.assertIn("if self.val_dataloader is None:\n            raise RuntimeError", text)
        validation_helper = text.split("def _validation_enabled(self) -> bool:", 1)[1].split("def _create_dataloader", 1)[0]
        self.assertNotIn("val_generations_to_log > 0", validation_helper)

    def test_val_only_triggers_pre_train_validation(self):
        text = self._trainer_text()
        self.assertIn(
            "if self.val_reward_fn is not None and (self.config.trainer.val_before_train or self.config.trainer.val_only):",
            text,
        )

    def test_validation_repeats_inputs_when_rollout_n_is_greater_than_one(self):
        text = self._trainer_text()
        self.assertIn("if len(test_output_gen_batch) != len(test_batch):", text)
        self.assertIn("val_repeat_times = len(test_output_gen_batch) // len(test_batch)", text)
        self.assertIn("test_batch = test_batch.repeat(repeat_times=val_repeat_times, interleave=True)", text)
        self.assertIn("text for text in input_texts for _ in range(val_repeat_times)", text)

    def test_prefix_continuation_handles_one_to_all_worker_output(self):
        text = self._trainer_text()
        self.assertIn("def _select_continuation_output(cf_output_raw: Any, expected_size: int) -> DataProto:", text)
        self.assertIn("if isinstance(cf_output_raw, (list, tuple)):", text)
        self.assertIn(
            "cf_output = self._select_continuation_output(cf_output_raw, expected_size=len(prefix_texts))",
            text,
        )

    def test_prefix_continuation_uses_pad_token_fallback(self):
        text = self._trainer_text()
        self.assertIn("pad_id = self.tokenizer.pad_token_id", text)
        self.assertIn("if pad_id is None:", text)
        self.assertIn("pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0", text)

    def test_prefix_continuation_preserves_multimodal_context(self):
        text = self._trainer_text()
        self.assertIn("multi_modal_data_all=None", text)
        self.assertIn("max_prefix_tokens: Optional[int] = None", text)
        self.assertIn("max_continuation_prefix_tokens = max(", text)
        self.assertIn("max_resp_tokens = self.max_prefix_tokens - len(prompt_token_ids)", text)
        self.assertIn("resp_tokens = resp_tokens[-max_resp_tokens:]", text)
        self.assertIn('non_tensors["multi_modal_data"] = np.array([multi_modal_data for _ in range(n)], dtype=object)', text)
        self.assertIn("continuation_multi_modal_data = gen_batch.non_tensor_batch.get(\"multi_modal_data\")", text)
        self.assertIn('batch.non_tensor_batch["_continuation_multi_modal_data"] = np.repeat(', text)
        self.assertIn('multi_modal_data_all=batch.non_tensor_batch.get("_continuation_multi_modal_data")', text)
        self.assertIn("max_prefix_tokens=max_continuation_prefix_tokens", text)
        self.assertIn('batch.non_tensor_batch.pop("_continuation_multi_modal_data", None)', text)

    def test_causal_spatial_critic_is_optional_and_separate_from_ppo_critic(self):
        text = self._trainer_text()
        self.assertIn("from ..models.causal_spatial_critic import CausalSpatialCritic", text)
        self.assertIn("from ..utils.causal_critic_data import build_critic_examples", text)
        self.assertIn("self.causal_spatial_critic = None", text)
        self.assertIn("CausalSpatialCritic.load(critic_path)", text)
        self.assertIn("causal_critic_use_online_fallback", text)
        self.assertIn("build_critic_examples(", text)
        self.assertIn("self.causal_spatial_critic.score_batch(critic_examples)", text)
        self.assertIn('metrics["csr/causal_signal_is_critic"]', text)
        self.assertIn('metrics["csr/critic_causal_mean"]', text)
        self.assertIn("self.critic_wg", text)

    def test_causal_spatial_critic_config_fields_exist(self):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("enable_causal_spatial_critic: bool = False", text)
        self.assertIn('causal_critic_path: str = ""', text)
        self.assertIn("causal_critic_min_target_confidence: float = 0.0", text)
        self.assertIn("causal_critic_use_online_fallback: bool = True", text)


if __name__ == "__main__":
    unittest.main()
