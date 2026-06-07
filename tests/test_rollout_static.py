import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM_ROLLOUT_PATH = os.path.join(REPO_ROOT, "verl", "workers", "rollout", "vllm_rollout_spmd.py")


class VLLMRolloutStaticTest(unittest.TestCase):
    def _rollout_text(self):
        with open(VLLM_ROLLOUT_PATH, encoding="utf-8") as f:
            return f.read()

    def test_rollout_padding_uses_integer_pad_fallback(self):
        text = self._rollout_text()
        self.assertIn("self.pad_token_id = tokenizer.pad_token_id", text)
        self.assertIn("if self.pad_token_id is None:", text)
        self.assertIn("self.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0", text)
        self.assertIn("VF.pad_2d_list_to_length(\n                response_ids, self.pad_token_id", text)

    def test_main_rollout_checks_vllm_response_count(self):
        text = self._rollout_text()
        self.assertIn("expected_response_count = batch_size * self.sampling_params.n", text)
        self.assertIn("if len(response_ids) != expected_response_count:", text)
        self.assertIn("vLLM returned {len(response_ids)} responses", text)

    def test_counterfactual_continuation_respects_model_window(self):
        text = self._rollout_text()
        self.assertIn("total_max_len = self.config.prompt_length + self.config.response_length", text)
        self.assertIn("max_new_tokens = min(max(int(max_new_tokens), 1), max(total_max_len - 1, 1))", text)
        self.assertIn("max_prefix_len = max(total_max_len - max_new_tokens, 1)", text)
        self.assertIn("prefix_ids_list = [list(prefix_ids)[-max_prefix_len:] for prefix_ids in prefix_ids_list]", text)

    def test_counterfactual_continuation_handles_empty_vllm_outputs(self):
        text = self._rollout_text()
        self.assertIn("for completion in completions:", text)
        self.assertIn("if completion.outputs:", text)
        self.assertIn("continuation_ids.append([])", text)
        self.assertNotIn("completion.outputs[0].token_ids) for completion in completions", text)

    def test_counterfactual_continuation_accepts_multimodal_data(self):
        text = self._rollout_text()
        self.assertIn("multi_modal_data_list: Any = None", text)
        self.assertIn("if multi_modal_data_list is not None and len(multi_modal_data_list) != len(prefix_ids_list):", text)
        self.assertIn('item["multi_modal_data"] = multi_modal_data', text)


if __name__ == "__main__":
    unittest.main()
