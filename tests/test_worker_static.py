import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSDP_WORKERS_PATH = os.path.join(REPO_ROOT, "verl", "workers", "fsdp_workers.py")


class FSDPWorkerStaticTest(unittest.TestCase):
    def _worker_text(self):
        with open(FSDP_WORKERS_PATH, encoding="utf-8") as f:
            return f.read()

    def test_model_and_generation_config_use_pad_token_fallback(self):
        text = self._worker_text()
        self.assertIn("def _pad_token_id(self) -> int:", text)
        self.assertIn("pad_token_id=self._pad_token_id(),", text)
        self.assertIn("if self.generation_config.pad_token_id is None:", text)
        self.assertIn("self.generation_config.pad_token_id = self._pad_token_id()", text)
        self.assertIn("else self._pad_token_id(),", text)

    def test_prefix_continuation_padding_handles_missing_pad_token(self):
        text = self._worker_text()
        self.assertIn("if len(continuation_ids_list) != batch_size:", text)
        self.assertIn("continuation_ids_list.extend([[] for _ in range(batch_size - len(continuation_ids_list))])", text)
        self.assertIn("max_requested_tokens = max(int(cf_max_tokens), 1)", text)
        self.assertIn("max_cont_len = max(1, min(max_cont_len, max_requested_tokens))", text)
        self.assertIn("pad_token_id = self._pad_token_id()", text)

    def test_prefix_continuation_forwards_multimodal_data(self):
        text = self._worker_text()
        self.assertIn('multi_modal_data = data.non_tensor_batch.get("multi_modal_data")', text)
        self.assertIn("multi_modal_data_list=multi_modal_data", text)


if __name__ == "__main__":
    unittest.main()
