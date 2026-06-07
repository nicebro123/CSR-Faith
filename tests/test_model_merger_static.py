import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_MERGER_PATH = os.path.join(REPO_ROOT, "scripts", "model_merger.py")


class ModelMergerStaticTest(unittest.TestCase):
    def _text(self):
        with open(MODEL_MERGER_PATH, encoding="utf-8") as f:
            return f.read()

    def test_first_shard_load_uses_weights_only_false(self):
        text = self._text()
        self.assertIn('return torch.load(model_path, map_location="cpu", weights_only=False)', text)

    def test_threadpool_futures_are_checked(self):
        text = self._text()
        self.assertIn("futures = [executor.submit(process_one_shard, rank)", text)
        self.assertIn("future.result()", text)

    def test_local_dir_is_normalized_and_huggingface_subdir_rejected(self):
        text = self._text()
        self.assertIn("def _normalize_actor_dir", text)
        self.assertIn("os.path.normpath(os.path.abspath(local_dir))", text)
        self.assertIn('os.path.basename(local_dir) == "huggingface"', text)

    def test_no_bare_asserts_for_user_facing_model_merger_errors(self):
        text = self._text()
        self.assertNotIn("assert not args.local_dir.endswith", text)
        self.assertNotIn("assert world_size", text)
        self.assertNotIn("assert isinstance(weight", text)


if __name__ == "__main__":
    unittest.main()
