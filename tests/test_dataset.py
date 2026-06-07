import os
import unittest

try:
    from verl.utils.dataset import RLHFDataset
except ModuleNotFoundError:
    RLHFDataset = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(REPO_ROOT, "verl", "utils", "dataset.py")


class RLHFDatasetStaticTest(unittest.TestCase):
    def test_postprocess_uses_pad_token_fallback(self):
        with open(DATASET_PATH, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("def _pad_token_id(self) -> int:", text)
        self.assertIn("if self.tokenizer.pad_token_id is not None:", text)
        self.assertIn("if self.tokenizer.eos_token_id is not None:", text)
        self.assertIn("pad_token_id=self._pad_token_id(),", text)

    def test_image_key_resolution_skips_none_values_before_fallback(self):
        with open(DATASET_PATH, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("if key in row_dict and row_dict[key] is not None:", text)
        self.assertIn("for key in (self.image_key, \"image\", \"images\"):\n            if key in row_dict:\n                return key", text)

    def test_local_parquet_file_is_loaded_directly(self):
        with open(DATASET_PATH, encoding="utf-8") as f:
            text = f.read()
        self.assertIn('data_path, data_split = data_path.split("@", 1)', text)
        file_branch = text.split("elif os.path.isfile(data_path):", 1)[1].split("else:  # remote dataset", 1)[0]
        self.assertIn("data_files={data_split: data_path}", file_branch)
        self.assertNotIn("get_data_files(data_path, data_split)", file_branch)


@unittest.skipIf(RLHFDataset is None, "dataset dependencies are not installed")
class RLHFDatasetTest(unittest.TestCase):
    def test_resolve_image_key_falls_back_to_singular_image(self):
        dataset = object.__new__(RLHFDataset)
        dataset.image_key = "images"
        self.assertEqual(dataset._resolve_image_key({"image": object()}), "image")

    def test_resolve_image_key_prefers_configured_key(self):
        dataset = object.__new__(RLHFDataset)
        dataset.image_key = "images"
        self.assertEqual(dataset._resolve_image_key({"images": object(), "image": object()}), "images")

    def test_resolve_image_key_skips_none_configured_value(self):
        dataset = object.__new__(RLHFDataset)
        dataset.image_key = "images"
        self.assertEqual(dataset._resolve_image_key({"images": None, "image": object()}), "image")


if __name__ == "__main__":
    unittest.main()
