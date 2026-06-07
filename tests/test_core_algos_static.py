import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_ALGOS_PATH = os.path.join(REPO_ROOT, "verl", "trainer", "core_algos.py")


class CoreAlgosStaticTest(unittest.TestCase):
    def _core_algos_text(self):
        with open(CORE_ALGOS_PATH, encoding="utf-8") as f:
            return f.read()

    def test_group_zscore_uses_population_std_to_avoid_sparse_nan(self):
        text = self._core_algos_text()
        self.assertIn("vals.std(unbiased=False)", text)
        self.assertNotIn("vals.std() if vals.numel() > 1", text)
        self.assertNotIn("id2std[idx] = vals.std()\n", text)


if __name__ == "__main__":
    unittest.main()
