import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(REPO_ROOT, "verl", "trainer", "metrics.py")


class MetricsStaticTest(unittest.TestCase):
    def _metrics_text(self):
        with open(METRICS_PATH, encoding="utf-8") as f:
            return f.read()

    def test_data_metrics_use_safe_tensor_stats(self):
        text = self._metrics_text()
        self.assertIn("def _safe_tensor_stat(values: torch.Tensor, stat: str, default: float = 0.0) -> float:", text)
        self.assertIn("if values.numel() == 0:", text)
        self.assertIn('"critic/advantages/max": _safe_tensor_stat(valid_adv, "max")', text)
        self.assertIn('"critic/returns/min": _safe_tensor_stat(valid_returns, "min")', text)
        self.assertNotIn("torch.max(valid_adv)", text)
        self.assertNotIn("torch.max(valid_returns)", text)

    def test_critic_variance_and_timing_division_are_safe(self):
        text = self._metrics_text()
        self.assertIn("def _safe_tensor_var(values: torch.Tensor) -> torch.Tensor:", text)
        self.assertIn("return torch.var(values, unbiased=False)", text)
        self.assertIn("def _safe_divisor(value: float) -> float:", text)
        self.assertIn("_safe_divisor(num_tokens_of_section[name])", text)
        self.assertIn('"perf/throughput": total_num_tokens / _safe_divisor(time * n_gpus)', text)


if __name__ == "__main__":
    unittest.main()
