import unittest

try:
    import numpy as np
    import torch
    from tensordict import TensorDict

    from verl.protocol import DataProto
    from verl.workers.reward.config import RewardConfig
    from verl.workers.reward.custom import CustomRewardManager
except ModuleNotFoundError:
    np = None
    torch = None
    TensorDict = None
    DataProto = None
    RewardConfig = None
    CustomRewardManager = None


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        if len(token_ids) == 0:
            return ""
        return "<answer>yes</answer>"


@unittest.skipIf(CustomRewardManager is None, "reward manager dependencies are not installed")
class CustomRewardManagerTest(unittest.TestCase):
    def test_non_spatial_reward_does_not_require_problem_field(self):
        manager = CustomRewardManager(_Tokenizer(), RewardConfig(score_function="r1v"))
        data = DataProto(
            batch=TensorDict(
                {
                    "responses": torch.tensor([[1, 2, 3]]),
                    "response_mask": torch.tensor([[1, 1, 1]]),
                },
                batch_size=1,
            ),
            non_tensor_batch={
                "ground_truth": np.array(["<answer>yes</answer>"], dtype=object),
            },
        )

        rewards, metrics = manager(data)
        self.assertEqual(rewards[0, 2].item(), 1.0)
        self.assertIn("overall", metrics)

    def test_empty_response_mask_leaves_reward_zero(self):
        manager = CustomRewardManager(_Tokenizer(), RewardConfig(score_function="r1v"))
        data = DataProto(
            batch=TensorDict(
                {
                    "responses": torch.tensor([[1, 2, 3]]),
                    "response_mask": torch.tensor([[0, 0, 0]]),
                },
                batch_size=1,
            ),
            non_tensor_batch={
                "ground_truth": np.array(["<answer>yes</answer>"], dtype=object),
            },
        )

        rewards, metrics = manager(data)
        self.assertTrue(torch.equal(rewards, torch.zeros_like(rewards)))
        self.assertEqual(metrics, {})

    def test_scoring_exception_sets_zero_reward_and_error_metric(self):
        manager = CustomRewardManager(_Tokenizer(), RewardConfig(score_function="r1v"))

        def fail_score(response, ground_truth):
            raise RuntimeError("bad sample")

        manager.compute_score = fail_score
        data = DataProto(
            batch=TensorDict(
                {
                    "responses": torch.tensor([[1, 2, 3]]),
                    "response_mask": torch.tensor([[1, 1, 1]]),
                },
                batch_size=1,
            ),
            non_tensor_batch={
                "ground_truth": np.array(["<answer>yes</answer>"], dtype=object),
            },
        )

        rewards, metrics = manager(data)
        self.assertTrue(torch.equal(rewards, torch.zeros_like(rewards)))
        self.assertEqual(metrics["overall"], [0.0])
        self.assertEqual(metrics["error"], [1.0])


if __name__ == "__main__":
    unittest.main()
