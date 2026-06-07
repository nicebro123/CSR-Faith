import unittest

try:
    import numpy as np
    import torch

    from verl.trainer import core_algos
except ModuleNotFoundError:
    np = None
    torch = None
    core_algos = None


@unittest.skipIf(core_algos is None, "numpy/torch dependencies are not installed")
class CSRAdvantageTest(unittest.TestCase):
    def setUp(self):
        self.rewards = torch.tensor([[1.0], [3.0], [2.0], [4.0]])
        self.mask = torch.ones_like(self.rewards)
        self.index = np.array(["a", "a", "b", "b"], dtype=object)

    def test_invalid_step_cfs_and_zero_lambdas_match_grpo(self):
        rationale = torch.tensor([0.1, 0.9, 0.2, 0.8])
        step_cfs = torch.full((4,), -1.0)
        csr_adv, csr_returns = core_algos.compute_csrfaith_grpo_advantage(
            self.rewards,
            self.mask,
            self.index,
            rationale,
            step_cfs,
            lambda_coverage=0.0,
            lambda_step_cfs=0.0,
        )
        grpo_adv, grpo_returns = core_algos.compute_grpo_outcome_advantage(
            self.rewards.clone(),
            self.mask,
            self.index,
        )
        self.assertTrue(torch.allclose(csr_adv, grpo_adv))
        self.assertTrue(torch.allclose(csr_returns, grpo_returns))

    def test_grpo_accepts_tensor_group_index(self):
        tensor_index = torch.tensor([0, 0, 1, 1])
        np_adv, _ = core_algos.compute_grpo_outcome_advantage(
            self.rewards.clone(),
            self.mask,
            self.index,
        )
        tensor_adv, _ = core_algos.compute_grpo_outcome_advantage(
            self.rewards.clone(),
            self.mask,
            tensor_index,
        )
        self.assertTrue(torch.allclose(tensor_adv, np_adv))

    def test_rloo_uses_grouped_leave_one_out_baseline(self):
        tensor_index = torch.tensor([0, 0, 1, 1])
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            self.rewards.clone(),
            self.mask,
            tensor_index,
        )
        expected = torch.tensor([[-2.0], [2.0], [-2.0], [2.0]])
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertTrue(torch.allclose(returns, expected))

    def test_rationale_lambda_changes_advantage(self):
        rationale = torch.tensor([1.0, 0.0, 1.0, 0.0])
        step_cfs = torch.full((4,), -1.0)
        base_adv, _ = core_algos.compute_csrfaith_grpo_advantage(
            self.rewards,
            self.mask,
            self.index,
            rationale,
            step_cfs,
            lambda_coverage=0.0,
            lambda_step_cfs=0.0,
        )
        csr_adv, _ = core_algos.compute_csrfaith_grpo_advantage(
            self.rewards,
            self.mask,
            self.index,
            rationale,
            step_cfs,
            lambda_coverage=1.0,
            lambda_step_cfs=0.0,
        )
        self.assertFalse(torch.allclose(base_adv, csr_adv))

    def test_csr_multiplier_update(self):
        lambda_coverage, lambda_step = core_algos.update_csr_lagrangian_multipliers(
            lambda_coverage=0.0,
            lambda_step_cfs=0.3,
            batch_coverage_mean=0.4,
            batch_step_cfs_mean=-1.0,
            tau_coverage=0.7,
            tau_step_cfs=0.5,
            eta=0.1,
        )
        self.assertAlmostEqual(lambda_coverage, 0.03)
        self.assertAlmostEqual(lambda_step, 0.3)


if __name__ == "__main__":
    unittest.main()
