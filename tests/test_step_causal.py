import unittest

from verl.utils.causal_rationale import RationaleTarget, SpatialFact
from verl.utils.step_causal import (
    build_prefixes_for_step_interventions,
    compute_step_causal_score,
    generate_step_interventions,
)


class StepCausalTest(unittest.TestCase):
    def setUp(self):
        self.target = RationaleTarget(
            objects=["red cube.1", "blue sphere.2"],
            relations=[
                SpatialFact(
                    fact_type="relation",
                    subject="red cube.1",
                    predicate="left of",
                    object="blue sphere.2",
                )
            ],
            answer="yes",
            confidence=1.0,
            source="unit",
        )
        self.response = (
            "<observe>objects</observe>"
            "<scene>{\"objects\": [], \"relationships\": []}</scene>"
            "<think>1. The red cube is left of the blue sphere. "
            "2. Therefore the answer is yes.</think>"
            "<answer>yes</answer>"
        )

    def test_generate_step_interventions_changes_one_thinking_step(self):
        interventions = generate_step_interventions(
            self.response,
            self.target,
            max_steps=2,
            max_interventions_per_step=1,
            rollout_index=7,
        )
        self.assertGreaterEqual(len(interventions), 1)
        intervention = interventions[0]
        self.assertEqual(intervention.rollout_index, 7)
        self.assertEqual(intervention.step_index, 0)
        self.assertIn("<think>", intervention.perturbed_response)
        self.assertIn("<answer>yes</answer>", intervention.perturbed_response)
        self.assertNotEqual(intervention.perturbed_response, self.response)

    def test_build_prefixes_stop_at_answer_tag(self):
        interventions = generate_step_interventions(self.response, self.target)
        prefixes = build_prefixes_for_step_interventions(interventions)
        self.assertGreaterEqual(len(prefixes), 1)
        self.assertTrue(prefixes[0].endswith("<answer>"))
        self.assertNotIn("<answer>yes", prefixes[0])

    def test_build_prefixes_stop_at_mixed_case_answer_tag(self):
        response = self.response.replace("<answer>yes</answer>", "<Answer>yes</Answer>")
        interventions = generate_step_interventions(response, self.target)
        prefixes = build_prefixes_for_step_interventions(interventions)
        self.assertGreaterEqual(len(prefixes), 1)
        self.assertTrue(prefixes[0].endswith("<Answer>"))
        self.assertNotIn("<Answer>yes", prefixes[0])

    def test_compute_step_causal_score_uses_normalized_answers(self):
        interventions = generate_step_interventions(self.response, self.target)
        score = compute_step_causal_score(
            original_answer="<answer>yes</answer>",
            interventions=interventions,
            counterfactual_answers=["No."] + ["yes"] * max(0, len(interventions) - 1),
        )
        self.assertGreaterEqual(score.mean, 0.0)
        self.assertLessEqual(score.mean, 1.0)
        self.assertGreater(score.valid_ratio, 0.0)

    def test_compute_step_causal_score_invalid_without_answers(self):
        interventions = generate_step_interventions(self.response, self.target)
        score = compute_step_causal_score("yes", interventions, [None] * len(interventions))
        self.assertEqual(score.mean, -1.0)
        self.assertEqual(score.valid_ratio, 0.0)

    def test_entity_swap_handles_underscore_ids(self):
        response = (
            "<think>The object red_cube.1 is important. "
            "The blue_sphere.2 is the comparison object.</think><answer>yes</answer>"
        )
        interventions = generate_step_interventions(
            response,
            self.target,
            max_steps=1,
            max_interventions_per_step=3,
        )
        entity_interventions = [item for item in interventions if item.intervention_type == "entity"]
        self.assertGreaterEqual(len(entity_interventions), 1)
        self.assertNotEqual(entity_interventions[0].perturbed_response, response)
        self.assertIn("blue_sphere.2", entity_interventions[0].perturbed_response)


if __name__ == "__main__":
    unittest.main()
