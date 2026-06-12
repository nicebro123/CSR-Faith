import json
import os
import tempfile
import unittest

from verl.utils.causal_critic_data import (
    build_critic_examples,
    filter_labeled_examples,
    read_jsonl,
    write_jsonl,
)
from verl.utils.causal_rationale import RationaleTarget, SpatialFact
from verl.utils.step_causal import generate_step_interventions


class CausalCriticDataTest(unittest.TestCase):
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
        scene = {
            "objects": [],
            "relationships": [],
        }
        self.response = (
            f"<scene>{json.dumps(scene)}</scene>"
            "<think>1. The red cube is left of the blue sphere. "
            "2. Therefore the answer is yes.</think>"
            "<answer>yes</answer>"
        )

    def test_build_critic_examples_labels_answer_changes(self):
        interventions = generate_step_interventions(self.response, self.target)
        examples = build_critic_examples(
            problem="Is the red cube left of the blue sphere?",
            target=self.target,
            response_text=self.response,
            interventions=interventions,
            counterfactual_answers=["no"],
            uid="sample-1",
            response_answer="yes",
        )
        self.assertGreaterEqual(len(examples), 1)
        self.assertEqual(examples[0].label_answer_changed, 1)
        self.assertEqual(examples[0].reward_vector["causal_effect"], 1.0)
        self.assertEqual(examples[0].target_relations[0]["predicate"], "left of")

    def test_invalid_counterfactual_answer_is_unlabeled(self):
        interventions = generate_step_interventions(self.response, self.target)
        examples = build_critic_examples(
            problem="Is the red cube left of the blue sphere?",
            target=self.target,
            response_text=self.response,
            interventions=interventions,
            counterfactual_answers=[None],
            uid="sample-1",
            response_answer="yes",
        )
        self.assertIsNone(examples[0].label_answer_changed)
        self.assertEqual(filter_labeled_examples(examples), [])

    def test_jsonl_roundtrip_preserves_schema(self):
        interventions = generate_step_interventions(self.response, self.target)
        examples = build_critic_examples(
            problem="Is the red cube left of the blue sphere?",
            target=self.target,
            response_text=self.response,
            interventions=interventions,
            counterfactual_answers=["yes"],
            uid="sample-1",
            response_answer="yes",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "critic.jsonl")
            write_jsonl(examples, path)
            loaded = read_jsonl(path)
        self.assertEqual(len(loaded), len(examples))
        self.assertEqual(loaded[0].label_answer_changed, 0)
        self.assertIn("intervention_valid", loaded[0].reward_vector)


if __name__ == "__main__":
    unittest.main()
