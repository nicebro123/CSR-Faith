import os
import tempfile
import unittest

from verl.models.causal_spatial_critic import CausalSpatialCritic, serialize_critic_input
from verl.utils.causal_critic_data import CausalCriticExample


def _example(uid, step, intervention_type, label):
    return CausalCriticExample(
        uid=uid,
        question="Is the red cube left of the blue sphere?",
        target_objects=["red cube.1", "blue sphere.2"],
        target_relations=[
            {"subject": "red cube.1", "predicate": "left of", "object": "blue sphere.2"},
        ],
        target_confidence=1.0,
        response_answer="yes",
        step_index=0,
        step_text=step,
        intervention_type=intervention_type,
        perturbed_step_preview=step,
        counterfactual_answer="no" if label else "yes",
        label_answer_changed=label,
        source="unit",
        curriculum_phase="discovery",
        reward_vector={"causal_effect": float(label), "intervention_valid": 1.0},
    )


class CausalSpatialCriticStaticTest(unittest.TestCase):
    def test_serialization_includes_core_fields(self):
        example = _example("a", "The red cube is left of the blue sphere.", "relation", 1)
        text = serialize_critic_input(example)
        self.assertIn("[QUESTION]", text)
        self.assertIn("[TARGET_RELATIONS]", text)
        self.assertIn("[STEP]", text)
        self.assertIn("[INTERVENTION] relation", text)

    def test_score_batch_outputs_probabilities(self):
        examples = [
            _example("a", "The red cube is left of the blue sphere.", "relation", 1),
            _example("b", "The visible objects are mentioned but no relation changes.", "mask", 0),
            _example("c", "The red cube is left of the blue sphere.", "entity", 1),
            _example("d", "The final sentence only repeats the answer.", "mask", 0),
        ]
        critic = CausalSpatialCritic.train(examples, epochs=3, n_features=256, learning_rate=0.2)
        scores = critic.score_batch(examples)
        self.assertEqual(len(scores), len(examples))
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_save_and_load_preserves_scores(self):
        examples = [
            _example("a", "The red cube is left of the blue sphere.", "relation", 1),
            _example("b", "The final sentence only repeats the answer.", "mask", 0),
        ]
        critic = CausalSpatialCritic.train(examples, epochs=2, n_features=128, learning_rate=0.2)
        with tempfile.TemporaryDirectory() as tmpdir:
            critic.save(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "critic.json")))
            loaded = CausalSpatialCritic.load(tmpdir)
        before = critic.score_batch(examples)
        after = loaded.score_batch(examples)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
