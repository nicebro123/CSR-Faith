import json
import unittest

from verl.utils.answer_normalization import answers_equal, normalize_answer
from verl.utils.causal_rationale import (
    build_causal_rationale_target,
    extract_facts_from_response,
    extract_gt_scene_and_answer,
    score_rationale,
    split_thinking_steps,
)


def _wrapped_scene(scene, answer="left"):
    return f"<scene>{json.dumps(scene)}</scene><answer>{answer}</answer>"


class AnswerNormalizationTest(unittest.TestCase):
    def test_normalizes_tags_and_option_prefixes(self):
        self.assertEqual(normalize_answer("<answer>Choice: (A) Left.</answer>"), "left")
        self.assertTrue(answers_equal("<answer>Left.</answer>", "left"))


class CausalRationaleTest(unittest.TestCase):
    def setUp(self):
        self.scene = {
            "objects": [
                {"id": "red_cube.1", "bbox": [0, 0, 10, 10]},
                {"id": "blue_sphere.2", "bbox": [20, 0, 30, 10]},
                {"id": "green_cone.3", "bbox": [40, 0, 50, 10]},
            ],
            "relationships": [
                {"subject": "red_cube.1", "predicate": "left of", "object": "blue_sphere.2"},
                {"subject": "blue_sphere.2", "predicate": "left of", "object": "green_cone.3"},
            ],
        }

    def test_extract_gt_scene_and_answer(self):
        scene, answer = extract_gt_scene_and_answer(_wrapped_scene(self.scene, "red cube"))
        self.assertEqual(answer, "red cube")
        self.assertEqual(scene["objects"][0]["id"], "red_cube.1")

    def test_build_target_prefers_question_overlap_relations(self):
        target = build_causal_rationale_target(
            "Is the red cube left of the blue sphere?",
            self.scene,
            "yes",
            max_relations=1,
        )
        self.assertEqual(target.source, "relation_overlap")
        self.assertEqual(len(target.relations), 1)
        self.assertEqual(target.relations[0].subject, "red cube.1")
        self.assertIn("blue sphere.2", target.objects)

    def test_build_target_falls_back_when_no_overlap(self):
        target = build_causal_rationale_target("Which option is correct?", self.scene, "yes")
        self.assertEqual(target.source, "relation_fallback")
        self.assertGreater(len(target.relations), 0)
        self.assertGreater(target.confidence, 0.0)

    def test_extract_facts_from_response_scene(self):
        response = _wrapped_scene(self.scene, "yes")
        facts = extract_facts_from_response(response)
        relation_facts = [fact for fact in facts if fact.fact_type == "relation"]
        self.assertEqual(len(relation_facts), 2)
        self.assertEqual(relation_facts[0].predicate, "left of")

    def test_score_rationale_rewards_coverage_and_precision(self):
        target = build_causal_rationale_target(
            "Is the red cube left of the blue sphere?",
            self.scene,
            "yes",
            max_relations=1,
        )
        response = (
            "<observe>objects</observe>"
            "<scene>{}</scene>"
            "<think>The red cube is left of the blue sphere, so the answer is yes.</think>"
            "<answer>yes</answer>"
        ).format(json.dumps(self.scene))
        score = score_rationale(response, target)
        self.assertEqual(score.coverage, 1.0)
        self.assertGreater(score.precision, 0.0)
        self.assertGreater(score.overall, 0.8)

    def test_score_rationale_penalizes_missing_target_fact(self):
        target = build_causal_rationale_target(
            "Is the red cube left of the blue sphere?",
            self.scene,
            "yes",
            max_relations=1,
        )
        response = (
            "<observe>objects</observe><scene>{}</scene>"
            "<think>The green cone is visible.</think><answer>yes</answer>"
        ).format(json.dumps({"objects": [], "relationships": []}))
        score = score_rationale(response, target)
        self.assertLess(score.coverage, 1.0)
        self.assertLess(score.overall, 0.8)

    def test_split_thinking_steps(self):
        response = "<think>1. First relation. 2. Second relation.</think>"
        steps = split_thinking_steps(response)
        self.assertGreaterEqual(len(steps), 1)
        self.assertIn("relation", steps[0])


if __name__ == "__main__":
    unittest.main()
