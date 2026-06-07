import unittest

from verl.utils.counterfactual import build_prefix_for_continuation, generate_counterfactual_inputs


class CounterfactualTest(unittest.TestCase):
    def test_build_prefix_for_continuation_is_answer_tag_case_insensitive(self):
        response = "<think>The object is left.</think><Answer>yes</Answer>"
        prefix = build_prefix_for_continuation(response)
        self.assertEqual(prefix, "<think>The object is left.</think><Answer>")

    def test_generate_counterfactual_inputs_accepts_mixed_case_scene_tag(self):
        response = (
            "<Scene>{\"objects\":[{\"id\":\"red_cube.1\"},{\"id\":\"blue_sphere.2\"}],"
            "\"relationships\":[{\"subject\":\"red_cube.1\",\"predicate\":\"left of\","
            "\"object\":\"blue_sphere.2\"}]}</Scene>"
            "<Think>The red cube is left of the blue sphere.</Think>"
            "<answer>yes</answer>"
        )
        interventions = generate_counterfactual_inputs(response)
        self.assertGreaterEqual(len(interventions), 1)


if __name__ == "__main__":
    unittest.main()
