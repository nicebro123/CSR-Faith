import argparse
import importlib.util
import json
import os
import tempfile
import unittest


SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "debug_csr_batch.py",
)

spec = importlib.util.spec_from_file_location("debug_csr_batch", SCRIPT_PATH)
debug_csr_batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debug_csr_batch)
from scripts.csr_debug_io import load_json_records


class DebugCSRBatchTest(unittest.TestCase):
    def setUp(self):
        self.scene = {
            "objects": [
                {"id": "red_cube.1", "bbox": [0, 0, 10, 10]},
                {"id": "blue_sphere.2", "bbox": [20, 0, 30, 10]},
            ],
            "relationships": [
                {"subject": "red_cube.1", "predicate": "left of", "object": "blue_sphere.2"},
            ],
        }

    def _record(self):
        scene_text = json.dumps(self.scene)
        return {
            "problem": "Is the red cube left of the blue sphere?",
            "ground_truth": f"<scene>{scene_text}</scene><answer>yes</answer>",
            "response": (
                f"<observe>objects</observe><scene>{scene_text}</scene>"
                "<think>The red cube is left of the blue sphere.</think><answer>yes</answer>"
            ),
        }

    def test_analyze_record_outputs_rationale_and_interventions(self):
        record = self._record()
        args = argparse.Namespace(
            problem_key="problem",
            ground_truth_key="ground_truth",
            response_key="response",
            max_relations=4,
            max_objects=6,
            max_steps=6,
            max_step_interventions=1,
            preview_chars=300,
        )

        output = debug_csr_batch.analyze_record(record, args)
        self.assertEqual(output["gt_answer"], "yes")
        self.assertEqual(output["target"]["source"], "relation_overlap")
        self.assertGreater(output["rationale_score"]["overall"], 0.0)
        self.assertGreaterEqual(len(output["step_interventions"]), 1)

    def test_load_json_records_accepts_object_and_jsonl(self):
        record = self._record()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "sample.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(record, f)
            self.assertEqual(len(load_json_records(json_path)), 1)

            jsonl_path = os.path.join(tmpdir, "sample.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.write(json.dumps(record) + "\n")
            self.assertEqual(len(load_json_records(jsonl_path)), 2)

    def test_response_falls_back_to_ground_truth(self):
        record = self._record()
        record.pop("response")
        args = argparse.Namespace(
            problem_key="problem",
            ground_truth_key="ground_truth",
            response_key="response",
            max_relations=4,
            max_objects=6,
            max_steps=6,
            max_step_interventions=1,
            preview_chars=300,
        )
        output = debug_csr_batch.analyze_record(record, args)
        self.assertEqual(output["response_answer"], "yes")


if __name__ == "__main__":
    unittest.main()
