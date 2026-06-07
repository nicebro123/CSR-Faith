import argparse
import importlib.util
import json
import os
import tempfile
import unittest


SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "build_csr_target_cache.py",
)

spec = importlib.util.spec_from_file_location("build_csr_target_cache", SCRIPT_PATH)
build_csr_target_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_csr_target_cache)


class BuildCSRTargetCacheTest(unittest.TestCase):
    def _record(self):
        scene = {
            "objects": [
                {"id": "red_cube.1", "bbox": [0, 0, 10, 10]},
                {"id": "blue_sphere.2", "bbox": [20, 0, 30, 10]},
            ],
            "relationships": [
                {"subject": "red_cube.1", "predicate": "left of", "object": "blue_sphere.2"},
            ],
        }
        scene_text = json.dumps(scene)
        return {
            "id": "sample-1",
            "problem": "Is the red cube left of the blue sphere?",
            "ground_truth": f"<scene>{scene_text}</scene><answer>yes</answer>",
        }

    def test_build_cache_records(self):
        args = argparse.Namespace(
            problem_key="problem",
            ground_truth_key="ground_truth",
            max_relations=4,
            max_objects=6,
        )
        records = build_csr_target_cache.build_cache_records([self._record()], args)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "sample-1")
        self.assertEqual(records[0]["gt_answer"], "yes")
        self.assertGreater(records[0]["target_fact_count"], 0)
        self.assertEqual(records[0]["target_source"], "relation_overlap")

    def test_write_jsonl(self):
        records = [{"id": "a", "target_fact_count": 1}]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "targets.jsonl")
            build_csr_target_cache.write_jsonl(records, output_path)
            with open(output_path, encoding="utf-8") as f:
                loaded = [json.loads(line) for line in f]
        self.assertEqual(loaded, records)


if __name__ == "__main__":
    unittest.main()
