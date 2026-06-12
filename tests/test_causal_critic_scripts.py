import json
import os
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(name):
    return os.path.join(REPO_ROOT, "scripts", name)


def _record(sample_id, counterfactual_answer):
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
        "id": sample_id,
        "problem": "Is the red cube left of the blue sphere?",
        "ground_truth": f"<scene>{scene_text}</scene><answer>yes</answer>",
        "response": (
            f"<scene>{scene_text}</scene>"
            "<think>1. The red cube is left of the blue sphere. "
            "2. Therefore the answer is yes.</think>"
            "<answer>yes</answer>"
        ),
        "counterfactual_answers": [counterfactual_answer],
    }


class CausalCriticScriptTest(unittest.TestCase):
    def test_build_train_evaluate_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "rollouts.jsonl")
            dataset_path = os.path.join(tmpdir, "critic.jsonl")
            ckpt_dir = os.path.join(tmpdir, "critic_ckpt")
            metrics_path = os.path.join(tmpdir, "eval_metrics.json")

            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(_record("positive", "no")) + "\n")
                f.write(json.dumps(_record("negative", "yes")) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    _script("build_causal_critic_dataset.py"),
                    "--input-json",
                    input_path,
                    "--output",
                    dataset_path,
                    "--drop-invalid-labels",
                ],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertTrue(os.path.exists(dataset_path))

            subprocess.run(
                [
                    sys.executable,
                    _script("train_causal_spatial_critic.py"),
                    "--train-jsonl",
                    dataset_path,
                    "--val-jsonl",
                    dataset_path,
                    "--output-dir",
                    ckpt_dir,
                    "--epochs",
                    "2",
                    "--n-features",
                    "128",
                ],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "critic.json")))
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "metrics.json")))

            subprocess.run(
                [
                    sys.executable,
                    _script("evaluate_causal_spatial_critic.py"),
                    "--critic-path",
                    ckpt_dir,
                    "--eval-jsonl",
                    dataset_path,
                    "--output-json",
                    metrics_path,
                ],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            self.assertEqual(metrics["eval"]["num_examples"], 2)


if __name__ == "__main__":
    unittest.main()
