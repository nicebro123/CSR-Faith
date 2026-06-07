import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_FUNCTIONAL_PATH = os.path.join(REPO_ROOT, "verl", "utils", "py_functional.py")


class PyFunctionalStaticTest(unittest.TestCase):
    def test_union_two_dict_uses_explicit_value_errors(self):
        with open(PY_FUNCTIONAL_PATH, encoding="utf-8") as f:
            text = f.read()
        union_body = text.split("def union_two_dict", 1)[1].split("def append_to_dict", 1)[0]
        self.assertIn("torch.equal(left, right)", union_body)
        self.assertIn("np.array_equal(left, right)", union_body)
        self.assertIn("raise ValueError", union_body)
        self.assertNotIn("assert dict1[key] == dict2[key]", union_body)


if __name__ == "__main__":
    unittest.main()
