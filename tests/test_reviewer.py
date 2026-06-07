import unittest

try:
    from verl.utils.reviewer import extract_answer_text, extract_scene_text, extract_think
except ModuleNotFoundError:
    extract_answer_text = None
    extract_scene_text = None
    extract_think = None


@unittest.skipIf(extract_answer_text is None, "reviewer dependencies are not installed")
class ReviewerExtractionTest(unittest.TestCase):
    def test_extractors_are_tag_case_insensitive(self):
        text = "<Scene>{}</Scene><Think>reasoning</Think><Answer>yes</Answer>"
        self.assertEqual(extract_scene_text(text), "{}")
        self.assertEqual(extract_think(text), "reasoning")
        self.assertEqual(extract_answer_text(text), "yes")


if __name__ == "__main__":
    unittest.main()
