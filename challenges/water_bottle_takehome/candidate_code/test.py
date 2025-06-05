import unittest

from water_bottle_challenge import classify_preprocessed_audio

class TestClassify(unittest.TestCase):
    def test_top(self):
        "Correct top classification with known top signal"
        fname = "train/top.csv"
        result = classify_preprocessed_audio(fname)
        self.assertEqual(result, 0)

    def test_bottom(self):
        "Correct bottom classification with known bottom signal"
        fname = "train/bottom.csv"
        result = classify_preprocessed_audio(fname)
        self.assertEqual(result, 1)

    def test_unknown(self):
        "Correct ambiguous classification with known null signal"
        fname = "train/null.csv"
        result = classify_preprocessed_audio(fname)
        print(result)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
