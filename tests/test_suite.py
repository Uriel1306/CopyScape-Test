"""Tests: python -m unittest tests.test_suite  (or: python tests/test_suite.py)"""
import os
import subprocess
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import detector as D  # noqa: E402

RES = D.EMPTY_RESOURCES
NAMES = D.feature_names(D.BASE_GROUPS)
IDX = {n: i for i, n in enumerate(NAMES)}

SENTENCE = ("The old lighthouse keeper recorded every passing ship in a leather notebook "
            "throughout the long winter storms and never missed a single night")  # 18+ words


def feats(src, cand, res=RES, groups=D.BASE_GROUPS):
    return D.extract_features(src, cand, res, groups)


def filler(n_words: int, seed: int) -> str:
    rng = np.random.RandomState(seed)
    return " ".join(f"w{i}" for i in rng.randint(0, 20000, n_words))


class TestCleaning(unittest.TestCase):
    def test_html_nbsp_and_glued_digits(self):
        self.assertEqual(D.clean_text("<div>since the&nbsp;1680s</div>"), D.clean_text("since the 1680s"))
        self.assertEqual(D.clean_text("since the1680s"), "since the 1680 s")

    def test_adjacent_tags_do_not_glue_words(self):
        self.assertEqual(D.clean_text("<div>one</div><div>two</div>"), "one two")

    def test_invisibles_and_fullwidth(self):
        self.assertEqual(D.clean_text("light​house ke­eper ｔｅａ"), "lighthouse keeper tea")

    def test_cyrillic_and_greek_homoglyphs(self):
        self.assertEqual(D.clean_text("соld еggs аnd νery"), "cold eggs and very")

    def test_non_string_inputs(self):
        for bad in (None, float("nan"), 7, "", "   ", "<br><br>"):
            self.assertEqual(D.clean_text(bad), "")

    def test_views_aligned(self):
        toks, masked = D._tokenize_pass("Yesterday Kessow paid 12,000 to Anna.")
        self.assertEqual(masked, ["yesterday", D.ENT, "paid", D.NUM, D.NUM, "to", D.ENT])
        self.assertEqual(len(toks), len(masked))


class TestResources(unittest.TestCase):
    def test_hash_is_process_independent(self):
        code = ("import sys; sys.path.insert(0, %r); import detector as D; "
                "print(D.shingle_hashes(['a','b','c','d'], 3).tolist())" % ROOT)
        outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env={**os.environ, "PYTHONHASHSEED": s}).stdout for s in ("1", "2")}
        self.assertEqual(len(outs), 1)

    def test_common_counts_distinct_articles_of_sources_only(self):
        boiler = "follow us on social media for daily facts"
        groups = {i: [f"{boiler} unique{i} text", f"{boiler} again"] for i in range(6)}
        groups["lonely"] = ["a b c d e f g h i"] * 10
        res = D.learn_resources(groups, min_sources=5)
        self.assertTrue(set(D.shingle_hashes(D.tokenize(boiler), 3)) <= set(res.common.tolist()))
        self.assertFalse(set(D.shingle_hashes(list("abcdefghi"), 3)) & set(res.common.tolist()))

    def test_empty_groups(self):
        res = D.learn_resources({}, 5)
        self.assertEqual(len(res.common), 0)


class TestFeatures(unittest.TestCase):
    def test_shape_and_finite_for_all_groups(self):
        v = feats(SENTENCE, SENTENCE + " and more", groups=D.BASE_GROUPS + D.OPTIONAL_GROUPS)
        self.assertEqual(v.shape, (len(D.feature_names(D.BASE_GROUPS + D.OPTIONAL_GROUPS)),))
        self.assertTrue(np.isfinite(v).all())

    def test_identical_text(self):
        v = feats(SENTENCE, SENTENCE)
        self.assertAlmostEqual(v[IDX["cand_cont3"]], 1.0)
        self.assertEqual(v[IDX["longest_run5"]], len(SENTENCE.split()))

    def test_empty_and_degenerate_inputs_give_zeros(self):
        for s, c in (("", SENTENCE), (SENTENCE, ""), (None, None), (float("nan"), SENTENCE),
                     ("<br>", "<br>")):
            self.assertFalse(feats(s, c).any(), (s, c))

    def test_homoglyph_copy_matches_latin_source(self):
        evaded = SENTENCE.replace("o", "о").replace("a", "а").replace("e", "е")
        self.assertAlmostEqual(feats(SENTENCE, evaded)[IDX["cand_cont3"]], 1.0)

    def test_boilerplate_only_overlap_is_masked(self):
        boiler = "follow us on social media for daily facts and photo essays from our team"
        res = D.learn_resources({i: [boiler + f" x{i} y{i} z{i}"] for i in range(6)}, min_sources=5)
        a, b = filler(40, 1) + " " + boiler, filler(40, 2) + " " + boiler
        self.assertGreaterEqual(feats(a, b)[IDX["longest_run5"]], 10)
        masked = feats(a, b, res)
        self.assertLess(masked[IDX["longest_run5"]], 4)
        self.assertLess(masked[IDX["cand_cont3"]], 0.1)

    def test_coincidental_facts_have_no_shared_run(self):
        a = "In 2014 Kessow shipped 12,000 tonnes of grain to the northern ports."
        b = "Grain exports from the harbour reached 12,000 tonnes during 2014 under Kessow."
        self.assertLess(feats(a, b)[IDX["longest_run5"]], 4)

    def test_run_and_window_helpers(self):
        self.assertEqual(D.run_lengths(np.array([1, 1, 0, 1, 0, 1, 1, 1], bool)).tolist(), [2, 1, 3])
        self.assertEqual(D.run_lengths(np.zeros(4, bool)).tolist(), [])
        self.assertEqual(D.window_stats(np.zeros(0, bool), 5, 2), (0.0, 0.0))
        hit = np.zeros(50, bool)
        hit[40:] = True
        self.assertEqual(D.window_stats(hit, 10, 7)[0], 1.0)

    def test_needle_in_25000_word_document(self):
        needle = "the ferry captain wrote down each foggy crossing in a small green logbook every single evening"
        long_cand = filler(12000, 3) + " " + needle + " " + filler(13000, 4)
        long_src = filler(12000, 5) + " " + needle + " " + filler(13000, 6)
        t0 = time.time()
        v = feats(long_src, long_cand)
        self.assertLess(time.time() - t0, 10)
        self.assertGreaterEqual(v[IDX["longest_run5"]], len(needle.split()))
        self.assertGreater(v[IDX["win_max5"]], 0.45)  # a 17-word needle fills ~half a 30-word window
        self.assertLess(v[IDX["cand_cont3"]], 0.01)  # global overlap alone would miss it


if __name__ == "__main__":
    unittest.main(verbosity=2)
