import unittest
from functools import partial

import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np
import scipy as sp

from parallel_easy import base, pandas_easy


# A couple functions for testing parallel easy
# Must be defined outside of the test class for some reason.
def _abfunc(x, a, b=1):
    return x * a * b
abfunc = partial(_abfunc, 2, 3)

def frame_to_series(frame):
    x = frame.iloc[0, 0]
    return pd.Series([x] * len(frame.columns), index=frame.columns)


class TestBase(unittest.TestCase):
    """
    Tests the base module.
    """
    def setUp(self):
        self.numbers = range(5)
        self.benchmark = [0, 6, 12, 18, 24]

    def test_map_easy_1job(self):
        result = base.map_easy(abfunc, self.numbers, 1)
        self.assertEqual(result, self.benchmark)

    def test_map_easy_3job(self):
        result = base.map_easy(abfunc, self.numbers, 3)
        self.assertEqual(result, self.benchmark)

    def test_imap_easy_1job(self):
        result_iterator = base.imap_easy(abfunc, self.numbers, 1, 1)
        result = []
        for number in result_iterator:
            result.append(number)
        self.assertEqual(result, self.benchmark)

    def test_imap_easy_3job(self):
        result_iterator = base.imap_easy(abfunc, self.numbers, 3, 1)
        result = []
        for number in result_iterator:
            result.append(number)
        self.assertEqual(result, self.benchmark)

    def test_n_jobs_wrap_positive(self):
        """
        For n_jobs positive, the wrap should return n_jobs.
        """
        for n_jobs in range(1, 5):
            result = base._n_jobs_wrap(n_jobs)
            self.assertEqual(result, n_jobs)

    def test_n_jobs_wrap_zero(self):
        """
        For n_jobs zero, the wrap should raise a ValueError
        """
        self.assertRaises(ValueError, base._n_jobs_wrap, 0)


class TestPandasEasy(unittest.TestCase):
    """
    Tests the pandas_easy module.
    """
    def setUp(self):
        pass

    def test_groupby_to_scalar_to_series_1(self):
        df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]})
        benchmark = df.groupby('a').apply(max)
        result = pandas_easy.groupby_to_scalar_to_series(df, max, 1, by='a')
        assert_series_equal(result, benchmark)

    def test_groupby_to_scalar_to_series_2(self):
        s = pd.Series([1, 2, 3, 4])
        labels = ['a', 'a', 'b', 'b']
        benchmark = s.groupby(labels).apply(max)
        result = pandas_easy.groupby_to_scalar_to_series(
            s, max, 1, by=labels)
        assert_series_equal(result, benchmark)

    def test_groupby_to_series_to_frame_1(self):
        df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]})
        labels = ['g1', 'g1', 'g2']
        benchmark = df.groupby(labels).mean()
        result = pandas_easy.groupby_to_series_to_frame(
            df, np.mean, 1, use_apply=True, by=labels)
        assert_frame_equal(result, benchmark)

    def test_groupby_to_series_to_frame_2(self):
        df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]})
        labels = ['g1', 'g1', 'g2']
        benchmark = df.groupby(labels).apply(frame_to_series)
        result = pandas_easy.groupby_to_series_to_frame(
            df, frame_to_series, 1, use_apply=False, by=labels)
        assert_frame_equal(result, benchmark)
