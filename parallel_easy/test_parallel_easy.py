import unittest
from functools import partial

import parallel_easy


# A couple functions for testing parallel easy
# Must be defined outside of the test class for some reason.
def _abfunc(x, a, b=1):
    return x * a * b
abfunc = partial(_abfunc, 2, 3)


class TestParallelEasy(unittest.TestCase):
    """
    Tests the parallel_easy module.
    """
    def setUp(self):
        self.numbers = range(5)
        self.benchmark = [0, 6, 12, 18, 24]

    def test_map_easy_1job(self):
        result = parallel_easy.map_easy(abfunc, self.numbers, 1)
        self.assertEqual(result, self.benchmark)

    def test_map_easy_3job(self):
        result = parallel_easy.map_easy(abfunc, self.numbers, 3)
        self.assertEqual(result, self.benchmark)

    def test_imap_easy_1job(self):
        result_iterator = parallel_easy.imap_easy(abfunc, self.numbers, 1, 1)
        result = []
        for number in result_iterator:
            result.append(number)
        self.assertEqual(result, self.benchmark)

    def test_imap_easy_3job(self):
        result_iterator = parallel_easy.imap_easy(abfunc, self.numbers, 3, 1)
        result = []
        for number in result_iterator:
            result.append(number)
        self.assertEqual(result, self.benchmark)

    def test_n_jobs_wrap_positive(self):
        """
        For n_jobs positive, the wrap should return n_jobs.
        """
        for n_jobs in range(1, 5):
            result = parallel_easy._n_jobs_wrap(n_jobs)
            self.assertEqual(result, n_jobs)

    def test_n_jobs_wrap_zero(self):
        """
        For n_jobs zero, the wrap should raise a ValueError
        """
        self.assertRaises(ValueError, parallel_easy._n_jobs_wrap, 0)

