"""
Functions to assist in parallel processing with Python 2.7.

Simple wrappers for the multiprocessing library that allow
* Exit with Ctrl-C
* Easier use of n_jobs
* When n_jobs == 1, processing is serial

Similar to joblib.Parallel but with the addition of imap functionality
and a different way of handling Ctrl-C exit (add a timeout).
"""
import itertools
from functools import partial
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import IMapUnorderedIterator, IMapIterator
import cPickle
import sys

import numpy as np
import pandas as pd


################################################################################
# Globals
################################################################################
# Used as the timeout
GOOGLE = 1e100


def imap_easy(func, iterable, n_jobs, chunksize, ordered=True):
    """
    Returns a parallel iterator of func over iterable.

    Worker processes return one "chunk" of data at a time, and the iterator
    allows you to deal with each chunk as they come back, so memory can be
    handled efficiently.

    Parameters
    ----------
    func : Function of one variable
        You can use functools.partial to build this.  
        A lambda function will not work
    iterable : List, iterator, etc...
        func is applied to this
    n_jobs : Integer
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.
    chunksize : Integer
        Jobs/results will be sent between master/slave processes in chunks of
        size chunksize.  If chunksize is too small, communication overhead
        slows things down.  If chunksize is too large, one process ends up
        doing too much work (and large results will up in memory).
    ordered : Boolean
        If True, results are dished out in the order corresponding to iterable.
        If False, results are dished out in whatever order workers return them.

    Examples
    --------
    >>> from functools import partial
    >>> from jrl_utils.src.parallel_easy import imap_easy, map_easy
    >>> def abfunc(x, a, b=1):
    ...     return x * a * b
    >>> some_numbers = range(5)
    >>> func = partial(abfunc, 2, b=3)
    >>> results_iterator = imap_easy(func, some_numbers, 2, 5)
    >>> for result in results_iterator:
    ...     print result
    0
    6
    12
    18
    24

    Notes
    -----
    This is an implementation of multiprocessing.Pool.imap that allows easier
    use of n_jobs and exit with Ctrl-C.
    """
    n_jobs = _n_jobs_wrap(n_jobs)

    if n_jobs == 1:
        results_iter = itertools.imap(func, iterable)
    else:
        _trypickle(func)
        pool = Pool(n_jobs)
        if ordered:
            results_iter = pool.imap(func, iterable, chunksize=chunksize)
        else:
            results_iter = pool.imap_unordered(
                func, iterable, chunksize=chunksize)

    return results_iter


def map_easy(func, iterable, n_jobs):
    """
    Returns a parallel map of func over iterable.
    Returns all results at once, so if results are big memory issues may arise
    
    Parameters
    ----------
    func : Function of one variable
        You can use functools.partial to build this.  
        A lambda function will not work
    iterable : List, iterator, etc...
        func is applied to this
    n_jobs : Integer
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    Examples
    --------
    >>> from functools import partial
    >>> from jrl_utils.src.parallel_easy import imap_easy, map_easy
    >>> def abfunc(x, a, b=1):
    ...     return x * a * b
    >>> some_numbers = range(5)
    >>> func = partial(abfunc, 2, b=3)
    >>> map_easy(func, some_numbers)
    [0, 6, 12, 18, 24]

    Notes
    -----
    This is an implementation of multiprocessing.Pool.map_async that allows
    easier use of n_jobs and exit with Ctrl-C.
    """
    _trypickle(func)
    n_jobs = _n_jobs_wrap(n_jobs)

    if n_jobs == 1:
        return map(func, iterable)
    else:
        return Pool(n_jobs).map_async(func, iterable).get(GOOGLE)



def _n_jobs_wrap(n_jobs):
    """
    For dealing with positive or negative n_jobs.

    Parameters
    ----------
    n_jobs : Integer

    Returns
    -------
    n_jobs_modified : Integer
        If -1, equal to multiprocessing.cpu_count() (all CPU's used).
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.
    """
    if not isinstance(n_jobs, int):
        raise ValueError(
            "type(n_jobs) = %s, but n_jobs should be an int" % type(n_jobs))

    if (n_jobs == 0) or (n_jobs < -1 * cpu_count()):
        msg = "Must have -1 + cpu_count() <= n_jobs < 0  OR  1 <= n_jobs"
        raise ValueError("n_jobs = %d, but %s" % (n_jobs, msg))

    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)

    return n_jobs


def _imap_wrap(func):
    """
    Adds timeout to IMapIterator and IMapUnorderedIterator.
    This allows exit upon Ctrl-C.  This is a fix
    of the known python bug  bugs.python.org/issue8296 given by 
    https://gist.github.com/aljungberg/626518

    Parameters
    ----------
    func : Either IMapIterator or IMapUnorderedIterator

    Returns
    -------
    wrap : Function
        Wrapped version of func, with timeout specified
    """
    # func will be a next() method of IMapIterator.  Note that the first argument
    # to methods are always 'self'.
    def wrap(self, timeout=None):
        return func(self, timeout=timeout if timeout is not None else GOOGLE)
    return wrap


def _trypickle(func):
    """
    Attempts to pickle func since multiprocessing needs to do this.
    """
    boundmethodmsg = """
    func contained a bound method, and these cannot be pickled.  This causes
    multiprocessing to fail.  A bound method occurs when e.g. you set an
    attribute equal to a method, as in self.myfunc = self.mymethod, or when
    you set an attribute equal to a lambda function.
    """
    genericmsg = "Pickling of func (necessary for multiprocessing) failed."

    try:
        cPickle.dumps(func)
    except TypeError as e:
        if 'instancemethod' in e.message:
            sys.stderr.write(boundmethodmsg + "\n")
        else:
            sys.stderr.write(genericmsg + '\n')
        raise 
    except:
        sys.stderr.write(genericmsg + '\n')
        raise

# Redefine IMapUnorderedIterator so we can exit with Ctrl-C
IMapUnorderedIterator.next = _imap_wrap(IMapUnorderedIterator.next)
IMapIterator.next = _imap_wrap(IMapIterator.next)


if __name__ == '__main__':
    # Can't get doctest to work with multiprocessing...
    pass
