parallel_easy
=============

Functions to assist in parallel processing with Python 2.7.

Simple wrappers for the multiprocessing library that allow

* Exit with Ctrl-C
* Easier use of n_jobs
* When n_jobs == 1, processing is serial

Similar to joblib.Parallel but with the addition of imap functionality
and a different way of handling Ctrl-C exit (add a timeout).

Install
=======

Place this directory somewhere in your PYTHONPATH.

Now

    >>> from parallel_easy.parallel_easy import parallel_easy
