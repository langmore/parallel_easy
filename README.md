parallel_easy
=============

Functions to assist in parallel processing with Python 2.7.

Simple wrappers for the multiprocessing library that allow

* Memory-friendly iterator functionality (wrapping `Pool.imap`).
* Exit with `Ctrl-C`.
* Easier use of `n_jobs`.
* When `n_jobs == 1`, processing is serial.
* Wrapping of some *Pandas* "groupby and apply" functionality.

Similar to `joblib.Parallel` but with the addition of `imap` functionality
and a more effective way of handling `Ctrl-C` exit (we add a timeout).

Install
=======

Place this directory somewhere in your `PYTHONPATH`.

Now

    >>> from parallel_easy.parallel_easy import base, pandas_easy
