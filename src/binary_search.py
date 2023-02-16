from typing import Any
# from bintrees import FastBinaryTree
# https://pypi.org/project/bintrees/


def binary_search(key, pairs) -> int:
    """Find index of `key` in `pairs` by binary search.

    If `key` exists`, return the exact index. Otherwise, return the index where it would appear.

    Args:
        key: A key to search for.
        pairs: A list of key-value pairs, sorted by key.
    """
    if len(pairs) == 0:
        return 0
    return _binary_search(key, pairs, 0, len(pairs))


def _binary_search(key, pairs, start, stop) -> int:
    """Helper function for binary search.

    Args:
        key: A key to search for.
        pairs: A list of key-value pairs, sorted by key.
        start: First valid pair.
        stop: First invalid pair.
    """
    if start + 1 == stop:
        return start

    mid_idx = (start + stop) // 2
    mid_key, _ = pairs[mid_idx]
    if key < mid_key:
        return _binary_search(key, pairs, start, mid_idx)
    else:
        return _binary_search(key, pairs, mid_idx, stop)