from unittest import TestCase
# from bintrees import FastBinaryTree

from src.binary_search import binary_search


class TestBinarySearch(TestCase):

    def test_exists(self):
        pairs = [(0, "0"), (1, "1"), (3, "3"), (4, "4"), (5, "5"), (6, "6"), (8, "8")]
        # pairs = FastBinaryTree(pairs)
        self.assertEqual(binary_search(3, pairs), 2)

    def test_not_exists(self):
        pairs = [(0, "0"), (1, "1"), (3, "3"), (4, "4"), (5, "5"), (6, "6"), (8, "8")]
        self.assertEqual(binary_search(2, pairs), 2)
