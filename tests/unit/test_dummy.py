"""Module contains a dummy unit test that always passes."""

import unittest


class DummyTest(unittest.TestCase):
    def test_0(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
