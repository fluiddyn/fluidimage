import unittest

from .util import cprint, imread, is_memory_full, str_short


class TestUtil(unittest.TestCase):
    def test_util(self):

        try:
            imread("__oups__.png")
        except (NameError, FileNotFoundError):
            print("IOError", IOError)

        cprint("nice")
        is_memory_full()
        str_short("string")


if __name__ == "__main__":
    unittest.main()
