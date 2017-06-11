import unittest
from files import ImageFile

TEST_FILE = 'digits1'
UNICODE_TEST_FILE = 'unicode1'

files_to_test = [TEST_FILE, UNICODE_TEST_FILE]


class TestImageFile(unittest.TestCase):
    def test_ground(self):
        for file_name in files_to_test:
            imgf = ImageFile(file_name)
            self.assertEqual(imgf.is_grounded, True)
            imgf.set_ground(imgf.ground.segments, imgf.ground.classes, write_file=False)
            self.assertEqual(imgf.is_grounded, True)
            imgf.remove_ground(remove_file=False)
            self.assertEqual(imgf.is_grounded, False)

