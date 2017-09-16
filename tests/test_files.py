import unittest
from simpleocr.files import ImageFile

TEST_FILE = 'digits1'
UNICODE_TEST_FILE = 'unicode1'


class TestImageFile(unittest.TestCase):
    def test_ground(self):
        imgf = ImageFile(TEST_FILE)
        self.assertEqual(imgf.is_grounded, True)
        imgf.set_ground(imgf.ground.segments, imgf.ground.classes, write_file=False)
        self.assertEqual(imgf.is_grounded, True)
        imgf.remove_ground(remove_file=False)
        self.assertEqual(imgf.is_grounded, False)

    def test_ground_unicode(self):
        imgf = ImageFile(UNICODE_TEST_FILE)
        self.assertEqual(imgf.is_grounded, True)
        imgf.set_ground(imgf.ground.segments, imgf.ground.classes, write_file=False)
        self.assertEqual(imgf.is_grounded, True)
        imgf.remove_ground(remove_file=False)
        self.assertEqual(imgf.is_grounded, False)
