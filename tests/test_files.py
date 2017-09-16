import os
import unittest

import simpleocr.files
from simpleocr.files import open_img

TEST_FILE = 'digits1'
TEST_FILE_EXT = 'digits1.png'
UNICODE_TEST_FILE = 'unicode1'

class TestImageFile(unittest.TestCase):
    def test_open_img(self):
        # in data dir, no extension
        open_img(TEST_FILE)
        # in data dir, with extension
        open_img(TEST_FILE_EXT)
        # absolute path, no extension
        data_dir = simpleocr.files.DATA_DIRECTORY
        open_img(os.path.join(data_dir, TEST_FILE))
        # absolute path, with extension
        data_dir = simpleocr.files.DATA_DIRECTORY
        open_img(os.path.join(data_dir, TEST_FILE_EXT))
        #
        data_dir_name = os.path.basename(data_dir)
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(data_dir)) # set cwd to one above data_dir
        try:
            # relative path, no extension
            open_img(os.path.join(data_dir_name, TEST_FILE))
            # relative path, with extension
            open_img(os.path.join(data_dir_name, TEST_FILE_EXT))
        finally:
            os.chdir(old_cwd)

    def test_open_img_inexistent(self):
        with self.assertRaises(IOError):
            open_img("inexistent")


    def test_ground(self):
        imgf = open_img(TEST_FILE)
        self.assertEqual(imgf.is_grounded, True)
        imgf.set_ground(imgf.ground.segments, imgf.ground.classes, write_file=False)
        self.assertEqual(imgf.is_grounded, True)
        imgf.remove_ground(remove_file=False)
        self.assertEqual(imgf.is_grounded, False)

    def test_ground_unicode(self):
        imgf = open_img(UNICODE_TEST_FILE)
        self.assertEqual(imgf.is_grounded, True)
        imgf.set_ground(imgf.ground.segments, imgf.ground.classes, write_file=False)
        self.assertEqual(imgf.is_grounded, True)
        imgf.remove_ground(remove_file=False)
        self.assertEqual(imgf.is_grounded, False)
