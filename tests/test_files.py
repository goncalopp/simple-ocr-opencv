import os
import unittest
from PIL import Image as PillowImage
import simpleocr.files
from simpleocr.files import Image

TEST_FILE = 'digits1'
TEST_FILE_EXT = 'digits1.png'
UNICODE_TEST_FILE = 'unicode1'


class TestImageFile(unittest.TestCase):
    def test_image_from_file(self):
        # in data dir, no extension
        Image.from_file(TEST_FILE)
        # in data dir, with extension
        Image.from_file(TEST_FILE_EXT)
        # absolute path, no extension
        data_dir = simpleocr.files.DATA_DIRECTORY
        Image.from_file(os.path.join(data_dir, TEST_FILE))
        # absolute path, with extension
        data_dir = simpleocr.files.DATA_DIRECTORY
        Image.from_file(os.path.join(data_dir, TEST_FILE_EXT))
        #
        data_dir_name = os.path.basename(data_dir)
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(data_dir)) # set cwd to one above data_dir
        try:
            # relative path, no extension
            Image.from_file(os.path.join(data_dir_name, TEST_FILE))
            # relative path, with extension
            Image.from_file(os.path.join(data_dir_name, TEST_FILE_EXT))
        finally:
            os.chdir(old_cwd)

    def test_from_file_inexistent(self):
        with self.assertRaises(IOError):
            Image.from_file("inexistent")

    def test_ground(self):
        imgf = Image.from_file(TEST_FILE)
        self.assertEqual(imgf.is_grounded, True)
        imgf.set_ground(imgf.ground.segments, imgf.ground.classes, write=False)
        self.assertEqual(imgf.is_grounded, True)
        imgf.remove_ground(remove=False)
        self.assertEqual(imgf.is_grounded, False)

    def test_ground_unicode(self):
        imgf = Image.from_file(UNICODE_TEST_FILE)
        self.assertEqual(imgf.is_grounded, True)
        imgf.set_ground(imgf.ground.segments, imgf.ground.classes, write=False)
        self.assertEqual(imgf.is_grounded, True)
        imgf.remove_ground(remove=False)
        self.assertEqual(imgf.is_grounded, False)

    def test_image_from_pillow(self):
        pillow = PillowImage.open(Image.get_absolute_path(TEST_FILE))
        image = Image.from_pil(pillow)
        self.assertIsInstance(image, Image)
