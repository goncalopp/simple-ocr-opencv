import os
import functools
from pkg_resources import resource_filename
import cv2
from .tesseract_utils import read_boxfile, write_boxfile

IMAGE_EXTENSIONS = ['.png', '.tif', '.jpg', '.jpeg']
DATA_DIRECTORY = resource_filename("simpleocr", "data")
GROUND_EXTENSIONS = ['.box']
GROUND_EXTENSIONS_DEFAULT = GROUND_EXTENSIONS[0]


def open_img(path_or_name):
    """
    Fuzzy finds a image file given a absolute or relative path, or a name.
    The name might have no extension, or be in the DATA_DIRECTORY
    """
    try_img_ext = functools.partial(try_extensions, IMAGE_EXTENSIONS)
    data_dir_path = os.path.join(DATA_DIRECTORY, path_or_name)
    path = path_or_name
    if not os.path.exists(path):
        # proceed even when there's no result. ImageFile decides on the exception to raise
        path = try_img_ext(path_or_name) or try_img_ext(data_dir_path) or path
    return ImageFile(path)

def try_extensions(extensions, path):
    """checks if various extensions of a path exist"""
    for ext in [""] + extensions:
        if os.path.exists(path + ext):
            return path + ext
    return None


class GroundFile(object):
    """A file with ground truth data about a image (i.e.: characters and their position)"""
    def __init__(self, path):
        self.path = path
        self.segments = None
        self.classes = None

    def read(self):
        self.classes, self.segments = read_boxfile(self.path)

    def write(self):
        write_boxfile(self.path, self.classes, self.segments)


class ImageFile(object):
    """
    An OCR image file. Has an image and its file path, and optionally
    a ground (ground segments and classes) and it's file path
    """

    def __init__(self, image_path):
        if not os.path.exists(image_path):
            raise IOError("Image file not found: " + image_path)
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        basepath = os.path.splitext(image_path)[0]
        self.ground_path = try_extensions(GROUND_EXTENSIONS, basepath)
        if self.ground_path:
            self.ground = GroundFile(self.ground_path)
            self.ground.read()
        else:
            self.ground_path = basepath + GROUND_EXTENSIONS_DEFAULT
            self.ground = None

    @property
    def is_grounded(self):
        """checks if this file is grounded"""
        return not (self.ground is None)

    def set_ground(self, segments, classes, write_file=False):
        """creates the ground, saves it to a file"""
        if self.is_grounded:
            print("Warning: grounding already grounded file")
        self.ground = GroundFile(self.ground_path)
        self.ground.segments = segments
        self.ground.classes = classes
        if write_file:
            self.ground.write()

    def remove_ground(self, remove_file=False):
        """removes ground, optionally deleting it's file"""
        if not self.is_grounded:
            print("Warning: ungrounding ungrounded file")
        self.ground = None
        if remove_file:
            os.remove(self.ground_path)
