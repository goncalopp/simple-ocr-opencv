import os
from pkg_resources import resource_filename
import cv2
from .tesseract_utils import read_boxfile, write_boxfile

IMAGE_EXTENSIONS = ['.png', '.tif', '.jpg', '.jpeg']
DATA_DIRECTORY = resource_filename("simpleocr", "data")
GROUND_EXTENSIONS = ['.box']
GROUND_EXTENSIONS_DEFAULT = GROUND_EXTENSIONS[0]


def try_extensions(extensions, path):
    """Checks for various extensions of a path exist if the extension is appended"""
    for ext in [""] + extensions:
        if os.path.exists(path + ext):
            return path + ext
    return None


def open_image(path):
    return ImageFile(get_file_path(path))


def get_file_path(path, ground=False):
    """Get the absolute path for an image or ground file.
    The path can be either absolute, relative to the CWD or relative to the
    DATA_DIRECTORY. The file extension may be omitted.
    :param path: image path (str)
    :param ground: whether the file must be a ground file
    :return: The absolute path to the file requested
    """
    extensions = GROUND_EXTENSIONS if ground else IMAGE_EXTENSIONS
    # If the path exists, return the path, but make sure it's an absolute path first
    if os.path.exists(path):
        return os.path.abspath(path)
    # Try to find the file with the passed path with the various extensions
    image_with_extension = try_extensions(extensions, os.path.splitext(path)[0])
    if image_with_extension:
        return os.path.abspath(image_with_extension)
    # The file must be in the data directory if it has not yet been found
    image_datadir = try_extensions(extensions, os.path.join(DATA_DIRECTORY, path))
    if image_datadir:
        return os.path.abspath(image_datadir)
    raise IOError # file not found


class Ground(object):
    """Data class that includes labeled characters of an Image and their positions"""
    def __init__(self, segments, classes):
        self.segments = segments
        self.classes = classes


class GroundFile(Ground):
    """Ground with file support. This class can write the data
    to a box file so it can be restored when the image file the ground data belongs
    to is opened again.
    """
    def __init__(self, path, segments, classes):
        Ground.__init__(self, segments, classes)
        self.path = path

    def read(self):
        """Update the ground data stored by reading the box file from disk"""
        self.classes, self.segments = read_boxfile(self.path)

    def write(self):
        """Write a new box file to disk containing the stored ground data"""
        write_boxfile(self.path, self.classes, self.segments)


class Image(object):
    """An image stored in memory. It optionally contains a Ground"""
    def __init__(self, array):
        """:param array: array with image data, must be OpenCV compatible
        """
        self._image = array
        self._ground = None

    def set_ground(self, segments, classes):
        """Creates the ground data"""
        self._ground = Ground(segments=segments, classes=classes)

    def remove_ground(self):
        """Removes the grounding data for the Image"""
        self._ground = None

    # These properties prevent the user from altering the attributes stored within
    # the object and thus emphasize the immutability of the object
    @property
    def image(self):
        return self._image

    @property
    def is_grounded(self):
        return not (self._ground is None)

    @property
    def ground(self):
        return self._ground


class ImageFile(Image):
    """
    Complete class that contains functions for creation from file.
    Also supports grounding in memory.
    """
    def __init__(self, path):
        """
        :param path: path to the image to read, must be valid and absolute
        """
        if not os.path.isabs(path):
            raise ValueError("path value is not absolute: {0}".format(path))
        array = cv2.imread(path)
        Image.__init__(self, array)
        self._path = path
        basepath = os.path.splitext(path)[0]
        self._ground_path = try_extensions(GROUND_EXTENSIONS, basepath)
        if self._ground_path:
            self._ground = GroundFile(self._ground_path, None, None)
            self._ground.read()
        else:
            self._ground_path = basepath + GROUND_EXTENSIONS_DEFAULT
            self._ground = None

    def set_ground(self, segments, classes, write_file=False):
        """Creates the ground, saves it to a file"""
        self._ground = GroundFile(self._ground_path, segments=segments, classes=classes)
        if write_file:
            self.ground.write()

    def remove_ground(self, remove_file=False):
        """Removes ground, optionally deleting it's file"""
        self._ground = None
        if remove_file:
            os.remove(self._ground_path)

    @property
    def path(self):
        return self._path

    @property
    def ground_path(self):
        return self._ground_path


