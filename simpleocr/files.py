import os
from pkg_resources import resource_filename
import cv2
import numpy
from .tesseract_utils import read_boxfile, write_boxfile

IMAGE_EXTENSIONS = ['.png', '.tif', '.jpg', '.jpeg']
DATA_DIRECTORY = resource_filename("simpleocr", "data")
GROUND_EXTENSIONS = ['.box']
GROUND_EXTENSIONS_DEFAULT = GROUND_EXTENSIONS[0]


def try_extensions(extensions, path):
    """Checks for various extensions if a path exist if the extension is appended"""
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


class Image(object):
    """
    Complete class that contains functions for creation from file,
    as well as from PIL Images. Also supports grounding in memory.
    """
    def __init__(self, array=None, path=None, debug=False):
        # If array is actually a numpy array the truth value is ambiguous so it should be compared to None
        if array is None:
            # If array and path are both None, the object is created without any image data, so raise a ValueError
            if not path:
                raise ValueError("Image instance cannot be created without either an image array or an image path")
            # Make sure the path is an absolute path to the image
            path = self.get_absolute_path(path)
            # Read the image and get an OpenCV compatible array
            array = cv2.imread(path)
        # Set all the attributes for the instance
        self.image = array
        self.path = path
        self._debug = debug
        # If a path is set, then the file is created with a file reference, so the existence of a ground file should
        # be checked, and if it's available, the old ground data should be restored to memory
        if self.path:
            # Get the correct path for a ground file
            self.ground_path = try_extensions(GROUND_EXTENSIONS, os.path.splitext(self.path)[0])
            if os.path.exists(self.ground_path):
                # Read the old ground data
                self.ground = GroundFile(self.ground_path)
                self.ground.read()
            else:
                # If the file is not available, set ground data to None
                self.ground = None
        else:
            # No file reference is available, so no ground data is available as the file has not yet been grounded
            self.ground = None
            self.ground_path = None

    @staticmethod
    def from_file(path):
        """
        Create an image object with a path to an image. As the path is
        put through the get_absolute_path function, the valid inputs are the same
        as for the path passed to that function
        :param path: image path, str
        :return: Image object with file reference
        """
        path = Image.get_absolute_path(path)
        array = cv2.imread(path)
        return Image(array=array, path=path)

    @staticmethod
    def from_pil(pillow):
        """ Create an Image object based off of a Pillow Image """
        return Image(Image.get_array_from_pil(pillow))

    @staticmethod
    def get_absolute_path(image):
        """
        Get the absolute path for an image. Valid inputs are:
        - Relative path with or without extension
        - File in DATA_DIRECTORY with or without extension
        - Absolute path with or without extension
        :param image: image path in str type
        :return: The absolute image path
        """
        # If the path exists, return the path, but make sure it's an absolute path first
        if os.path.exists(image):
            return os.path.abspath(image)
        # Try to find the file with the passed path with the various extensions
        image_with_extension = try_extensions(IMAGE_EXTENSIONS, os.path.splitext(image)[0])
        if image_with_extension:
            return os.path.abspath(image_with_extension)
        # The file must be in the data directory if it has not yet been found
        image_basename = os.path.basename(image)
        image_datadir = try_extensions(IMAGE_EXTENSIONS, os.path.join(DATA_DIRECTORY, image_basename))
        if image_datadir:
            return os.path.abspath(image_datadir)
        # The file cannot be found, so raise a FileNotFound Error
        raise Image.FileNotFound

    @staticmethod
    def get_array_from_pil(pillow):
        """
        Returns an OpenCV compatible array from a Pillow Image object
        :param pillow: pillow image object
        :return: OpenCV compatible numpy array
        """
        imagefile = numpy.array(pillow)
        return imagefile[:, :, ::-1].copy()

    @property
    def is_grounded(self):
        """
        :return: True if Image is grounded
        """
        return not (self.ground is None)

    @property
    def has_file_reference(self):
        """
        :return: True if Image has a file reference
        """
        return self.path is not None

    def set_ground(self, segments, classes, write=False):
        """
        Creates the ground data in memory, optionally writing it to a ground file,
        if write is set to True
        :param segments: segments to be grounded
        :param classes: classes to be grounded
        :param write: if True, writes the ground data to a ground file
        :return: None
        """
        if self.is_grounded and self._debug:
            print("Warning: grounding already grounded file")
        self.ground = GroundFile(self.ground_path)
        self.ground.segments = segments
        self.ground.classes = classes
        if write:
            if not self.ground_path:
                raise ValueError("Cannot write ground file for an Image without file reference")
            self.ground.write()

    def remove_ground(self, remove=False):
        """
        Removes the grounding data in memory for the Image, optionally deleting its
        ground file along with it, if remove is set to True
        :param remove: if True, also removes the ground file
        :return: None
        """
        if not self.is_grounded:
            print("Warning: removing ground for file without ground data")
        self.ground = None
        if remove:
            if not self.ground_path:
                raise ValueError("Cannot remove ground file for an Image without file reference")
            os.remove(self.ground_path)

    class FileNotFound(IOError):
        """
        Child-error for IOError for use in the Image class. If error handling is required,
        this error can still be caught in an except block catching an IOError, but it will
        be immediately clear in stack trace where this error is thrown otherwise.

        Note: This error is defined within the Image class as to not override a built-in error
        under Python 3.
        """
        pass


