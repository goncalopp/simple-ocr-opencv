try:
    import cv2
except ImportError as e:
    import sys


    def is_python_3():
        return sys.version_info[0] is 3

    # Valid values for sys.platform on Linux include "linux" and "linux2"
    if "linux" in sys.platform:
        if is_python_3():
            print(
                "OpenCV-Python could not be imported. As your are running Linux and Python 3, you have the following "
                "options to install it:"
                "\n- Compile OpenCV-Python yourself"
                "\n- Install the unofficial \"opencv-python\" package from PyPI using pip"
            )
        else:
            print(
                "OpenCV-Python could not be imported. As you are running Linux and Python 2, you have the following "
                "options to install it:"
                "\n- Compile OpenCV-Python yourself"
                "\n- Install the \"python-opencv\" package with your distro's package manager if it is available"
                "\n- Install the unofficial \"opencv-python\" package from PyPI using pip"
            )
    # The only valid value for Windows is "win32"
    elif sys.platform is "win32":
        print(
            "OpenCV-Python could not be imported. As you are running Windows, you have the following options to "
            "install it:"
            "\n- Compile OpenCV-Python yourself"
            "\n- Install the unofficial \"opencv-python\" package from PyPI using pip"
        )
    else:
        print(
            "OpenCV-Python could not be imported, but there are no installation instructions available for your OS."
        )
    raise


# Classifiers
from simpleocr.classification import KNNClassifier
# Files
from simpleocr.files import open_image, Image, ImageFile
# Grounders
from simpleocr.grounding import TerminalGrounder, TextGrounder, UserGrounder
# Improver functions
from simpleocr.improver import enhance_image, crop_image, image_to_pil
# OCR functions
from simpleocr.ocr import reconstruct_chars, show_differences, OCR
# Segmenters
from simpleocr.segmentation import RawContourSegmenter, ContourSegmenter
# Extraction
from simpleocr.feature_extraction import FeatureExtractor, SimpleFeatureExtractor
# Pillow functions
from simpleocr.pillow_utils import pil_to_image
