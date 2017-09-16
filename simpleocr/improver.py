from PIL import ImageEnhance, Image, ImageOps
import numpy
import cv2

"""
These functions are not suitable for use on images to be grounded and then trained, as the file on disk is not actually
modified. These functions are only to be used on ImageFile objects that are meant to be performed OCR on, nothing else.
These functions offer various improvement options to make the segmentation and classification of the segments in the
image easier. However, they are no miracle workers, images still need to be of decent quality and provide clear
characters to classify.
"""


def enhance_image(imagefile, color=None, brightness=None, contrast=None, sharpness=None, invert=False):
    """
    Enhance an image to make the chance of success of performing OCR on it larger.
    :param imagefile: ImageFile object
    :param color: Color saturation increase, float
    :param brightness: Brightness increase, float
    :param contrast: Contrast increase, float
    :param sharpness: Sharpness increase, float
    :param invert: Invert the colors of the image, bool
    :return: modified ImageFile object, with no changes written to the actual file
    """
    image = imagefile_to_pillow(imagefile)
    if color is not None:
        image = ImageEnhance.Color(image).enhance(color)
    if brightness is not None:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast is not None:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if sharpness is not None:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    if invert:
        image = ImageOps.invert(image)
    imagefile.image = pillow_to_numpy(image)
    return imagefile


def crop_image(imagefile, box):
    """
    Crop an ImageFile object image to the box coordinates. This function is not suitable for use on images to be
    grounded and then trained, as the file on disk is not actually modified.
    :param imagefile: ImageFile object
    :param box: (x, y, x, y) tuple
    :return: modified ImageFile object
    """
    if not isinstance(box, tuple):
        raise ValueError("The box parameter is not a tuple")
    if not len(box) == 4:
        raise ValueError("The box parameter does not have length 4")
    image = imagefile_to_pillow(imagefile)
    image.crop(box)
    imagefile.image = pillow_to_numpy(image)
    return imagefile


def imagefile_to_pillow(imagefile):
    """
    Convert an ImageFile object to a Pillow Image object
    :param imagefile: ImageFile object
    :return: Image object
    """
    pillow = cv2.cvtColor(imagefile.image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pillow)


def pillow_to_numpy(pillow):
    """
    Convert a Pillow Image object to an ImageFile object
    :param pillow: Image object
    :return: cv2 compatible array that fits into ImageFile.image
    """
    imagefile = numpy.array(pillow)
    return imagefile[:, :, ::-1].copy()
