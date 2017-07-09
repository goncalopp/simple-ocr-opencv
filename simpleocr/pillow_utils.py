from .files import ImageBuffer
from PIL import Image
import numpy
import cv2


def imagebuffer_to_pil(imagefile):
    """
    Convert an ImageFile or ImageBuffer object to a Pillow Image object
    :param imagefile: ImageFile object
    :return: Image object
    """
    pillow = cv2.cvtColor(imagefile.image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pillow)


def pil_to_imagebuffer(pillow):
    """
    Convert a Pillow Image object to an ImageBuffer object
    """
    return ImageBuffer(pil_to_cv_array(pillow))


def pil_to_cv_array(pillow):
    """ Convert a Pillow Image object to a cv compatible array """
    imagefile = numpy.array(pillow)
    return imagefile[:, :, ::-1].copy()
