# Written by Goncalopp and RedFantom
# Thranta Squadron GSF CombatLog Parser, Copyright (C) 2016 by Goncalopp and RedFantom
# All additions are under the copyright of their respective authors
# For license see LICENSE
from PIL import ImageEnhance, Image, ImageOps
import files
import numpy
import cv2


class ImageImprover(object):
    """
    Improver class that provides a wrap around the ImageEnhance classes from PIL/pillow
    The Improved image can be accessed through object.image, this is a PIL/pillow RGB
    format image, that is NOT compatible with the cv2 based files.ImageFile class!
    """
    def __init__(self, filename):
        self.filename = filename
        try:
            self.image = Image.open(filename)
        except IOError:
            raise ValueError("File path/name not valid")

    def enhance(self, color=0.0, brightness=0.05, contrast=15.0,
                sharpness=20.0, invert=False):
        self.image = ImageEnhance.Color(self.image).enhance(color)
        self.image = ImageEnhance.Brightness(self.image).enhance(brightness)
        self.image = ImageEnhance.Contrast(self.image).enhance(contrast)
        self.image = ImageEnhance.Sharpness(self.image).enhance(sharpness)
        if invert:
            self.image = ImageOps.invert(self.image)

    def crop(self, box):
        if not isinstance(box, tuple):
            raise ValueError("The provided box is not a tuple")
        if not len(box) == 4:
            raise ValueError("The provided box tuple is not the right size")
        self.image = self.image.crop(box)


class ImageFileImprover(object):
    def __init__(self, imagefile):
        self.imagefile = imagefile
        self.image = cv2.cvtColor(self.imagefile.image, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(self.image)

    def enhance(self, color=0.1, brightness=0.1, contrast=10.0,
                sharpness=10.0, invert=False):
        self.image = ImageEnhance.Color(self.image).enhance(color)
        self.image = ImageEnhance.Brightness(self.image).enhance(brightness)
        self.image = ImageEnhance.Contrast(self.image).enhance(contrast)
        self.image = ImageEnhance.Sharpness(self.image).enhance(sharpness)
        if invert:
            self.image = ImageOps.invert(self.image)
        self.imagefile.image = numpy.array(self.image)
        self.imagefile.image = self.imagefile.image[:, :, ::-1].copy()

    def crop(self, box):
        if not isinstance(box, tuple):
            raise ValueError("The provided box is not a tuple")
        if not len(box) == 4:
            raise ValueError("The provided box tuple is not the right size")
        self.image = self.image.crop(box)
        self.imagefile.image = numpy.array(self.image)
        self.imagefile.image = self.imagefile.image[:, :, ::-1].copy()


def imagefile_to_pillow(imagefile):
    pillow = cv2.cvtColor(imagefile.image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pillow)


def pillow_to_numpy(pillow):
    imagefile = numpy.array(pillow)
    return imagefile[:, :, ::-1].copy()
