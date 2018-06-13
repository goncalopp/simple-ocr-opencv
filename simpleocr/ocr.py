import numpy
from .opencv_utils import show_image_and_wait_for_key, draw_segments
from . import segmentation as segmenters
from . import classification as classifiers
from . import feature_extraction as extractors
from . import grounding as grounders
from .files import open_image, Image
from six import unichr

SEGMENTERS = {
    "contour": segmenters.ContourSegmenter,
    "raw": segmenters.RawSegmenter,
    "rawcontour": segmenters.RawContourSegmenter,
}
EXTRACTORS = {"simple": extractors.SimpleFeatureExtractor}
CLASSIFIERS = {"knn": classifiers.KNNClassifier}
GROUNDERS = {"user": grounders.UserGrounder, "text": grounders.TextGrounder}


def show_differences(image, segments, ground_classes, result_classes):
    image = image.copy()
    good = (ground_classes == result_classes)
    good.shape = (len(good),)  # transform nx1 matrix into vector
    draw_segments(image, segments[good, :], (0, 255, 0))
    draw_segments(image, segments[numpy.logical_not(good), :], (0, 0, 255))
    show_image_and_wait_for_key(image, "differences")


def reconstruct_chars(classes):
    result_string = "".join(map(unichr, classes))
    return result_string


def accuracy(expected, result):
    if expected.shape != result.shape:
        raise Exception("expected " + str(expected.shape) + ", got " + str(result.shape))
    correct = expected == result
    return float(numpy.count_nonzero(correct)) / correct.shape[0]


def get_instance_from(x, class_dict, default_key):
    """Gets a instance of a class, given a class dict and x.
    X can be either a instance (already), the key to the dict, or None.
    If x is None, class_dict[default_key] will be instanciated"""
    k = x or default_key
    cls = class_dict.get(k)
    instance = cls() if cls else x
    return instance


class OCR(object):
    def __init__(self, segmenter=None, extractor=None, classifier=None, grounder=None):
        self.segmenter = get_instance_from(segmenter, SEGMENTERS, "contour")
        self.extractor = get_instance_from(extractor, EXTRACTORS, "simple")
        self.classifier = get_instance_from(classifier, CLASSIFIERS, "knn")
        self.grounder = get_instance_from(grounder, GROUNDERS, "text")

    def train(self, image_file):
        """feeds the training data to the OCR"""
        if not isinstance(image_file, Image):
            image_file = open_image(image_file)
        if not image_file.is_grounded:
            raise Exception("The provided file is not grounded")
        features = self.extractor.extract(image_file.image, image_file.ground.segments)
        self.classifier.train(features, image_file.ground.classes)

    def ocr(self, image_file, show_steps=False):
        """performs ocr used trained classifier"""
        if not isinstance(image_file, Image):
            image_file = open_image(image_file)
        segments = self.segmenter.process(image_file.image)
        if show_steps:
            self.segmenter.display()
        features = self.extractor.extract(image_file.image, segments)
        classes = self.classifier.classify(features)
        chars = reconstruct_chars(classes)
        return chars, classes, segments

    def ground(self, image_file, text=None):
        """
        Ground an image file for use in the OCR object.
        :param image_file: The name of the image file or an ImageFile object
        :param text: The text, if self.grounder is a TextGrounder (defaults to None)
        :return:
        """
        if not isinstance(image_file, Image):
            image_file = open_image(image_file)
        segments = self.segmenter.process(image_file.image)
        if isinstance(self.grounder, grounders.TextGrounder):
            if not text:
                raise ValueError("Trying to ground file with TextGrounder without specifying text argument.")
            self.grounder.ground(image_file, segments, text)
        else:
            self.grounder.ground(image_file, segments)
        image_file.ground.write()  # save to file
