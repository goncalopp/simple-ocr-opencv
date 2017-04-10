from opencv_utils import show_image_and_wait_for_key, draw_segments
import numpy
import segmentation as segmenters
import classification as classifiers
import feature_extraction as extractors
import grounding as grounders
from files import ImageFile


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


class OCR(object):
    def __init__(self, segmenter, feature_extractor, classifier):
        self.segmenter = segmenter
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def train(self, image_file):
        """feeds the training data to the OCR"""
        if not image_file.is_grounded():
            raise Exception("The provided file is not grounded")
        features = self.feature_extractor.extract(image_file.image, image_file.ground.segments)
        self.classifier.train(features, image_file.ground.classes)

    def ocr(self, image_file, show_steps=False):
        """performs ocr used trained classifier"""
        segments = self.segmenter.process(image_file.image)
        if show_steps:
            self.segmenter.display()
        features = self.feature_extractor.extract(image_file.image, segments)
        classes = self.classifier.classify(features)
        return classes, segments


class SimpleOCR(object):
    def __init__(self, segmentation="contour", extraction="simple", classification="knn",
                 grounding="text"):
        """
        Makes the OCR process easier by providing all objects required for doing OCR
        by itself, but allowing the user to choose other types of objects by specifying
        them as strings in keyword arguments.

        :param segmentation: contour, rawcontour or raw
        :param extraction: simple (currently no other options are available)
        :param classifier: knn (currently no other options are available)
        :param grounding: text or user
        """
        if segmentation == "contour":
            self.segmenter = segmenters.ContourSegmenter()
        elif segmentation == "raw":
            self.segmenter = segmenters.RawSegmenter()
        elif segmentation == "rawcontour":
            self.segmenter = segmenters.RawContourSegmenter()
        else:
            raise NotImplementedError
        if extraction == "simple":
            self.extractor = extractors.SimpleFeatureExtractor()
        else:
            raise NotImplementedError
        if classification == "knn":
            self.classifier = classifiers.KNNClassifier()
        else:
            raise NotImplementedError
        if grounding == "text":
            self.grounder = grounders.TextGrounder()
        elif grounding == "user":
            self.grounder = grounders.UserGrounder()
        else:
            raise NotImplementedError

    def train(self, image_file, text=None):
        """
        feeds the training data to the OCR
        :param image_file: The image file to be trained
        :param text: Text to feed TextGrounder if this is the grounder type
        :return:
        """
        if not image_file.is_grounded():
            if not text and isinstance(self.grounder, grounders.TextGrounder):
                raise ValueError("Ungrounded file with TextGrounder and text is None")
            segments = self.segmenter.process(image_file.image)
            if isinstance(self.grounder, grounders.TextGrounder):
                self.grounder.ground(image_file.image, segments, text)
            elif isinstance(self.grounder, grounders.UserGrounder):
                self.grounder.ground(image_file.image, segments)
            else:
                raise NotImplementedError
        features = self.extractor.extract(image_file.image, image_file.ground.segments)
        self.classifier.train(features, image_file.ground.classes)

    def ocr(self, image_file, show_steps=False):
        """
        performs ocr used trained classifier
        :param image_file: the image file to perform ocr on
        :param show_steps: shows individual steps if True
        :return: classes, segments
        """
        segments = self.segmenter.process(image_file.image)
        if show_steps:
            self.segmenter.display()
        features = self.extractor.extract(image_file.image, segments)
        classes = self.classifier.classify(features)
        return classes, segments

    def ocr_chars(self, image_file, show_steps=False):
        """
        performs ocr used trained classifier, but returns only characters
        :param image_file: the image file to perform ocr on
        :param show_steps: shows individual steps if True
        :return: classes, segments
        """
        segments = self.segmenter.process(image_file.image)
        if show_steps:
            self.segmenter.display()
        features = self.extractor.extract(image_file.image, segments)
        classes = self.classifier.classify(features)
        return reconstruct_chars(classes)

    def ground_file(self, filename, text=None):
        """
        Ground an image file for use in the OCR object.
        :param filename: The name of the image file (either in cwd/data or full path)
        :param text: The text, if self.grounder is a TextGrounder (defaults to None)
        :return:
        """
        try:
            new_img = ImageFile(filename)
        except Exception:
            raise ValueError("File name is not valid: %s" % filename)
        segments = self.segmenter.process(new_img.image)
        if isinstance(self.grounder, grounders.TextGrounder):
            if not text:
                raise ValueError("Trying to ground file with TextGrounder without specified text.")
            self.grounder.ground(new_img, segments, text)
        elif isinstance(self.grounder, grounders.UserGrounder):
            self.grounder.ground(new_img, segments)


contour_segmenter = "contour"
raw_segmenter = "raw"
rawcontour_segmenter = "rawcontour"
simple_extractor = "simple"
KNN_classifier = "knn"
user_grounder = "user"
text_grounder = "text"
