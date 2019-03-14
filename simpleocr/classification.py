from .feature_extraction import FEATURE_DATATYPE
import numpy
import cv2
from .opencv_utils import get_opencv_version
from six import unichr

CLASS_DATATYPE = numpy.uint16
CLASS_SIZE = 1
CLASSES_DIRECTION = 0  # vertical - a classes COLUMN

BLANK_CLASS = unichr(35)  # marks unclassified elements


def classes_to_numpy(classes):
    """given a list of unicode chars, transforms it into a numpy array"""
    import array
    # utf-32 starts with constant ''\xff\xfe\x00\x00', then has little endian 32 bits chars
    # this assumes little endian architecture!
    assert unichr(15).encode('utf-32') == b'\xff\xfe\x00\x00\x0f\x00\x00\x00'
    assert array.array("I").itemsize == 4
    int_classes = array.array("I", "".join(classes).encode('utf-32')[4:])
    assert len(int_classes) == len(classes)
    classes = numpy.array(int_classes, dtype=CLASS_DATATYPE, ndmin=2)  # each class in a column. numpy is strange :(
    classes = classes if CLASSES_DIRECTION == 1 else numpy.transpose(classes)
    return classes


def classes_from_numpy(classes):
    """reverses classes_to_numpy"""
    classes = classes if CLASSES_DIRECTION == 0 else classes.tranpose()
    classes = list(map(unichr, classes))
    return classes


class Classifier(object):
    def train(self, features, classes):
        """trains the classifier with the classified feature vectors"""
        raise NotImplementedError()

    @staticmethod
    def _filter_unclassified(features, classes):
        classified = (classes != classes_to_numpy(BLANK_CLASS)).reshape(-1)
        return features[classified], classes[classified]

    def classify(self, features):
        """returns the classes of the feature vectors"""
        raise NotImplementedError


class KNNClassifier(Classifier):
    def __init__(self, k=1, debug=False):
        if get_opencv_version() >= 3:
            self.knn = cv2.ml.KNearest_create()
        else:
            self.knn = cv2.KNearest()
        self.k = k
        self.debug = debug

    def train(self, features, classes):
        if FEATURE_DATATYPE != numpy.float32:
            features = numpy.asarray(features, dtype=numpy.float32)
        if CLASS_DATATYPE != numpy.float32:
            classes = numpy.asarray(classes, dtype=numpy.float32)
        features, classes = Classifier._filter_unclassified(features, classes)
        if get_opencv_version() >= 3:
            self.knn.train(features, cv2.ml.ROW_SAMPLE, classes)
        else:
            self.knn.train(features, classes)

    def classify(self, features):
        if FEATURE_DATATYPE != numpy.float32:
            features = numpy.asarray(features, dtype=numpy.float32)
        if get_opencv_version() >= 3:
            retval, result_classes, neigh_resp, dists = self.knn.findNearest(features, k=1)
        else:
            retval, result_classes, neigh_resp, dists = self.knn.find_nearest(features, k=1)
        return result_classes
