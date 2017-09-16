import numpy
import cv2
from .segmentation import region_from_segment
from .opencv_utils import background_color

FEATURE_DATATYPE = numpy.float32
# FEATURE_SIZE is defined on the specific feature extractor instance
FEATURE_DIRECTION = 1  # horizontal - a COLUMN feature vector
FEATURES_DIRECTION = 0  # vertical - ROWS of feature vectors


class FeatureExtractor(object):
    """given a list of segments, returns a list of feature vectors"""
    def extract(self, image, segments):
        raise NotImplementedError()


class SimpleFeatureExtractor(FeatureExtractor):
    def __init__(self, feature_size=10, stretch=False):
        self.feature_size = feature_size
        self.stretch = stretch

    def extract(self, image, segments):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fs = self.feature_size
        bg = background_color(image)

        regions = numpy.ndarray(shape=(0, fs), dtype=FEATURE_DATATYPE)
        for segment in segments:
            region = region_from_segment(image, segment)
            if self.stretch:
                region = cv2.resize(region, (fs, fs))
            else:
                x, y, w, h = segment
                proportion = float(min(h, w)) / max(w, h)
                new_size = (fs, int(fs * proportion)) if min(w, h) == h else (int(fs * proportion), fs)
                region = cv2.resize(region, new_size)
                s = region.shape
                newregion = numpy.ndarray((fs, fs), dtype=region.dtype)
                newregion[:, :] = bg
                newregion[:s[0], :s[1]] = region
                region = newregion
            regions = numpy.append(regions, region, axis=0)
        regions.shape = (len(segments), fs ** 2)
        return regions
