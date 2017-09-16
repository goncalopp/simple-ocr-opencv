from .opencv_utils import show_image_and_wait_for_key, draw_segments, BlurProcessor, get_opencv_version
from .processor import DisplayingProcessor, DisplayingProcessorStack, create_broadcast
from .segmentation_aux import SegmentOrderer
from .segmentation_filters import create_default_filter_stack
import numpy
import cv2

SEGMENT_DATATYPE = numpy.uint16
SEGMENT_SIZE = 4
SEGMENTS_DIRECTION = 0  # vertical axis in numpy


def segments_from_numpy(segments):
    """reverses segments_to_numpy"""
    segments = segments if SEGMENTS_DIRECTION == 0 else segments.tranpose()
    segments = [map(int, s) for s in segments]
    return segments


def segments_to_numpy(segments):
    """given a list of 4-element tuples, transforms it into a numpy array"""
    segments = numpy.array(segments, dtype=SEGMENT_DATATYPE, ndmin=2)  # each segment in a row
    segments = segments if SEGMENTS_DIRECTION == 0 else numpy.transpose(segments)
    return segments


def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    return image[y:y + h, x:x + w]


class RawSegmenter(DisplayingProcessor):
    """A image segmenter. input is image, output is segments"""

    def _segment(self, image):
        """segments an opencv image for OCR. returns list of 4-element tuples (x,y,width, height)."""
        # return segments
        raise NotImplementedError()

    def _process(self, image):
        segments = self._segment(image)
        self.image, self.segments = image, segments
        return segments


class FullSegmenter(DisplayingProcessorStack):
    pass


class RawContourSegmenter(RawSegmenter):
    PARAMETERS = RawSegmenter.PARAMETERS + {"block_size": 11, "c": 10}

    def _segment(self, image):
        self.image = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY, blockSize=self.block_size, C=self.c)
        if get_opencv_version() == 3:
            _, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        segments = segments_to_numpy([cv2.boundingRect(c) for c in contours])
        self.contours, self.hierarchy = contours, hierarchy  # store, may be needed for debugging
        return segments

    def display(self, display_before=False):
        copy = self.image.copy()
        if display_before:
            show_image_and_wait_for_key(copy, "image before segmentation")
        copy.fill(255)
        cv2.drawContours(copy, self.contours, contourIdx=-1, color=(0, 0, 0))
        show_image_and_wait_for_key(copy, "ContourSegmenter contours")
        copy = self.image.copy()
        draw_segments(copy, self.segments)
        show_image_and_wait_for_key(copy, "image after segmentation by " + self.__class__.__name__)


class ContourSegmenter(FullSegmenter):
    def __init__(self, **args):
        filters = create_default_filter_stack()
        stack = [BlurProcessor(), RawContourSegmenter()] + filters + [SegmentOrderer()]
        FullSegmenter.__init__(self, stack, **args)
        stack[0].add_prehook(create_broadcast("_input", filters, "image"))
