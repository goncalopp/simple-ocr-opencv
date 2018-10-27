from .opencv_utils import show_image_and_wait_for_key, BrightnessProcessor, draw_segments
from .segmentation_aux import contained_segments_matrix, LineFinder, guess_segments_lines
from .processor import DisplayingProcessor, create_broadcast
import numpy


def create_default_filter_stack():
    stack = [LargeFilter(), SmallFilter(), LargeAreaFilter(), ContainedFilter(), LineFinder(), NearLineFilter()]
    stack[4].add_poshook(create_broadcast("lines_topmiddlebottoms", stack[5]))
    return stack


class Filter(DisplayingProcessor):
    """A filter processes given segments, returning only the desirable ones"""

    PARAMETERS = DisplayingProcessor.PARAMETERS

    def display(self, display_before=False):
        """shows the effect of this filter"""
        try:
            copy = self.image.copy()
        except AttributeError:
            raise Exception("You need to set the Filter.image attribute for displaying")
        copy = BrightnessProcessor(brightness=0.6).process(copy)
        s, g = self._input, self.good_segments_indexes
        draw_segments(copy, s[g], (0, 255, 0))
        draw_segments(copy, s[True ^ g], (0, 0, 255))
        show_image_and_wait_for_key(copy, "segments filtered by " + self.__class__.__name__)

    def _good_segments(self, segments):
        raise NotImplementedError

    def _process(self, segments):
        good = self._good_segments(segments)
        self.good_segments_indexes = good
        segments = segments[good]
        if not len(segments):
            raise Exception("0 segments after filter " + self.__class__.__name__)
        return segments


class LargeFilter(Filter):
    """desirable segments are larger than some width or height"""
    PARAMETERS = Filter.PARAMETERS + {"min_width": 4, "min_height": 8}

    def _good_segments(self, segments):
        good_width = segments[:, 2] >= self.min_width
        good_height = segments[:, 3] >= self.min_height
        return good_width * good_height  # AND


class SmallFilter(Filter):
    """desirable segments are smaller than some width or height"""
    PARAMETERS = Filter.PARAMETERS + {"max_width": 30, "max_height": 50}

    def _good_segments(self, segments):
        good_width = segments[:, 2] <= self.max_width
        good_height = segments[:, 3] <= self.max_height
        return good_width * good_height  # AND


class LargeAreaFilter(Filter):
    """desirable segments' area is larger than some"""
    PARAMETERS = Filter.PARAMETERS + {"min_area": 45}

    def _good_segments(self, segments):
        return (segments[:, 2] * segments[:, 3]) >= self.min_area


class ContainedFilter(Filter):
    """desirable segments are not contained by any other"""

    def _good_segments(self, segments):
        m = contained_segments_matrix(segments)
        return True ^ numpy.max(m, axis=1)


class NearLineFilter(Filter):
    PARAMETERS = Filter.PARAMETERS + {"nearline_tolerance": 5.0}  # percentage distance stddev
    '''desirable segments have their y near a line'''

    def _good_segments(self, segments):
        lines = guess_segments_lines(segments, self.lines_topmiddlebottoms, nearline_tolerance=self.nearline_tolerance)
        good = lines != -1
        return good
