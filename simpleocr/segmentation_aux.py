from .processor import Processor, DisplayingProcessor
from .opencv_utils import draw_lines, show_image_and_wait_for_key
import numpy
import cv2
from functools import reduce


class SegmentOrderer(Processor):
    PARAMETERS = Processor.PARAMETERS + {"max_line_height": 20, "max_line_width": 10000}

    def _process(self, segments):
        """sort segments in read order - left to right, up to down"""
        # sort_f= lambda r: max_line_width*(r[1]/max_line_height)+r[0]
        # segments= sorted(segments, key=sort_f)
        # segments= segments_to_numpy( segments )
        # return segments
        mlh, mlw = self.max_line_height, self.max_line_width
        s = segments.astype(numpy.uint32)  # prevent overflows
        order = mlw * (s[:, 1] // mlh) + s[:, 0]
        sort_order = numpy.argsort(order)
        return segments[sort_order]


class LineFinder(DisplayingProcessor):
    @staticmethod
    def _guess_lines(ys, max_lines=50, confidence_minimum=0.0):
        """guesses and returns text inter-line distance, number of lines, y_position of first line"""
        ys = ys.astype(numpy.float32)
        compactness_list, means_list, diffs, deviations = [], [], [], []
        start_n = 1
        for k in range(start_n, min(len(ys), max_lines)):
            compactness, classified_points, means = cv2.kmeans(data=ys, K=k, bestLabels=None, criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=2, flags=cv2.KMEANS_PP_CENTERS)
            means = numpy.sort(means, axis=0)
            means_list.append(means)
            compactness_list.append(compactness)
            if k < 3:
                tmp1 = [1, 2, 500, 550]  # forge data for bad clusters
            else:
                # calculate the center of each cluster. Assuming lines are equally spaced...
                tmp1 = numpy.diff(means, axis=0)  # diff will be equal or very similar
            tmp2 = numpy.std(tmp1) / numpy.mean(means)  # so variance is minimal
            tmp3 = numpy.sum((tmp1 - numpy.mean(tmp1)) ** 2)  # root mean square deviation, more sensitive than std
            diffs.append(tmp1)
            deviations.append(tmp3)

        compactness_list = numpy.diff(
            numpy.log(numpy.array(compactness_list) + 0.01))  # sum small amount to avoid log(0)
        deviations = numpy.array(deviations[1:])
        deviations[0] = numpy.mean(deviations[1:])
        compactness_list = (compactness_list - numpy.mean(compactness_list)) / numpy.std(compactness_list)
        deviations = (deviations - numpy.mean(deviations)) / numpy.std(deviations)
        aglomerated_metric = 0.1 * compactness_list + 0.9 * deviations

        i = numpy.argmin(aglomerated_metric) + 1
        lines = means_list[i]

        # calculate confidence
        betterness = numpy.sort(aglomerated_metric, axis=0)
        confidence = (betterness[1] - betterness[0]) / (betterness[2] - betterness[1])
        if confidence < confidence_minimum:
            raise Exception("low confidence")
        return lines  # still floating points

    def _process(self, segments):
        segment_tops = segments[:, 1]
        segment_bottoms = segment_tops + segments[:, 3]
        tops = self._guess_lines(segment_tops)
        bottoms = self._guess_lines(segment_bottoms)
        if len(tops) != len(bottoms):
            raise Exception("different number of lines")
        middles = (tops + bottoms) / 2
        topbottoms = numpy.sort(numpy.append(tops, bottoms))
        topmiddlebottoms = numpy.sort(reduce(numpy.append, (tops, middles, bottoms)))
        self.lines_tops = tops
        self.lines_bottoms = bottoms
        self.lines_topbottoms = topbottoms
        self.lines_topmiddlebottoms = topmiddlebottoms
        return segments

    def display(self, display_before=False):
        copy = self.image.copy()
        draw_lines(copy, self.lines_tops, (0, 0, 255))
        draw_lines(copy, self.lines_bottoms, (0, 255, 0))
        show_image_and_wait_for_key(copy, "line starts and ends")


def guess_segments_lines(segments, lines, nearline_tolerance=5.0):
    """
    given segments, outputs a array of line numbers, or -1 if it
    doesn't belong to any
    """
    ys = segments[:, 1]
    closeness = numpy.abs(numpy.subtract.outer(ys, lines))  # each row a y, each collumn a distance to each line
    line_of_y = numpy.argmin(closeness, axis=1)
    distance = numpy.min(closeness, axis=1)
    bad = distance > numpy.mean(distance) + nearline_tolerance * numpy.std(distance)
    line_of_y[bad] = -1
    return line_of_y


def contained_segments_matrix(segments):
    """
    givens a n*n matrix m, n=len(segments), in which m[i,j] means
    segments[i] is contained inside segments[j]
    """
    x1, y1 = segments[:, 0], segments[:, 1]
    x2, y2 = x1 + segments[:, 2], y1 + segments[:, 3]
    n = len(segments)

    x1so, x2so, y1so, y2so = list(map(numpy.argsort, (x1, x2, y1, y2)))
    x1soi, x2soi, y1soi, y2soi = list(map(numpy.argsort, (x1so, x2so, y1so, y2so)))  # inverse transformations
    # let rows be x1 and collumns be x2. this array represents where x1<x2
    o1 = numpy.triu(numpy.ones((n, n)), k=1).astype(bool)
    # let rows be x1 and collumns be x2. this array represents where x1>x2
    o2 = numpy.tril(numpy.ones((n, n)), k=0).astype(bool)
    a_inside_b_x = o2[x1soi][:, x1soi] * o1[x2soi][:, x2soi]  # (x1[a]>x1[b] and x2[a]<x2[b])
    a_inside_b_y = o2[y1soi][:, y1soi] * o1[y2soi][:, y2soi]  # (y1[a]>y1[b] and y2[a]<y2[b])
    a_inside_b = a_inside_b_x * a_inside_b_y
    return a_inside_b
