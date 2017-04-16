import unittest

from segmentation import ContourSegmenter
from feature_extraction import SimpleFeatureExtractor
from files import ImageFile
from classification import KNNClassifier
from ocr import OCR, reconstruct_chars


class TestOCR(unittest.TestCase):
    def test_ocr_digits(self):
        segmenter = ContourSegmenter()
        self.assertTrue(segmenter)
        extractor = SimpleFeatureExtractor()
        self.assertTrue(extractor)
        classifier = KNNClassifier()
        self.assertTrue(classifier)
        ocr = OCR(segmenter, extractor, classifier)
        self.assertTrue(ocr)
        ocr.train(ImageFile('digits1'))
        digits = ImageFile('digits2')
        self.assertTrue(digits)
        chars, classes, segments = ocr.ocr(digits, show_steps=False)
        self.assertEqual(reconstruct_chars(classes), "31415926535897932384626433832795028841971693993751058209749445923"
                                                     "07816406286208998628034825342117067982148086513282306647093844609"
                                                     "55058223172535940812848111745028410270193852110555964462294895493"
                                                     "038196442881097566593344612847")
        self.assertEqual(chars, "31415926535897932384626433832795028841971693993751058209749445923"
                                "07816406286208998628034825342117067982148086513282306647093844609"
                                "55058223172535940812848111745028410270193852110555964462294895493"
                                "038196442881097566593344612847")
