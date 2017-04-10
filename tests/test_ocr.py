import unittest
from segmentation import ContourSegmenter
from feature_extraction import SimpleFeatureExtractor
from files import ImageFile
from classification import KNNClassifier
from ocr import OCR, reconstruct_chars, SimpleOCR


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
        classes, segments = ocr.ocr(digits, show_steps=False)
        self.assertEqual(reconstruct_chars(classes), "31415926535897932384626433832795028841971693993751058209749445923"
                                                     "07816406286208998628034825342117067982148086513282306647093844609"
                                                     "55058223172535940812848111745028410270193852110555964462294895493"
                                                     "038196442881097566593344612847")

    def test_simple_ocr(self):
        simple = SimpleOCR()
        characters = ['9', '8', '2', '1', '4', '8', '0', '8', '6', '5', '1', '3', '2', '8',
                      '2', '3', '0', '6', '6', '4', '7', '0', '9', '3', '8', '4', '4', '6',
                      '0', '9', '5', '5', '0', '5', '8', '2', '2', '3', '1', '7', '2', '5',
                      '3', '5', '9', '4', '0', '8', '1', '2', '8', '4', '8', '1', '1', '1',
                      '7', '4', '5', '0', '2', '8', '4', '1', '0', '2', '7', '0', '1', '9',
                      '3', '8', '5', '2', '1', '1', '0', '5', '5', '5', '9', '6', '4', '4',
                      '6', '2', '2', '9', '4', '8', '9', '5', '4', '9', '3', '0', '3', '8',
                      '1', '9', '6', '4', '4', '2', '8', '8', '1', '0', '9', '7', '5', '6',
                      '6', '5', '9', '3', '3', '4', '4', '6', '1', '2', '8', '4', '7']
        simple.ground_file('digits1', characters)
        simple.train(ImageFile('digits1'))
        classes, segments = simple.ocr(ImageFile('digits2'), show_steps=False)
        self.assertEqual(reconstruct_chars(classes), "31415926535897932384626433832795028841971693993751058209749445923"
                                                     "07816406286208998628034825342117067982148086513282306647093844609"
                                                     "55058223172535940812848111745028410270193852110555964462294895493"
                                                     "038196442881097566593344612847")
        self.assertEqual(simple.ocr_chars(ImageFile('digits2')), "31415926535897932384626433832795028841971693993751058"
                                                                 "20974944592307816406286208998628034825342117067982148"
                                                                 "08651328230664709384460955058223172535940812848111745"
                                                                 "0284102701938521105559644622948954930381964428810975"
                                                                 "66593344612847")
