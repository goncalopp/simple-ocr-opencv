import unittest

from segmentation import ContourSegmenter
from feature_extraction import SimpleFeatureExtractor
from files import ImageFile
from classification import KNNClassifier
from ocr import OCR, reconstruct_chars


class TestOCR(unittest.TestCase):
    def test_ocr_digits(self):
        # get data from images
        img1 = ImageFile('digits1')
        img2 = ImageFile('digits2')
        ground_truth = img2.ground.classes
        img2.remove_ground()
        # create OCR
        segmenter = ContourSegmenter()
        extractor = SimpleFeatureExtractor()
        classifier = KNNClassifier()
        ocr = OCR(segmenter, extractor, classifier)
        # train and test
        ocr.train(img1)
        chars, classes, _ = ocr.ocr(img2, show_steps=False)
        self.assertEqual(list(classes), list(ground_truth))
        self.assertEqual(chars, reconstruct_chars(ground_truth))
