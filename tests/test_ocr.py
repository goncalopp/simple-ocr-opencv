import unittest
from simpleocr.segmentation import ContourSegmenter
from simpleocr.feature_extraction import SimpleFeatureExtractor
from simpleocr.files import open_img
from simpleocr.classification import KNNClassifier
from simpleocr.ocr import OCR, reconstruct_chars


class TestOCR(unittest.TestCase):
    def _test_ocr(self, train_file, test_file):
        # get data from images
        ground_truth = test_file.ground.classes
        test_file.remove_ground()
        # create OCR
        segmenter = ContourSegmenter(blur_y=5, blur_x=5)
        extractor = SimpleFeatureExtractor()
        classifier = KNNClassifier()
        ocr = OCR(segmenter, extractor, classifier)
        # train and test
        ocr.train(train_file)
        chars, classes, _ = ocr.ocr(test_file, show_steps=False)
        print(chars)
        print(reconstruct_chars(ground_truth))
        self.assertEqual(chars, reconstruct_chars(ground_truth))
        self.assertEqual(list(classes), list(ground_truth))

    def test_ocr_digits(self):
        self._test_ocr(open_img('digits1'), open_img('digits2'))

    def test_ocr_unicode(self):
        self._test_ocr(open_img('unicode1'), open_img('unicode1'))
