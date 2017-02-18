import unittest
from files import ImageFile
from grounding import TextGrounder
from segmentation import ContourSegmenter
from feature_extraction import SimpleFeatureExtractor
from classification import KNNClassifier
from ocr import OCR, reconstruct_chars


class Testing(unittest.TestCase):
    def test_grounding_digits(self):
        grounder = TextGrounder()
        self.assertTrue(grounder)
        digits = ImageFile('digits1')
        self.assertTrue(digits)
        segmenter = ContourSegmenter()
        self.assertTrue(segmenter)
        segments = segmenter.process(digits.image)
        # self.assertTrue(segments)
        characters = ['9', '8', '2', '1', '4', '8', '0', '8', '6', '5', '1', '3', '2', '8',
                      '2', '3', '0', '6', '6', '4', '7', '0', '9', '3', '8', '4', '4', '6',
                      '0', '9', '5', '5', '0', '5', '8', '2', '2', '3', '1', '7', '2', '5',
                      '3', '5', '9', '4', '0', '8', '1', '2', '8', '4', '8', '1', '1', '1',
                      '7', '4', '5', '0', '2', '8', '4', '1', '0', '2', '7', '0', '1', '9',
                      '3', '8', '5', '2', '1', '1', '0', '5', '5', '5', '9', '6', '4', '4',
                      '6', '2', '2', '9', '4', '8', '9', '5', '4', '9', '3', '0', '3', '8',
                      '1', '9', '6', '4', '4', '2', '8', '8', '1', '0', '9', '7', '5', '6',
                      '6', '5', '9', '3', '3', '4', '4', '6', '1', '2', '8', '4', '7']
        grounder.ground(digits, segments, characters)
        self.assertTrue(digits.is_grounded())

    def test_grounding_raise(self):
        grounder = TextGrounder()
        digits = ImageFile('digits1')
        segmenter = ContourSegmenter()
        segments = segmenter.process(digits.image)
        characters = ['9', '8', '2', '1', '4', '8', '0', '8', '6', '5', '1', '3', '2', '8',
                      '2', '3', '0', '6', '6', '4', '7', '0', '9', '3', '8', '4', '4', '6',
                      '0', '9', '5', '5', '0', '5', '8', '2', '2', '3', '1', '7', '2', '5',
                      '3', '5', '9', '4', '0', '8', '1', '2', '8', '4', '8', '1', '1', '1',
                      '7', '4', '5', '0', '2', '8', '4', '1', '0', '2', '7', '0', '1', '9',
                      '3', '8', '5', '2', '1', '1', '0', '5', '5', '5', '9', '6', '4', '4',
                      '6', '2', '2', '9', '4', '8', '9', '5', '4', '9', '3', '0', '3', '8',
                      '1', '9', '6', '4', '4', '2', '8', '8', '1', '0', '9', '7', '5', '6',
                      '6', '5', '9', '3', '3', '4', '4', '6', '1', '2', '8', '4', '7', '2']
        self.assertRaises(ValueError, lambda: grounder.ground(digits, segments, characters[:-4]))
        self.assertRaises(ValueError, lambda: grounder.ground(digits, segments, characters))

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

unittest.main()

