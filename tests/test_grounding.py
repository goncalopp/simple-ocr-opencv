import unittest
from files import ImageFile
from grounding import TextGrounder, UserGrounder
from segmentation import ContourSegmenter
from feature_extraction import SimpleFeatureExtractor
from classification import KNNClassifier
from ocr import OCR, reconstruct_chars
import mock


class TestGrounding(unittest.TestCase):
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

    def test_user_grounder(self):
        characters = [57, 56, 50, 49, 52, 56, 48, 56, 54, 53, 49, 51, 50, 56, 50, 51, 48, 54, 54, 52, 55, 48, 57, 51,
                      56, 52, 52, 54, 48, 57, 53, 53, 48, 53, 56, 50, 50, 51, 49, 55, 50, 53, 51, 53, 57, 52, 48, 56,
                      49, 50, 56, 52, 56, 49, 49, 49, 55, 52, 53, 48, 50, 56, 52, 49, 48, 50, 55, 48, 49, 57, 51, 56,
                      53, 50, 49, 49, 48, 53, 53, 53, 57, 54, 52, 52, 54, 50, 50, 57, 52, 56, 57, 53, 52, 57, 51, 48,
                      51, 56, 49, 57, 54, 52, 52, 50, 56, 56, 49, 48, 57, 55, 53, 54, 54, 53, 57, 51, 51, 52, 52, 54,
                      49, 50, 56, 52, 55, 27]
        mock_generator = (char for char in characters)

        def mock_input(*args):
            return next(mock_generator)
        grounder = UserGrounder()
        segmenter = ContourSegmenter()
        image = ImageFile('digits1')
        segments = segmenter.process(image.image)
        with mock.patch('cv2.waitKey', mock_input):
            with mock.patch('cv2.imshow'):
                grounder.ground(image, segments)
        extractor = SimpleFeatureExtractor()
        classifier = KNNClassifier()
        ocr = OCR(segmenter, extractor, classifier)
        ocr.train(ImageFile('digits1'))
        digits = ImageFile('digits2')
        classes, segments = ocr.ocr(digits, show_steps=False)
        self.assertEqual(reconstruct_chars(classes), "31415926535897932384626433832795028841971693993751058209749445923"
                                                     "07816406286208998628034825342117067982148086513282306647093844609"
                                                     "55058223172535940812848111745028410270193852110555964462294895493"
                                                     "038196442881097566593344612847")
