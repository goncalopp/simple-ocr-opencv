import unittest
import mock
from files import ImageFile
from grounding import TextGrounder, UserGrounder
from segmentation import ContourSegmenter
from ocr import reconstruct_chars


class TestGrounding(unittest.TestCase):
    def setUp(self):
        self.img = ImageFile('digits1')
        self.img.remove_ground()
        self.assertFalse(self.img.is_grounded())
        self.segments = ContourSegmenter().process(self.img.image)

    def test_textgrounder(self):
        grounder = TextGrounder()
        characters = "0" * len(self.segments)
        grounder.ground(self.img, self.segments, characters)
        self.assertTrue(self.img.is_grounded())
        self.assertEqual(reconstruct_chars(self.img.ground.classes), characters)

    def test_textgrounder_wronglen(self):
        grounder = TextGrounder()
        characters = "0" * len(self.segments)
        with self.assertRaises(ValueError):
            grounder.ground(self.img, self.segments, characters[:-4])
        self.assertFalse(self.img.is_grounded())

    def test_usergrounder(self):
        ESC_KEY = 27
        ZERO_KEY = 48
        keys = [ZERO_KEY]*len(self.segments) + [ESC_KEY]
        mock_generator = iter(keys)
        def mock_input(*args):
            return next(mock_generator)

        grounder = UserGrounder()
        with mock.patch('cv2.waitKey', mock_input):
            with mock.patch('cv2.imshow'):
                grounder.ground(self.img, self.segments)
        self.assertTrue(self.img.is_grounded())
        self.assertEqual(reconstruct_chars(self.img.ground.classes), "0"*len(self.segments))
