import unittest
import mock
from files import ImageFile
from grounding import TextGrounder, TerminalGrounder, UserGrounder
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
        self.assertEquals(reconstruct_chars(self.img.ground.classes), characters)

    def test_textgrounder_wrong_len(self):
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
        self.assertEquals(reconstruct_chars(self.img.ground.classes), "0"*len(self.segments))

    def test_terminal_grounder(self):
        terminal = TerminalGrounder()
        segmenter = ContourSegmenter()
        image = ImageFile('digits1')
        segments = segmenter.process(image.image)
        characters = ['9', '8', '2', '1', '4', '8', '0', '8', '6', '5', '1', '3', '2', '8',
                      '2', '3', '0', '6', '6', '4', '7', '0', '9', '3', '8', '4', '4', '6',
                      '0', '9', '5', '5', '0', '5', '8', '2', '2', '3', '1', '7', '2', '5',
                      '3', '5', '9', '4', '0', '8', '1', '2', '8', '4', '8', '1', '1', '1',
                      '7', '4', '5', '0', '2', '8', '4', '1', '0', '2', '7', '0', '1', '9',
                      '3', '8', '5', '2', '1', '1', '0', '5', '5', '5', '9', '6', '4', '4',
                      '6', '2', '2', '9', '4', '8', '9', '5', '4', '9', '3', '0', '3', '8',
                      '1', '9', '6', '4', '4', '2', '8', '8', '1', '0', '9', '7', '5', '6',
                      '6', '5', '9', '3', '3', '4', '4', '6', '1', '2', '8', '4', '7']
        mock_input_gen = (char for char in characters)
        def mock_input(prompt):
            return next(mock_input_gen)
        with mock.patch('__builtin__.raw_input', mock_input):
            terminal.ground(image, segments)

