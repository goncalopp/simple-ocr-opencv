"""various classes for establishing ground truth"""

from .classification import classes_to_numpy, classes_from_numpy, BLANK_CLASS
from .opencv_utils import show_image_and_wait_for_key, draw_segments, draw_classes
import numpy
import string
from six import text_type, unichr, moves

NOT_A_SEGMENT = unichr(10)


class Grounder(object):
    def ground(self, imagefile, segments, external_data):
        """given an ImageFile, grounds it, through arbitrary data (better defined in subclasses)"""
        raise NotImplementedError()


class TerminalGrounder(Grounder):
    """
    Labels by using raw_input() to capture a character each line
    """

    def ground(self, imagefile, segments, _=None):
        classes = []
        character = ""
        print("Found %s segments to ground." % len(segments))
        print("Type 'exit' to stop grounding the file.")
        print("Type ' ' for anything that is not a character.")
        print("Grounding will exit automatically after all segments.")
        print("Going back to a previous segment is not possible at this time.")
        for num in range(len(segments)):
            while len(character) != 1:
                character = moves.input("Please enter the value for segment #%s:  " % (num+1))
                if character == "exit":
                    break
                if len(character) != 1:
                    print("That is not a single character. Please try again.")
            if character == " ":
                classes.append(NOT_A_SEGMENT)
            else:
                classes.append(character)
            character = ""
        classes = classes_to_numpy(classes)
        imagefile.set_ground(segments, classes)


class TextGrounder(Grounder):
    """labels from a string"""

    def ground(self, imagefile, segments, text):
        """tries to grounds from a simple string"""
        text = text_type(text)
        text = [c for c in text if c in string.ascii_letters + string.digits]
        if len(segments) != len(text):
            raise ValueError("segments/text length mismatch")
        classes = classes_to_numpy(text)
        imagefile.set_ground(segments, classes)


class UserGrounder(Grounder):
    """labels by interactively asking the user"""

    def ground(self, imagefile, segments, _=None):
        """asks the user to label each segment as either a character or "<" for unknown"""
        print("For each shown segment, please write the character that it represents, or spacebar if it's not a "
              "character. To undo a classification, press backspace. Press ESC when completed, arrow keys to move")
        i = 0
        if imagefile.is_grounded:
            classes = classes_from_numpy(imagefile.ground.classes)
            segments = imagefile.ground.segments
        else:
            classes = [BLANK_CLASS] * len(segments)
        done = False
        allowed_chars = list(map(ord, string.digits + string.ascii_letters + string.punctuation))
        while not done:
            image = imagefile.image.copy()
            draw_segments(image, [segments[i]])
            draw_classes(image, segments, classes)
            key = show_image_and_wait_for_key(image, "segment " + str(i))
            if key == 27:  # ESC
                break
            elif key == 8:  # backspace
                classes[i] = BLANK_CLASS
                i += 1
            elif key == 32:  # space
                classes[i] = NOT_A_SEGMENT
                i += 1
            elif key in (81, 65361):  # <-
                i -= 1
            elif key in (83, 65363):  # ->
                i += 1
            elif key in allowed_chars:
                classes[i] = unichr(key)
                i += 1
            if i >= len(classes):
                i = 0
            if i < 0:
                i = len(classes) - 1

        classes = numpy.array(classes)
        is_segment = classes != NOT_A_SEGMENT
        classes = classes[is_segment]
        segments = segments[is_segment]
        classes = list(classes)

        classes = classes_to_numpy(classes)
        print("classified ", numpy.count_nonzero(classes != classes_to_numpy(BLANK_CLASS)), "characters out of", max(
            classes.shape))
        imagefile.set_ground(segments, classes)
