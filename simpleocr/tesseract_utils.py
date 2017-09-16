from .classification import classes_from_numpy, classes_to_numpy
from .segmentation import segments_from_numpy, segments_to_numpy
import io


def read_boxfile(path):
    classes = []
    segments = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            s = line.split(" ")
            assert len(s) == 6
            assert s[5] == '0\n'
            classes.append(s[0])
            segments.append(list(map(int, s[1:5])))
    return classes_to_numpy(classes), segments_to_numpy(segments)


def write_boxfile(path, classes, segments):
    classes, segments = classes_from_numpy(classes), segments_from_numpy(segments)
    with io.open(path, 'w') as f:
        for c, s in zip(classes, segments):
            f.write(c + ' ' + ' '.join(map(str, s)) + " 0\n")
