from classification import classes_from_numpy, classes_to_numpy
from segmentation import segments_from_numpy, segments_to_numpy
import sys
if sys.version_info[0] is not 3:
    # Import this only for Python 2
    from io import open


def read_boxfile(path):
    classes = []
    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.split(" ")
            assert len(s) == 6
            assert s[5] == '0\n'
            classes.append(s[0])  # .decode('utf-8'))
            segments.append(list(map(int, s[1:5])))
    return classes_to_numpy(classes), segments_to_numpy(segments)


def write_boxfile(path, classes, segments):
    classes, segments = classes_from_numpy(classes), segments_from_numpy(segments)
    with open(path, 'w') as f:
        for c, s in zip(classes, segments):
            f.write(c.encode('utf-8') + ' ' + ' '.join(map(str, s)) + " 0\n")
