from simpleocr.files import open_image
from simpleocr.segmentation import ContourSegmenter
from simpleocr.feature_extraction import SimpleFeatureExtractor
from simpleocr.classification import KNNClassifier
from simpleocr.ocr import OCR, accuracy, show_differences

segmenter = ContourSegmenter(blur_y=5, blur_x=5, block_size=11, c=10)
extractor = SimpleFeatureExtractor(feature_size=10, stretch=False)
classifier = KNNClassifier()
ocr = OCR(segmenter, extractor, classifier)

ocr.train(open_image('digits1'))

test_image = open_image('digits2')
test_chars, test_classes, test_segments = ocr.ocr(test_image, show_steps=True)

print("accuracy:", accuracy(test_image.ground.classes, test_classes))
print("OCRed text:\n", test_chars)
