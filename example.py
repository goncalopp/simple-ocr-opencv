from files import ImageFile
from segmentation import ContourSegmenter, draw_segments
from feature_extraction import SimpleFeatureExtractor
from classification import KNNClassifier
from ocr import OCR, accuracy, show_differences, reconstruct_chars

segmenter=  ContourSegmenter( blur_radius=5, blocksize=11, c=10, min_area=0, show_steps=True )
extractor=  SimpleFeatureExtractor( feature_size=10, stretch=False )
classifier= KNNClassifier()
ocr= OCR( segmenter, extractor, classifier )

ocr.train( ImageFile('digits1') )

test_image= ImageFile('digits2')
test_classes, test_segments= ocr.ocr( test_image )

print "accuracy:", accuracy( test_image.ground.classes, test_classes )
print "OCRed text:\n", reconstruct_chars( test_classes )
show_differences( test_image.image, test_segments, test_image.ground.classes, test_classes)
