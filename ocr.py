from opencv_utils import show_image_and_wait_for_key, draw_segments
import numpy
import cv2

def show_differences( image, segments, ground_classes, result_classes):
    image= image.copy()
    good= (ground_classes==result_classes)
    good.shape= (len(good),) #transform nx1 matrix into vector
    draw_segments( image, segments[good,:], (0,255,0) )
    draw_segments( image, segments[numpy.logical_not(good),:], (0,0,255)  )   
    show_image_and_wait_for_key(image, "differences")


def reconstruct_chars( classes ):
    result_string= "".join(map(unichr, classes))
    return result_string

def accuracy( expected, result ):
    if( expected.shape!=result.shape ):
        raise Exception("expected "+str(expected.shape)+", got "+str(result.shape))
    correct= expected==result
    return float(numpy.count_nonzero(correct))/correct.shape[0]


class OCR( object ):
    def __init__( self, segmenter, feature_extractor, classifier):
        self.segmenter= segmenter
        self.feature_extractor= feature_extractor
        self.classifier= classifier

    def train( self, image_file ):
        '''feeds the training data to the OCR'''
        if not image_file.isGrounded():
            raise Exception("The provided file is not grounded")
        features= self.feature_extractor.extract( image_file.image, image_file.ground.segments )
        self.classifier.train( features, image_file.ground.classes )
        
    def ocr( self, image_file, show_steps=False ):
        '''performs ocr used trained classifier'''
        segments= self.segmenter.process( image_file.image )
        if show_steps:
            self.segmenter.display()
        features= self.feature_extractor.extract( image_file.image , segments )
        classes= self.classifier.classify( features )
        return classes, segments
