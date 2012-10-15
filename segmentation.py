from opencv_utils import show_image_and_wait_for_key, brightness, draw_segments
import numpy
import cv2

SEGMENT_DATATYPE=   numpy.uint16
SEGMENT_SIZE=       4
SEGMENTS_DIRECTION= 0 # vertical axis in numpy

def segments_from_numpy( segments ):
    '''reverses segments_to_numpy'''
    segments= segments if SEGMENTS_DIRECTION==0 else segments.tranpose()
    segments= [map(int,s) for s in segments]
    return segments

def segments_to_numpy( segments ):
    '''given a list of 4-element tuples, transforms it into a numpy array'''
    segments= numpy.array( segments, dtype=SEGMENT_DATATYPE, ndmin=2)   #each segment in a row
    segments= segments if SEGMENTS_DIRECTION==0 else numpy.transpose(segments)
    return segments

def best_segmenter(image):
    '''returns a segmenter instance which segments the given image well'''
    return ContourSegmenter()

def region_from_segment( image, segment ):
    '''given a segment (rectangle) and an image, returns it's corresponding subimage'''
    x,y,w,h= segment
    return image[y:y+h,x:x+w]

def order_segments( segments, max_line_height=20, max_line_width=10000 ):
    '''sort segments in read order - left to right, up to down'''
    #sort_f= lambda r: max_line_width*(r[1]/max_line_height)+r[0]
    #segments= sorted(segments, key=sort_f)
    #segments= segments_to_numpy( segments )
    #return segments
    s= segments.astype( numpy.uint32 ) #prevent overflows
    order= max_line_width*(s[:,1]/max_line_height)+s[:,0]
    sort_order= numpy.argsort( order )
    return segments[ sort_order ]
    
class Segmenter( object ):
    '''A image segmenter. Finds rectangular sections in an image'''    
    def segment( self, image ):
        '''segments an opencv image for OCR. returns list of 4-element tuples (x,y,width, height).'''
        #return segments
        raise NotImplementedError()

class ContourSegmenter( Segmenter ):
    def __init__(self, blur_radius=5, blocksize=11, c=10, min_area=0, tolerance=0.1, show_steps=False):
        '''blur_radius is for gaussian blur preprocessing.
        blocksize is for adaptiveThresold. 
        c is the thresold in adaptiveThreshold.
        min_area is the minimum area of the returned segments (of non-zero pixels, not rectangle area'''
        self.blur_radius=   blur_radius
        self.blocksize=     blocksize
        self.c=             c
        self.min_area=      min_area
        self.tolerance=     tolerance
        self.show_steps=    show_steps

    def _contour_segment( self, image ):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if self.blur_radius:
            br=self.blur_radius
            image = cv2.GaussianBlur(image,(br,br),0)
        image = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=self.blocksize, C=self.c)
        contours,hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        segments= segments_to_numpy( [cv2.boundingRect(c) for c in contours] )
        return segments, contours, hierarchy
    
    @staticmethod
    def _show_steps(image, contours, good_segments, bad_segments):
        copy= image.copy()
        copy.fill( (255,255,255) )
        cv2.drawContours(copy, contours, contourIdx=-1, color=(0,0,0))
        show_image_and_wait_for_key( copy, "contours")
        copy= image.copy()
        brightness(copy, 0.8)
        draw_segments( copy, good_segments, (0,255,0) )
        draw_segments( copy, bad_segments, (0,0,255) )
        show_image_and_wait_for_key( copy, "filtered segments")
        
    @staticmethod
    def _filter_segments( segments, tolerance=0.1 ):
        '''performs some statistics to remove outliers. returns filtered and bad segments. tolerance is difference of ratio of median to accept'''
        _, _, median_width, median_height= numpy.median(segments, 0)
        bad_height= numpy.absolute(segments[:,3]-median_height)/median_height>tolerance
        bad= bad_height
        good_segments, bad_segments= (segments[numpy.logical_not(bad)], segments[bad])
        return good_segments, bad_segments

    def segment( self, image ):
        segments, contours, hierarchy= self._contour_segment( image )
        segments= order_segments( segments )
        good_segments, bad_segments= ContourSegmenter._filter_segments( segments, self.tolerance)
        if self.show_steps:
            ContourSegmenter._show_steps( image, contours, good_segments, bad_segments )
        return good_segments
