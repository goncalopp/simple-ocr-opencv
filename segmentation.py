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

def _order_segments( segments, max_line_height=20, max_line_width=10000 ):
    '''sort segments in read order - left to right, up to down'''
    #sort_f= lambda r: max_line_width*(r[1]/max_line_height)+r[0]
    #segments= sorted(segments, key=sort_f)
    #segments= segments_to_numpy( segments )
    #return segments
    s= segments.astype( numpy.uint32 ) #prevent overflows
    order= max_line_width*(s[:,1]/max_line_height)+s[:,0]
    sort_order= numpy.argsort( order )
    return segments[ sort_order ]

def _filter_small_segments( segments, min_width=8, min_height=10 ):
    '''returns boolean array marking segments too small to be correct'''
    small_width= segments[:,2]<min_width
    small_height= segments[:,3]<min_height
    return small_width + small_height

def _filter_large_segments( segments, max_width=30, max_height=50 ):
    '''returns boolean array marking segments too large to be correct'''
    large_width= segments[:,2]>max_width
    large_height= segments[:,3]>max_height
    return large_width + large_height #bool array

def bool_indexing_to_indexes( bool_array ):
    return [i for i,x in enumerate(bool_array) if x]

def _filter_contained_segments( segments ):
    '''returns boolean array marking segments that are contained by some other'''
    segments= segments.astype( numpy.float)
    x1,y1= segments[:,0], segments[:,1]
    x2,y2= x1+segments[:,2], y1+segments[:,3]
    n=len(segments)
    
    x1so, x2so,y1so, y2so= map(numpy.argsort, (x1,x2,y1,y2))
    x1soi,x2soi, y1soi, y2soi= map(numpy.argsort, (x1so, x2so, y1so, y2so)) #inverse transformations
    o1= numpy.triu(numpy.ones( (n,n) ), k=1).astype(bool) # let rows be x1 and collumns be x2. this array represents where x1<x2
    o2= numpy.tril(numpy.ones( (n,n) ), k=0).astype(bool) # let rows be x1 and collumns be x2. this array represents where x1>x2
    
    a_inside_b_x= o2[x1soi][:,x1soi] * o1[x2soi][:,x2soi] #(x1[a]>x1[b] and x2[a]<x2[b])
    a_inside_b_y= o2[y1soi][:,y1soi] * o1[y2soi][:,y2soi] #(y1[a]>y1[b] and y2[a]<y2[b])
    a_inside_b= a_inside_b_x*a_inside_b_y
    bad= numpy.max(a_inside_b, axis=1)
    return bad


def _guess_interline_size( segments, max_lines=50, confidence1_minimum=1.5, confidence2_minimum=3 ):
    '''guesses and returns text inter-line distance, number of lines, y_position of first line'''
    ys= segments[:,1].astype(numpy.float32)
    
    means_list, diffs, deviations=[], [], []
    start_n= 3
    for k in range(start_n,max_lines):
        temp, classified_points, means = cv2.kmeans( data=ys, K=k, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, flags=cv2.KMEANS_PP_CENTERS)
        means=numpy.sort(means, axis=0)
        #calculate the center of each cluster. Assuming lines are equally spaced...
        tmp1=numpy.diff(means, axis=0) #diff will be equal or very similar
        tmp2= numpy.std(tmp1)/numpy.mean(means) #so variance is minimal
        tmp3= numpy.sum( (tmp1-numpy.mean(tmp1))**2) #root mean square deviation, more sensitive than std
        means_list.append( means )
        diffs.append(tmp1)
        deviations.append(tmp3)
    
    i= deviations.index(min(deviations))
    number_of_lines=  i+start_n
    inter_line_distance= numpy.mean(diffs[i])
    first_line= means_list[i][0][0]
    
    #calculate confidence
    betterness= numpy.sort(deviations, axis=0)
    betterness= 1/(betterness[:-1]/betterness[1:]) #how much better is each solution compared to the next best?
    confidence= ( betterness[0] - numpy.mean(betterness) ) / numpy.std(betterness) #number of stddevs
    if confidence<3:
        raise Exception("low confidence")
    return inter_line_distance, number_of_lines, first_line
    
class Segmenter( object ):
    '''A image segmenter. Finds rectangular sections in an image'''    
    def segment( self, image ):
        '''segments an opencv image for OCR. returns list of 4-element tuples (x,y,width, height).'''
        #return segments
        raise NotImplementedError()

class ContourSegmenter( Segmenter ):
    FILTERS= [_filter_large_segments, _filter_small_segments, _filter_contained_segments]
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
    def _show_steps(image, contours, goodbad_list):
        copy= image.copy()
        copy.fill( (255,255,255) )
        cv2.drawContours(copy, contours, contourIdx=-1, color=(0,0,0))
        show_image_and_wait_for_key( copy, "contours")
        
        for f_name, good, bad in goodbad_list:
            copy= image.copy()
            brightness(copy, 0.8)
            draw_segments( copy, good, (0,255,0) )
            draw_segments( copy, bad, (0,0,255) )
            show_image_and_wait_for_key( copy, "segments filtered by "+f_name)


    def segment( self, image ):
        segments, contours, hierarchy= self._contour_segment( image )
         
        goodbad_list= [] #list of tuples of (filter_name, good_segments, bad_segments) after application of each filter
        for filter_f in self.FILTERS:
            bad_indexes= filter_f(segments)
            segments, bad_segments= segments[ True - bad_indexes ], segments[ bad_indexes ]
            goodbad_list.append( (filter_f.__name__, segments, bad_segments) )
            if len(segments)==0:
                raise Exception("0 good segments after filter"+filter_f.__name__+". Cannot proceed")
        inter_line_distance, number_of_lines, first_line= _guess_interline_size( segments )
        segments= _order_segments( segments )
        
        if self.show_steps:
            ContourSegmenter._show_steps( image, contours, goodbad_list )
        return segments
