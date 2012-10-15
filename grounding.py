'''various classis for establishing ground truth'''

from files import ImageFile
from classification import classes_to_numpy
from segmentation import best_segmenter
from opencv_utils import background_color, show_image_and_wait_for_key, draw_segments, draw_classes
import numpy
import string


class Grounder( object ):
    @staticmethod
    def _createSegments( imagefile ):
        assert isinstance( imagefile, ImageFile)
        segmenter= best_segmenter( imagefile.image )
        return segmenter.segment( imagefile.image )
    
    def ground(self, imagefile, external_data):
        '''given an ImageFile, grounds it, through arbirary data (better defined in subclasses)'''
        raise NotImplementedError()

class TextGrounder( Grounder ):
    '''labels from a string'''
    def ground( self, imagefile, text ):
        '''tries to grounds from a simple string'''
        segments= Grounder._createSegments( imagefile )
        text= unicode( text )
        text= filter( lambda c: c in string.ascii_letters+string.digits, list(text))
        if len(segments)!=len(text):
            raise Exception( "segments/text length mismatch")
        classes= classes_to_numpy( text )
        imagefile.set_ground( segments, classes)

class UserGrounder( Grounder ):
    '''labels by interactively asking the user'''
    def ground( self, imagefile, _=None ):
        '''asks the user to label each segment as either a character or "<" for unknown'''
        print '''For each shown segment, please write the character that it represents, or spacebar if it's not a character. Press ESC when completed, arrow keys to move'''
        segments= Grounder._createSegments( imagefile )
        i=0
        classes= [chr(10)]*len(segments) #char(10) is newline. it represents a non-assigned label, and will b filtered
        done= False
        allowed_chars= map( ord, string.ascii_letters+string.digits )
        while not done:
            image= imagefile.image.copy()
            draw_segments( image, [segments[ i ]])
            draw_classes( image, segments, classes )
            key= show_image_and_wait_for_key( image, "segment "+str(i))
            if key==27: #ESC
                break
            elif key==65470:  #<
                classes[i]=None
            elif key==65361: #<-
                i-=1
            elif key==65363: #->
                i+=1
            elif key in allowed_chars:
                classes[i]= unichr(key)
                i+=1
            if i>=len(classes):
                i=0
            if i<0:
                i=len(classes)-1
        classes= classes_to_numpy( classes )
        grounded= classes != 10 #indexes
        classified_n= numpy.count_nonzero( grounded )
        print "classified ",classified_n, "characters out of", max(classes.shape)
        classes= classes[ grounded ]    #filter ungrounded segments
        segments= segments [ grounded ] #from the arrays
        imagefile.set_ground( segments, classes )
        
