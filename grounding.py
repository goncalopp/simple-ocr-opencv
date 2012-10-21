'''various classis for establishing ground truth'''

from files import ImageFile
from classification import classes_to_numpy, BLANK_CLASS
from opencv_utils import background_color, show_image_and_wait_for_key, draw_segments, draw_classes
import numpy
import string


class Grounder( object ):
    def ground(self, imagefile, segments, external_data):
        '''given an ImageFile, grounds it, through arbirary data (better defined in subclasses)'''
        raise NotImplementedError()

class TextGrounder( Grounder ):
    '''labels from a string'''
    def ground( self, imagefile, segments, text ):
        '''tries to grounds from a simple string'''
        text= unicode( text )
        text= filter( lambda c: c in string.ascii_letters+string.digits, list(text))
        if len(segments)!=len(text):
            raise Exception( "segments/text length mismatch")
        classes= classes_to_numpy( text )
        imagefile.set_ground( segments, classes)

class UserGrounder( Grounder ):
    '''labels by interactively asking the user'''
    def ground( self, imagefile, segments, _=None ):
        '''asks the user to label each segment as either a character or "<" for unknown'''
        print '''For each shown segment, please write the character that it represents, or spacebar if it's not a character. Press ESC when completed, arrow keys to move'''
        i=0
        classes= [BLANK_CLASS]*len(segments) #char(10) is newline. it represents a non-assigned label, and will b filtered
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
        print "classified ",numpy.count_nonzero( classes != classes_to_numpy(BLANK_CLASS) ), "characters out of", max(classes.shape)
        imagefile.set_ground( segments, classes )
        
