from numpy_utils import OverflowPreventer
import numpy
import cv2

def ask_for_key():
    return cv2.waitKey(0)

def background_color( image ):
    return numpy.median(numpy.median(image, 0),0).astype( numpy.int )
    
def show_image_and_wait_for_key( image, name="Image" ):
    '''Shows an image, outputting name. keygroups is a dictionary of keycodes to functions; they are executed when the corresponding keycode is pressed'''
    print "showing",name,"(waiting for input)"
    cv2.imshow('norm',image)
    return ask_for_key()


def brightness( image, adjustment=0 ):
    '''changes image brightness. 
    An adjustment of -1 will make the image all black; 
    one of 1 will make the image all white'''
    assert image.dtype==numpy.uint8
    assert -1<=adjustment<=1
    with OverflowPreventer(image) as img:
        img+=adjustment*256

def contrast( image, scale=1, center=128 ):
    '''changes image contrast.
    a scale of 1 will make no changes'''
    assert image.dtype==numpy.uint8
    with OverflowPreventer(image) as img:
        if scale<=1:
            img*=scale
            img+= int(center*(1-scale))
        else:
            img-=center*(1 - 1/scale)
            img*=scale

def draw_segments( image , segments, color=(255,0,0), line_width=1):
        '''draws segments on image'''
        for segment in segments:
            x,y,w,h= segment
            cv2.rectangle(image,(x,y),(x+w,y+h),color,line_width)

