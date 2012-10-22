from processor import Processor
import numpy
import cv2

class SegmentOrderer( Processor ):
    PARAMETERS= Processor.PARAMETERS + {"max_line_height":20, "max_line_width":10000}
    def _process( self, segments ):
        '''sort segments in read order - left to right, up to down'''
        #sort_f= lambda r: max_line_width*(r[1]/max_line_height)+r[0]
        #segments= sorted(segments, key=sort_f)
        #segments= segments_to_numpy( segments )
        #return segments
        mlh, mlw= self.max_line_height, self.max_line_width
        s= segments.astype( numpy.uint32 ) #prevent overflows
        order= mlw*(s[:,1]/mlh)+s[:,0]
        sort_order= numpy.argsort( order )
        return segments[ sort_order ]

def guess_line_starts( segments, asfloat=False ):
    ys= segments[:,1].astype(numpy.float32)
    l= _guess_lines( ys )
    return l if asfloat else map(int, l)

def guess_line_ends( segments, asfloat=False ):
    ys= numpy.sort((segments[:,1]+segments[:,3]).astype(numpy.float32))
    l= _guess_lines( ys )
    return l if asfloat else map(int, l)

def guess_line_starts_and_ends( segments, asfloat=False ):
    l1= guess_line_starts( segments, asfloat=True )
    l2= guess_line_ends( segments, asfloat=True )
    l= numpy.sort(numpy.append( l1, l2 ))
    return l if asfloat else map(int, l)
    
def guess_line_starts_ends_and_middles( segments, asfloat=False ):
    new_list=[]
    l= guess_line_starts_and_ends( segments, asfloat=True )
    for i,x in enumerate(l):
        new_list.append(x)
        if (i%2)==0:
            new_list.append( (l[i]+l[i+1])/2 )
    l= new_list
    return l if asfloat else map(int, l)



def _guess_lines( ys, max_lines=50, confidence_minimum=3 ):
    '''guesses and returns text inter-line distance, number of lines, y_position of first line'''
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
    lines= numpy.array( means_list[i] )
    
    #calculate confidence
    betterness= numpy.sort(deviations, axis=0)
    betterness= 1/(betterness[:-1]/betterness[1:]) #how much better is each solution compared to the next best?
    confidence= ( betterness[0] - numpy.mean(betterness) ) / numpy.std(betterness) #number of stddevs
    if confidence<confidence_minimum:
        raise Exception("low confidence")
    return lines #still floating points

def contained_segments_matrix( segments ):
    '''givens a n*n matrix m, n=len(segments), in which m[i,j] means
    segments[i] is contained inside segments[j]'''
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
    return a_inside_b
