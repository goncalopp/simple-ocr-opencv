from simpleocr.files import open_img
from simpleocr.grounding import UserGrounder
from simpleocr.segmentation import ContourSegmenter, draw_segments

segmenter = ContourSegmenter(blur_y=5, blur_x=5, block_size=11, c=10)
new_image = open_img('digits1')
segments = segmenter.process(new_image.image)

grounder = UserGrounder()
grounder.ground(new_image, segments)
new_image.ground.write()
