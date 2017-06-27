from .files import ImageFile
from .grounding import UserGrounder
from .segmentation import ContourSegmenter, draw_segments

segmenter = ContourSegmenter(blur_y=5, blur_x=5, block_size=11, c=10)
new_image = ImageFile('digits1')
segments = segmenter.process(new_image.image)

grounder = UserGrounder()
grounder.ground(new_image, segments)
new_image.ground.write()
