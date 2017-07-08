from .files import Image
from .grounding import UserGrounder
from .segmentation import ContourSegmenter

segmenter = ContourSegmenter(blur_y=5, blur_x=5, block_size=11, c=10)
new_image = Image.from_file('digits1')
segments = segmenter.process(new_image.image)

grounder = UserGrounder()
grounder.ground(new_image, segments)
new_image.ground.write()
