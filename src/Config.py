import os

#
# path and dataset parameter
#

DATA_PATH = '../data'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

MY_OWN_DATA_PATH = os.path.join(DATA_PATH, 'my_own_data_set')

CACHE_PATH = os.path.join(DATA_PATH, 'cache')

OUTPUT_DIR = os.path.join(DATA_PATH, 'training_output')

WEIGHTS_DIR = os.path.join(DATA_PATH, 'weights')

PASCAL_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                  'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor']

MY_OWN_DATA_CLASSES = ['car']

FLIPPED = False

#
# model parameter
#

IMAGE_SIZE = 448

# maybe better it would be 11
CELL_SIZE = 7

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = True

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001

DECAY_STEPS = 10000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 32

MAX_ITER = 3000

SUMMARY_ITER = 1111111111111111111111111111111

SAVE_ITER = 10


#
# test parameter
#

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
