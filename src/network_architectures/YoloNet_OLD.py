# coding=utf-8
import tensorflow as tf

from ActivationFunction import ActivationFunction
from NetworkBase_OLD import NetworkBase


class YoloNet(NetworkBase):
    """
    YoloNet (version 1) network architecture is inspired by the GoogLeNet model  for  image  classification.
    Instead of the inception modules used by GoogLeNet, we simply use 1×1 reduction layers
    followed by 3×3 convo-lutional layers
    """

    def __init__(self, phase, classes):
        super(YoloNet, self).__init__(phase, classes)

    def build_networks(self):
        if self.disp_console:
            print "Building YOLO_small graph..."
        self.x = tf.placeholder('float32', [None, 448, 448, 3])
        self.conv_1 = self.conv_layer(1, self.x, 64, 7, 2)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
        self.conv_5 = self.conv_layer(5, self.pool_4, 128, 1, 1)
        self.conv_6 = self.conv_layer(6, self.conv_5, 256, 3, 1)
        self.conv_7 = self.conv_layer(7, self.conv_6, 256, 1, 1)
        self.conv_8 = self.conv_layer(8, self.conv_7, 512, 3, 1)
        self.pool_9 = self.pooling_layer(9, self.conv_8, 2, 2)
        self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
        self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
        self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
        self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
        self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
        self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
        self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
        self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
        self.pool_20 = self.pooling_layer(20, self.conv_19, 2, 2)
        self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
        self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
        self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
        self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)
        self.conv_25 = self.conv_layer(25, self.conv_24, 1024, 3, 1)
        self.conv_26 = self.conv_layer(26, self.conv_25, 1024, 3, 2)
        self.conv_27 = self.conv_layer(27, self.conv_26, 1024, 3, 1)
        self.conv_28 = self.conv_layer(28, self.conv_27, 1024, 3, 1)
        self.fc_29 = self.fc_layer(
            29, self.conv_28, 512, flat=True, linear=False)
        self.fc_30 = self.fc_layer(
            30, self.fc_29, 4096, flat=False, linear=False)
        if self.phase == 'train':
            self.dropout_31 = tf.nn.dropout(self.fc_30, keep_prob=0.5)
            self.fc_32 = self.fc_layer(
                32, self.dropout_31, self.output_size, flat=False, linear=True)
            self.labels = tf.placeholder(
                'float32', [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss = self.loss_layer(33, self.fc_32, self.labels)
            tf.summary.scalar(self.phase + '/total_loss', self.loss)
        else:
            self.fc_32 = self.fc_layer(
                32, self.fc_30, self.output_size, flat=False, linear=True)