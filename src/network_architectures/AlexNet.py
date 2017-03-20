import tensorflow as tf

from ActivationFunction import ActivationFunction
from NetworkBase import NetworkBase


class AlexNet(NetworkBase):
    def build_networks(self):
        if self.disp_console:
            print "Building AlexNet graph..."
        self.x = tf.placeholder('float32', [None, 224, 224, 3])
        # layer 1
        self.conv_1 = self.conv_layer(1, self.x, 96, 11, 4, activation=ActivationFunction.ReLU)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 3, 2)
        # layer 2
        self.conv_3 = self.conv_layer(3, self.pool_2, 192, 5, 1, activation=ActivationFunction.ReLU)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 3, 2)
        # layer 3
        self.conv_5 = self.conv_layer(5, self.pool_4, 384, 3, 1, activation=ActivationFunction.ReLU)
        # layer 4
        self.conv_6 = self.conv_layer(6, self.conv_5, 384, 3, 1, activation=ActivationFunction.ReLU)
        # layer 5
        self.conv_7 = self.conv_layer(7, self.conv_6, 256, 3, 1, activation=ActivationFunction.ReLU)
        self.pool_8 = self.pooling_layer(8, self.conv_7, 3, 2)
        # layer 6
        self.fc_9 = self.fc_layer(9, self.pool_8, 4096, flat=True, activation=ActivationFunction.ReLU)
        self.dropout_10 = tf.nn.dropout(self.fc_9, keep_prob=0.5)
        # layer 7
        self.fc_11 = self.fc_layer(11, self.dropout_10, 4096, flat=False, activation=ActivationFunction.ReLU)
        self.dropout_12 = tf.nn.dropout(self.fc_11, keep_prob=0.5)
        # layer 8 - output
        if self.phase == 'train':
            self.fc_12 = self.fc_layer(12, self.dropout_12, self.output_size, flat=False, activation=None)
            self.labels = tf.placeholder('float32', [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss = self.loss_layer(13, self.fc_12, self.labels)
            tf.summary.scalar(self.phase + '/total_loss', self.loss)
        else:
            self.fc_12 = self.fc_layer(12, self.dropout_12, self.output_size, flat=False, activation=None)
