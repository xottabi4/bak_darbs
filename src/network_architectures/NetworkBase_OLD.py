import abc

import Config as cfg
import numpy as np
import tensorflow as tf


class NetworkBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, phase, classes):
        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        self.disp_console = cfg.DISP_CONSOLE
        self.phase = phase
        self.conv_layer_variables = []
        self.all_layer_variables = []
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.build_networks()
        # self.global_step = tf.Variable(0, trainable=False)

    @abc.abstractmethod
    def build_networks(self):
        return

    def conv_layer(self, idx, inputs, filters, size, stride):
        channels = inputs.get_shape()[3]
        stddev = np.math.sqrt(2.0 / int(channels))
        weight = tf.Variable(tf.truncated_normal(
            [size, size, int(channels), filters], stddev=stddev))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))
        self.conv_layer_variables.append(weight)
        self.conv_layer_variables.append(biases)
        self.all_layer_variables.append(weight)
        self.all_layer_variables.append(biases)

        pad_size = size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size],
                            [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1],
                            padding='VALID', name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')

        if self.disp_console:
            print '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
                idx, size, size, stride, filters, int(channels))
        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

    def pooling_layer(self, idx, inputs, size, stride):
        if self.disp_console:
            print '    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (
                idx, size, size, stride)
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                              name=str(idx) + '_pool')

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        stddev = np.math.sqrt(2.0 / dim)
        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=stddev))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
        self.all_layer_variables.append(weight)
        self.all_layer_variables.append(biases)
        if self.disp_console:
            print '    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (
                idx, hiddens, int(dim), int(flat), 1 - int(linear))
        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')
        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')

    def calc_iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                  (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                  (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, idx, predicts, labels):

        predict_classes = tf.reshape(predicts[:, :self.boundary1],
                                     [self.batch_size, self.cell_size, self.cell_size, self.num_class])
        predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],
                                    [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
        predict_boxes = tf.reshape(predicts[:, self.boundary2:],
                                   [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        response = tf.reshape(labels[:, :, :, 0],
                              [self.batch_size, self.cell_size, self.cell_size, 1])
        boxes = tf.reshape(labels[:, :, :, 1:5],
                           [self.batch_size, self.cell_size, self.cell_size, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
        classes = labels[:, :, :, 5:]

        offset = tf.constant(self.offset, dtype=tf.float32)
        offset = tf.reshape(offset,
                            [1, self.cell_size, self.cell_size, self.boxes_per_cell])
        offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
        predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                       (predict_boxes[:, :, :, :, 1] + tf.transpose(offset,
                                                                                    (0, 2, 1, 3))) / self.cell_size,
                                       tf.square(predict_boxes[:, :, :, :, 2]),
                                       tf.square(predict_boxes[:, :, :, :, 3])])
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

        iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
        # mask = tf.tile(response, [1, 1, 1, self.boxes_per_cell])

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                               boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

        # class_loss
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(response * (predict_classes - classes)),
                                                  reduction_indices=[1, 2, 3]), name='class_loss') * self.class_scale

        # object_loss
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_mask * (predict_scales - iou_predict_truth)),
                                                   reduction_indices=[1, 2, 3]), name='object_loss') * self.object_scale

        # noobject_loss
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_mask * predict_scales),
                                                     reduction_indices=[1, 2, 3]),
                                       name='noobject_loss') * self.noobject_scale

        # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta),
                                                  reduction_indices=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale

        tf.summary.scalar(self.phase + '/class_loss', class_loss)
        tf.summary.scalar(self.phase + '/object_loss', object_loss)
        tf.summary.scalar(self.phase + '/noobject_loss', noobject_loss)
        tf.summary.scalar(self.phase + '/coord_loss', coord_loss)

        tf.summary.histogram(self.phase + '/boxes_delta_x', boxes_delta[:, :, :, :, 0])
        tf.summary.histogram(self.phase + '/boxes_delta_y', boxes_delta[:, :, :, :, 1])
        tf.summary.histogram(self.phase + '/boxes_delta_w', boxes_delta[:, :, :, :, 2])
        tf.summary.histogram(self.phase + '/boxes_delta_h', boxes_delta[:, :, :, :, 3])
        tf.summary.histogram(self.phase + '/iou', iou_predict_truth)

        return class_loss + object_loss + noobject_loss + coord_loss
