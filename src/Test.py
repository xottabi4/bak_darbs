import os

import cv2
import numpy as np
import tensorflow as tf

import Config as cfg
from network_architectures.YoloNet import YoloNet
from utils.Timer import Timer


class Detector(object):
    def __init__(self, net, weight_file, classes):
        self.net = net
        self.weights_file = weight_file

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print 'Restoring weights from: ' + self.weights_file
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.fc_32, feed_dict={self.net.x: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(output[self.boundary1:self.boundary2],
                            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
                                                                       0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
                i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()

            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print 'Average detecting time: {:.3f}s'.format(detect_timer.average_time)

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

    def image_detector(self, imname, wait=0, resize=False):
        if not os.path.isfile(imname):
            raise ValueError("No such file is present: {}".format(imname))
        detect_timer = Timer()
        image = cv2.imread(imname)
        if resize:
            image = cv2.resize(image, (640, 360))

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print result
        print 'Average detecting time: {:.3f}s'.format(detect_timer.average_time)

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    network = YoloNet('test', cfg.MY_OWN_DATA_CLASSES)

    weight_file = '../data/weights/save.ckpt-100'
    # weight_file = '../data/output/2017_03_20_02_25/save.ckpt-30'

    detector = Detector(network, weight_file, cfg.MY_OWN_DATA_CLASSES)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    imname = '../test/1.png'
    detector.image_detector(imname, resize=True)

    # < xmin > 156 < / xmin >
    # < ymin > 97 < / ymin >
    # < xmax > 351 < / xmax >
    # < ymax > 270 < / ymax >
    # image = cv2.imread(imname)
    # # < width > 500 < / width >
    # # < height > 333 < / height >
    # # image = cv2.resize(image, (500, 333))
    # x = 156
    # y = 97
    # w = 351-156
    # h = 270-97
    # cv2.rectangle(image, (x , y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('ground truth image', image)
    #
    # # < x > 1347 < / x > < y > 788 < / y > < w > 284 < / w > < h > 208 < / h >
    # ground_truth_roi_true = [1347, 788, 284, 208]
    # x = int(ground_truth_roi_true[0])
    # y = int(ground_truth_roi_true[1])
    # w = int(ground_truth_roi_true[2])
    # h = int(ground_truth_roi_true[3])
    # image_true = cv2.imread(imname)
    #
    # image_true = cv2.circle(image_true, (x, y), 10, (0, 0, 255), -1)
    # image_true = cv2.circle(image_true, (x + w, y + h), 10, (0, 0, 255), -1)
    #
    # image_true = cv2.rectangle(image_true, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # image_true = cv2.resize(image_true, (448, 448))
    #
    # cv2.imshow('!!!before box transformation ground truth image', image_true)
    #
    # # asd
    #
    # image_size = 448
    # ground_truth_image_resized = cv2.imread(imname)
    #
    # h_ratio = 1.0 * image_size / ground_truth_image_resized.shape[0]
    # w_ratio = 1.0 * image_size / ground_truth_image_resized.shape[1]
    # print h_ratio
    # print w_ratio
    #
    # xmin = x
    # ymin = y + h
    # xmax = x + w
    # ymax = y
    #
    # x1 = max(min((xmin - 1) * w_ratio, image_size - 1), 0)
    # y1 = max(min((ymin - 1) * h_ratio, image_size - 1), 0)
    # x2 = max(min((xmax - 1) * w_ratio, image_size - 1), 0)
    # y2 = max(min((ymax - 1) * h_ratio, image_size - 1), 0)
    #
    # boxes = [(x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1]
    # print boxes
    # ground_truth_image_resized = cv2.resize(ground_truth_image_resized, (image_size, image_size))
    # x = int(boxes[0])
    # y = int(boxes[1])
    # w = int(boxes[2] / 2)
    # h = int(boxes[3] / 2)
    # cv2.rectangle(ground_truth_image_resized, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('ground truth image after resizing', ground_truth_image_resized)
    #
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()