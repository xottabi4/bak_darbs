import cPickle
import copy
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

import Config as cfg


class MyOwnDataFormat(object):
    def __init__(self, phase, rebuild=False):
        self.data_path = cfg.MY_OWN_DATA_PATH
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.MY_OWN_DATA_CLASSES
        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
        self.flipped = cfg.FLIPPED
        self.prediction_count = 5 + len(self.classes)
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.prepare()

    def get(self):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, self.prediction_count))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print 'Appending horizontally-flipped training examples ...'
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in xrange(self.cell_size):
                    for j in xrange(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][
                                i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cell_size_info = "cell_size" + str(self.cell_size)+"_"
        cache_file = os.path.join(self.cache_path, 'own_data_set_'+cell_size_info + self.phase + '_gt_labels.pkl')
        if os.path.exists(cache_file) and not self.rebuild:
            print 'Loading gt_labels from: ' + cache_file
            with open(cache_file, 'rb') as f:
                gt_labels = cPickle.load(f)
            return gt_labels

        if self.phase == 'train':
            txtname = os.path.join(self.data_path, 'train.txt')
        else:
            txtname = os.path.join(self.data_path, 'test.txt')
        with open(txtname, 'r') as f:
            image_names = [x.strip() for x in f.readlines()]

        gt_labels = []
        for image_name in image_names:
            label, num = self.load_annotation(image_name)
            imname = os.path.join(self.data_path, 'input_images', image_name + '.png')
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})
        print 'Saving gt_labels to: ' + cache_file
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_labels, f)
        return gt_labels

    def load_annotation(self, image_name):
        """
        Load image and bounding boxes info from XML file.
        """
        imname = os.path.join(self.data_path, 'input_images', image_name + '.png')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        label = np.zeros((self.cell_size, self.cell_size, self.prediction_count))
        filename = os.path.join(self.data_path, 'input_images_roi', image_name + '.xml')
        tree = ET.parse(filename)
        root = tree.getroot()

        # root = tree.find('boundingBoxes')
        bounding_boxes = root.findall('boundingBox')

        for bounding_box in bounding_boxes:

            # convert roi to pascal format
            xmin = float(bounding_box.find('x').text)
            ymin = float(bounding_box.find('y').text)
            xmax = float(bounding_box.find('x').text) + float(bounding_box.find('w').text)
            ymax = float(bounding_box.find('y').text) + float(bounding_box.find('h').text)

            # Make pixel indexes 0-based
            x1 = max(min((xmin - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((ymin - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((xmax - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((ymax - 1) * h_ratio, self.image_size - 1), 0)

            # cls_ind = self.class_to_ind[object.find('name').text.lower().strip()]
            # if len(self.classes) == 1:
            cls_ind = 0

            boxes = [(x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(bounding_boxes)
