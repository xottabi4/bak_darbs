import argparse
import datetime
import os

import tensorflow as tf

import Config as cfg
from network_architectures.AlexNet import AlexNet
from network_architectures.YoloNet import YoloNet
from training_data.MyOwnDataFormat import MyOwnDataFormat
from utils.Timer import Timer




class Solver(object):
    def __init__(self, net, data, weights_file=None, use_fc_layer_variables=True):
        self.net = net
        self.data = data
        self.weights_file = weights_file
        self.use_fc_layer_variables = use_fc_layer_variables
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(cfg.OUTPUT_DIR,
                                       datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.global_step = tf.get_variable('global_step', [],
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Custom gradient optimizer with decay
        # self.learning_rate = tf.train.exponential_decay(
        #     self.initial_learning_rate, self.global_step, self.decay_steps,
        #     self.decay_rate, self.staircase, name='learning_rate')
        # self.optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate=self.learning_rate).minimize(
        #     self.net.loss, global_step=self.global_step)

        # Adam optimizer
        self.learning_rate = self.initial_learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate) \
            .minimize(self.net.loss, global_step=self.global_step)

        # Without it training happens too slow
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(self.net.all_layer_variables, max_to_keep=None)
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print 'Restoring weights from: ' + self.weights_file
            if self.use_fc_layer_variables:
                saver = tf.train.Saver(self.net.all_layer_variables, max_to_keep=None)
            else:
                saver = tf.train.Saver(self.net.conv_layer_variables, max_to_keep=None)
            saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)
        self.log_file = open(os.path.join(self.output_dir, "training_log.txt"), "w")

    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in xrange(1, self.max_iter + 1):
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.x: images, self.net.labels: labels}

            train_timer.tic()
            # summary_str, loss = self.sess.run([self.summary_op, self.net.loss], feed_dict=feed_dict)
            _, summary_str, loss = self.sess.run([self.train_op, self.summary_op, self.net.loss], feed_dict=feed_dict)
            # _, loss = self.sess.run([self.train_op, self.net.loss], feed_dict=feed_dict)
            train_timer.toc()
            log_str = ('{} Epoch: {}, Step: {},'
                       ' Loss: {:5.3f} Speed: {:.3f}s/iter,'
                       ' Load: {:.3f}s/iter, Remain: {}').format(
                datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                self.data.epoch,
                int(step),
                # round(self.initial_learning_rate.eval(session=self.sess), 6),
                loss,
                train_timer.average_time,
                load_timer.average_time,
                train_timer.remain(step, self.max_iter))
            print log_str
            self.log_file.write(log_str + "\n")

            self.writer.add_summary(summary_str, step)

            if step % self.save_iter == 0:
                log_str = '{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    self.output_dir)
                print log_str
                self.log_file.write(log_str + "\n")
                path_where_variables_saved = self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step,
                                                             write_meta_graph=False)
                log_str = '{} Session variables successfully saved to: {}'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    path_where_variables_saved)
                print log_str
                self.log_file.write(log_str + "\n")

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=None, type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = str(args.gpu)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    network = YoloNet('train', cfg.MY_OWN_DATA_CLASSES)
    # network = AlexNet('train', cfg.MY_OWN_DATA_CLASSES)

    # data_set = PascalVoc('train')
    data_set = MyOwnDataFormat('train')

    # weight_file = "../data/training_output/2017_03_21_19_09/save.ckpt-30"
    weight_file = None
    # weight_file = '../data/training_output/2017_03_20_23_03/save.ckpt-120'
    # weight_file = '../data/training_output/2017_03_21_00_14/save.ckpt-130'

    solver = Solver(network, data_set, weights_file=weight_file, use_fc_layer_variables=True)

    solver.train()


if __name__ == '__main__':
    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
