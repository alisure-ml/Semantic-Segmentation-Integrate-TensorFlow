"""
    1. self.net_op_dict
        该节点是类的变量，存储运行需要的节点。
    2. self.build_net()
        该方法针对不同的网络需要重写。
    3. logits_batch, annotation_batch, number_classes
        这三个是训练的输入
    4. 包括度量、损失、学习率、日志、更新节点。
"""
import os
import time
import tensorflow as tf
import tensorflow.contrib.metrics as tcm
from Tools import Tools, ModelTools


class Train(object):

    def __init__(self, total_step, result_root, name, summary_path, model_path, model_name,
                 learn_rate_base, power, momentum):
        # 训练总步数
        self.total_step = total_step

        # 和保存模型相关的参数
        self.summary_path = Tools.new_dir(os.path.join(result_root, name, summary_path))
        self.model_path = Tools.new_dir(os.path.join(result_root, name, model_path))
        self.checkpoint_path_and_name = os.path.join(self.model_path, model_name)

        # 和训练相关的参数
        self.learn_rate_base = learn_rate_base
        self.power = power
        self.momentum = momentum

        # 网络节点字典：所有可用节点都存储再这里
        self.net_op_dict = dict()
        self.summary_op_dict = dict({"scalar":  dict()})

        # 构造网络
        self.build_net()

        # 检查是否满足条件
        self._check_op(self.net_op_dict)

        # 会话
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        self.summary_writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        pass

    # 子类需要重写改函数
    def build_net(self):

        pass

    @staticmethod
    def _check_op(net_op_dict):

        def has_key(key):
            for full_key in net_op_dict.keys():
                if key in full_key:
                    return True
            return False

        assert net_op_dict["step_op"] is not None
        assert net_op_dict["learn_rate_op"] is not None

        assert has_key("mean_iou_op")
        assert has_key("mean_iou_update_op")
        assert has_key("mean_iou_initializer_op")

        assert has_key("acc_op")
        assert has_key("acc_update_op")
        assert has_key("acc_initializer_op")

        assert net_op_dict["loss_op"] is not None
        assert net_op_dict["loss_cross_entropy_op"] is not None

        assert net_op_dict["summary_op"] is not None

        assert net_op_dict["train_op"] is not None

        assert net_op_dict["image_batch"] is not None
        assert net_op_dict["annotation_batch"] is not None
        assert net_op_dict["logits_batch"] is not None
        assert net_op_dict["prediction_batch"] is not None

        assert net_op_dict["logits_op"] is not None
        assert net_op_dict["annotation_op"] is not None
        assert net_op_dict["prediction_op"] is not None
        pass

    def _build_learn_ploy(self):
        learn_rate_base = tf.constant(self.learn_rate_base)
        step_op = tf.placeholder(dtype=tf.float32, shape=())
        learn_rate_op = tf.scalar_mul(learn_rate_base, tf.pow((1 - step_op / self.total_step), self.power))

        self.net_op_dict["step_op"] = step_op
        self.net_op_dict["learn_rate_op"] = learn_rate_op
        self.summary_op_dict["scalar"]["step_"] = step_op
        self.summary_op_dict["scalar"]["learn_rate"] = learn_rate_op
        pass

    def _build_prediction(self, logits_batch, image_batch, annotation_batch, number_classes):
        prediction_batch_op = tf.argmax(logits_batch, axis=3)
        prediction_batch = tf.reshape(logits_batch, [-1, number_classes])
        annotation_batch = tf.image.resize_nearest_neighbor(annotation_batch, tf.stack(logits_batch.get_shape()[1:3]))
        annotation_batch = tf.reshape(tf.squeeze(annotation_batch, squeeze_dims=[3]), [-1, ])

        # 忽略大于等于类别数的标签
        indices = tf.squeeze(tf.where(tf.less(annotation_batch, number_classes)), 1)

        annotation_op = tf.cast(tf.gather(annotation_batch, indices), tf.int32)  # [-1]
        logits_op = tf.gather(prediction_batch, indices)
        prediction_op = tf.cast(tf.argmax(logits_op, axis=1), tf.int32)  # [-1]

        self.net_op_dict["image_batch"] = image_batch
        self.net_op_dict["annotation_batch"] = annotation_batch
        self.net_op_dict["logits_batch"] = logits_batch
        self.net_op_dict["prediction_batch"] = prediction_batch_op
        self.net_op_dict["logits_op"] = logits_op
        self.net_op_dict["annotation_op"] = annotation_op
        self.net_op_dict["prediction_op"] = prediction_op
        pass

    @staticmethod
    def build_iou(prediction, annotation, number_classes):
        mean_iou_op, mean_iou_update_op = tcm.streaming_mean_iou(prediction, annotation, number_classes)
        mean_iou_initializer_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES))
        return mean_iou_op, mean_iou_update_op, mean_iou_initializer_op

    @staticmethod
    def _build_pca(prediction, annotation, number_classes):
        mean_pca_op, mean_pca_update_op = tf.metrics.mean_per_class_accuracy(annotation, prediction, number_classes)
        mean_pca_initializer_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES))
        return mean_pca_op, mean_pca_update_op, mean_pca_initializer_op

    @staticmethod
    def _build_acc(prediction, annotation):
        # accuracy = tf.equal(annotation, prediction)
        # accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        acc_op, acc_update_op = tcm.streaming_accuracy(prediction, annotation)
        acc_initializer_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES))
        return acc_op, acc_update_op, acc_initializer_op

    # 度量
    def _build_metrics(self, number_classes):
        prediction_op, annotation_op = self.net_op_dict["prediction_op"], self.net_op_dict["annotation_op"]

        acc_op, acc_update_op, acc_initializer_op = self._build_acc(prediction_op, annotation_op)
        mean_iou_op, mean_iou_update_op, mean_iou_initializer_op = self.build_iou(prediction_op, annotation_op,
                                                                                  number_classes)
        mean_pca_op, mean_pca_update_op, mean_pca_initializer_op = self._build_pca(prediction_op, annotation_op,
                                                                                   number_classes)

        self.net_op_dict["mean_iou_op"] = mean_iou_op
        self.net_op_dict["mean_iou_update_op"] = mean_iou_update_op
        self.net_op_dict["mean_iou_initializer_op"] = mean_iou_initializer_op
        self.net_op_dict["mean_pca_op"] = mean_pca_op
        self.net_op_dict["mean_pca_update_op"] = mean_pca_update_op
        self.net_op_dict["mean_pca_initializer_op"] = mean_pca_initializer_op
        self.net_op_dict["acc_op"] = acc_op
        self.net_op_dict["acc_update_op"] = acc_update_op
        self.net_op_dict["acc_initializer_op"] = acc_initializer_op

        self.summary_op_dict["scalar"]["mean_iou"] = mean_iou_op
        self.summary_op_dict["scalar"]["mean_pca"] = mean_pca_op
        self.summary_op_dict["scalar"]["acc"] = acc_op
        pass

    def _build_loss(self, other_loss_op=None):
        logits_op = self.net_op_dict["logits_op"]
        annotation_op = self.net_op_dict["annotation_op"]

        loss_cross_entropy_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_op,
                                                                                              labels=annotation_op))
        loss_op = loss_cross_entropy_op + (tf.add_n(other_loss_op) if other_loss_op is not None else 0)

        self.net_op_dict["loss_cross_entropy_op"] = loss_cross_entropy_op
        self.net_op_dict["loss_op"] = loss_op

        self.summary_op_dict["scalar"]["loss_cross_entropy"] = loss_cross_entropy_op
        self.summary_op_dict["scalar"]["loss"] = loss_op
        pass

    def _build_summary(self):
        summary_list = []
        scalar_dict = self.summary_op_dict["scalar"]
        if scalar_dict is not None:
            for key in scalar_dict:
                summary_list.append(tf.summary.scalar(key, scalar_dict[key]))
        summary_op = tf.summary.merge(summary_list)
        self.net_op_dict["summary_op"] = summary_op
        pass

    # 通用节点集合
    def build_common(self, logits_output, image_batch, annotation_batch, number_classes, other_loss_op=None):
        self._build_prediction(logits_output, image_batch, annotation_batch, number_classes)
        self._build_learn_ploy()
        self._build_metrics(number_classes)
        self._build_loss(other_loss_op)
        self._build_summary()
        pass

    # 训练节点
    def build_train(self, train_op):
        self.net_op_dict["train_op"] = train_op
        pass

    # 通用训练
    def train(self, save_freq=100):
        # 加载模型
        ModelTools.restore(self.sess, self.saver, self.model_path)

        tf.set_random_seed(1234)
        coord = tf.train.Coordinator()
        # 线程队列
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        # 迭代训练
        for step in range(self.total_step):
            start_time = time.time()
            loss, _, _, _, _, summary = self.sess.run([self.net_op_dict["loss_op"],
                                                       self.net_op_dict["mean_iou_update_op"],
                                                       self.net_op_dict["mean_pca_update_op"],
                                                       self.net_op_dict["acc_update_op"],
                                                       self.net_op_dict["train_op"],
                                                       self.net_op_dict["summary_op"]],
                                                      feed_dict={self.net_op_dict["step_op"]: step})
            duration = time.time() - start_time
            self.summary_writer.add_summary(summary, step)
            mean_iou, mean_pca, acc = self.sess.run([self.net_op_dict["mean_iou_op"],
                                                     self.net_op_dict["mean_pca_op"],
                                                     self.net_op_dict["acc_op"]])
            Tools.print('step {:d} loss={:.3f}, mean iou={}, mean pca={}, acc={} ({:.3f} sec/step)'
                        .format(step, loss, mean_iou, mean_pca, acc, duration))

            if step % save_freq == 0:
                self.saver.save(self.sess, self.checkpoint_path_and_name, global_step=step)
                self.sess.run([self.net_op_dict["mean_iou_initializer_op"],
                               self.net_op_dict["mean_pca_initializer_op"],
                               self.net_op_dict["acc_initializer_op"]])
                Tools.print('The checkpoint has been created.')
            pass

        coord.request_stop()
        coord.join(threads)
        pass

    pass
