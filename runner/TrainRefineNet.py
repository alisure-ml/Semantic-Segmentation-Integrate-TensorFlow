import tensorflow as tf

import RefineNet
from Train import Train


class TrainRefineNet(Train):

    def __init__(self, data, total_step, result_root="dist", name="test",
                 summary_path="summary", model_path="model", model_name="model.ckpt", pre_train=None,
                 learn_rate_base=0.001, power=0.9, momentum=0.9):
        # 读取数据
        self.data = data

        # 和模型训练相关的参数
        self.moving_average_decay = 0.997

        super().__init__(total_step, result_root, name, summary_path, model_path, model_name, pre_train,
                         learn_rate_base, power, momentum)
        pass

    @staticmethod
    def _get_train_op(loss_op, learn_rate_op, moving_average_decay, step_op):
        # 1.优化损失：loos updates
        gradient_updates_op = tf.train.AdamOptimizer(learn_rate_op).minimize(loss_op)

        # 2.滑动平均：moving average updates
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, step_op)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # 3.BN参数更新：batch norm updates
        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        # 4.合并训练节点
        with tf.control_dependencies([variables_averages_op, gradient_updates_op, batch_norm_updates_op]):
            train_op = tf.no_op(name='train_op')
        return train_op

    def build_net(self):
        # 网络
        logits_output = RefineNet.RefineNet_model(self.data.image_batch, is_training=True)
        logits_shape = logits_output.get_shape().as_list()
        logits_output.set_shape([logits_shape[0], self.data.input_size[0], self.data.input_size[1], logits_shape[-1]])
        self.build_common(logits_output,  self.data.image_batch, self.data.annotation_batch,
                          self.data.number_classes, other_loss_op=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        train_op = self._get_train_op(self.net_op_dict["loss_op"], self.net_op_dict["learn_rate_op"],
                                      self.moving_average_decay, self.net_op_dict["step_op"])
        self.build_train(train_op)
        pass

    pass
