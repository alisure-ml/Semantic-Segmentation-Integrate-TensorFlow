import tensorflow as tf

from PSPNet import PSPNet
from Train import Train


class TrainPSPNet(Train):

    def __init__(self, data, total_step, result_root="dist", name="test",
                 summary_path="summary", model_path="model", model_name="model.ckpt",
                 learn_rate_base=0.001, power=0.9, momentum=0.9):
        # 读取数据
        self.data = data
        self.last_pool_size = 90
        self.filter_number = 64
        if self.data.input_size[0] < self.last_pool_size * 8 or self.data.input_size[1] < self.last_pool_size * 8:
            raise Exception("必须保证input_size大于8倍的last_pool_size")

        # 和模型训练相关的参数
        self.train_beta_gamma = True
        self.weight_decay = 0.0001
        self.update_mean_var = True

        super().__init__(total_step, result_root, name, summary_path, model_path, model_name,
                         learn_rate_base, power, momentum)
        pass

    @staticmethod
    def _get_trainable_variables(train_beta_gamma):
        # According from the prototxt in Caffe implement, learning rate must multiply by 10.0 in pyramid module
        fc_list = ['conv5_3_pool1_conv', 'conv5_3_pool2_conv',
                   'conv5_3_pool3_conv', 'conv5_3_pool6_conv', 'conv6', 'conv5_4']
        # 所有可训练变量
        all_trainable = [v for v in tf.trainable_variables()
                         if ('beta' not in v.name and 'gamma' not in v.name) or train_beta_gamma]
        # fc_list中的全连接层可训练变量和卷积可训练变量
        fc_trainable = [v for v in all_trainable if v.name.split('/')[0] in fc_list]
        conv_trainable = [v for v in all_trainable if v.name.split('/')[0] not in fc_list]  # lr * 1.0
        fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
        fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
        # 验证
        assert (len(all_trainable) == len(fc_trainable) + len(conv_trainable))
        assert (len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
        return conv_trainable, fc_w_trainable, fc_b_trainable

    @staticmethod
    def _get_train_op(loss_op, conv_trainable, fc_w_trainable, fc_b_trainable, learn_rate_op, momentum):
        # 计算梯度
        grads = tf.gradients(loss_op, conv_trainable + fc_w_trainable + fc_b_trainable)
        grads_conv = grads[:len(conv_trainable)]
        grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
        grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]
        # 选择优化算法
        optimizer_op_conv = tf.train.MomentumOptimizer(learn_rate_op, momentum)
        optimizer_op_fc_w = tf.train.MomentumOptimizer(learn_rate_op * 10.0, momentum)
        optimizer_op_fc_b = tf.train.MomentumOptimizer(learn_rate_op * 20.0, momentum)
        # 更新梯度
        train_op_conv = optimizer_op_conv.apply_gradients(zip(grads_conv, conv_trainable))
        train_op_fc_w = optimizer_op_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
        train_op_fc_b = optimizer_op_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))
        # 一次完成多种操作
        train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
        return train_op

    def build_net(self):
        # 网络
        net = PSPNet({'data': self.data.image_batch}, is_training=True, num_classes=self.data.number_classes,
                     last_pool_size=self.last_pool_size, filter_number=self.filter_number)
        logits_output = net.layers['conv6']

        # l2 loss
        l2_losses = [self.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        self.build_common(logits_output,  self.data.image_batch, self.data.annotation_batch,
                          self.data.number_classes, other_loss_op=l2_losses)

        # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
        update_ops = None if not self.update_mean_var else tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # 对变量以不同的学习率优化：分别求梯度、应用梯度
        conv_trainable, fc_w_trainable, fc_b_trainable = self._get_trainable_variables(self.train_beta_gamma)
        with tf.control_dependencies(update_ops):
            train_op = self._get_train_op(self.net_op_dict["loss_op"], conv_trainable, fc_w_trainable, fc_b_trainable,
                                          self.net_op_dict["learn_rate_op"], self.momentum)
            pass
        self.build_train(train_op)
        pass

    pass
