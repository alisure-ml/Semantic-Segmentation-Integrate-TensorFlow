import tensorflow as tf

from DeepLabV3 import deeplabv3
from Train import Train


class TrainDeepLabV3(Train):

    def __init__(self, data, total_step, result_root="dist", name="test",
                 summary_path="summary", model_path="model", model_name="model.ckpt",
                 learn_rate_base=0.001, power=0.9, momentum=0.9):
        # 读取数据
        self.data = data
        # 和模型训练相关的参数
        self.number_layers = 101
        self.weight_decay = 0.0001
        self.bn_weight_decay = 0.9997
        self.not_restore_last = True
        self.freeze_bn = True

        super().__init__(total_step, result_root, name, summary_path, model_path, model_name,
                         learn_rate_base, power, momentum)
        pass

    # 获取需要更新的变量
    @staticmethod
    def _get_trainable_variables(not_restore_last, freeze_bn):
        restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not not_restore_last]
        if freeze_bn:
            all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
        else:
            all_trainable = [v for v in tf.trainable_variables()]
        conv_trainable = [v for v in all_trainable if 'fc' not in v.name]
        return restore_var, conv_trainable

    # 优化：计算梯度、选择优化算法、更新梯度
    @staticmethod
    def _get_train_op(loss_op, conv_trainable, learn_rate_op, momentum):
        # 计算梯度
        grads_conv = tf.gradients(loss_op, conv_trainable)
        # 选择优化算法
        optimizer_op_conv = tf.train.MomentumOptimizer(learn_rate_op, momentum)
        # 更新梯度
        train_op = optimizer_op_conv.apply_gradients(zip(grads_conv, conv_trainable))
        return train_op

    def build_net(self):
        # 构造网络，得到logits
        net, end_points = deeplabv3(self.data.image_batch, num_classes=self.data.number_classes,
                                    depth=self.number_layers, is_training=True, weight_decay=self.weight_decay,
                                    bn_weight_decay=self.bn_weight_decay)
        logits_output = end_points['resnet{}/logits'.format(self.number_layers)]

        # 组装损失并构造通用的节点
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.build_common(logits_output, self.data.image_batch, self.data.annotation_batch,
                          self.data.number_classes, other_loss_op=reg_losses)

        # 获取需要更新的变量并进行优化
        restore_var, conv_trainable = self._get_trainable_variables(self.not_restore_last, self.freeze_bn)
        train_op = self._get_train_op(self.net_op_dict["loss_op"], conv_trainable,
                                      self.net_op_dict["learn_rate_op"], self.momentum)
        self.build_train(train_op)
        pass

    pass
