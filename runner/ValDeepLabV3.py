from DeepLabV3 import deeplabv3
from Val import Val


class ValDeepLabV3(Val):

    def __init__(self, data, result_root="dist", name="test", model_path="model", model_name="model.ckpt"):
        # 读取数据
        self.data = data
        # 和模型训练相关的参数
        self.number_layers = 101
        self.weight_decay = 0.0001
        self.bn_weight_decay = 0.9997
        self.not_restore_last = True
        self.freeze_bn = True

        super().__init__(result_root, name, model_path, model_name)
        pass

    def build_net(self):
        # 构造网络，得到logits
        net, end_points = deeplabv3(self.data.image_batch, num_classes=self.data.number_classes,
                                    depth=self.number_layers, is_training=False, weight_decay=self.weight_decay,
                                    bn_weight_decay=self.bn_weight_decay)
        logits_output = end_points['resnet{}/logits'.format(self.number_layers)]
        self.build_common(logits_output, self.data.image_batch, self.data.annotation_batch, self.data.number_classes)
        pass

    pass
