from PSPNet import PSPNet
from Val import Val


class ValPSPNet(Val):

    def __init__(self, data, result_root="dist", name="test", model_path="model", model_name="model.ckpt"):
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

        super().__init__(result_root, name, model_path, model_name)
        pass

    def build_net(self):
        # 网络
        net = PSPNet({'data': self.data.image_batch}, is_training=True, num_classes=self.data.number_classes,
                     last_pool_size=self.last_pool_size, filter_number=self.filter_number)
        logits_output = net.layers['conv6']
        self.build_common(logits_output,  self.data.image_batch, self.data.annotation_batch, self.data.number_classes)
        pass

    pass
