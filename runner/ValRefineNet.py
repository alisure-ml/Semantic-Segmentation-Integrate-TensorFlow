import RefineNet
from Val import Val


class ValRefineNet(Val):

    def __init__(self, data, result_root="dist", name="test", model_path="model", model_name="model.ckpt"):
        # 读取数据
        self.data = data

        # 和模型训练相关的参数
        self.moving_average_decay = 0.997

        super().__init__(result_root, name, model_path, model_name)
        pass

    def build_net(self):
        # 网络
        logits_output = RefineNet.RefineNet_model(self.data.image_batch, is_training=False)
        logits_shape = logits_output.get_shape().as_list()
        logits_output.set_shape([logits_shape[0], self.data.input_size[0], self.data.input_size[1], logits_shape[-1]])
        self.build_common(logits_output,  self.data.image_batch, self.data.annotation_batch, self.data.number_classes)
        pass

    pass
