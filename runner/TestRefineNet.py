import RefineNet
from Test import Test


class TestRefineNet(Test):

    def __init__(self, data, result_root="dist", name="test", model_path="model",
                 model_name="model.ckpt", output_path="output"):
        # 读取数据
        self.data = data
        # 和模型训练相关的参数
        self.moving_average_decay = 0.997

        super().__init__(result_root, name, model_path, model_name, output_path)
        pass

    def build_net(self):
        # 网络
        logits_output = RefineNet.RefineNet_model(self.data.image_batch, is_training=True)
        self.build_common(logits_output, self.data.image_batch, self.data.size_batch, self.data.filename_batch)
        pass

    pass
