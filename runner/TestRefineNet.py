import RefineNet
from Test import Test


class TestRefineNet(Test):

    def __init__(self, data, result_root="dist", name="test", model_path="model",
                 model_name="model.ckpt", output_path="output"):
        # 读取数据
        self.data = data

        super().__init__(result_root, name, model_path, model_name, output_path)
        pass

    def build_net(self):
        # 网络
        logits_output = RefineNet.RefineNet_model(self.data.image_batch, is_training=False)
        self.build_common(logits_output, self.data.image_batch, self.data.size_batch, self.data.filename_batch)
        pass

    pass
