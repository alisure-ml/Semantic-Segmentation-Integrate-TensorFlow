import os
from Data import DataVOC2012Train
from Data import DataVOC2012Test
from Data import DataVOC2012Val
from Data import ImageToTFRecordTest
from TrainDeepLabV3 import TrainDeepLabV3  # [513, 513]
from TestDeepLabV3 import TestDeepLabV3
from ValDeepLabV3 import ValDeepLabV3
from TrainPSPNet import TrainPSPNet  # [720, 720]


class Main(object):

    # run_type=[1,train] [2, val] [3, test]
    def __init__(self, total_step, save_freq, run_type=1):
        if run_type == 1:
            self.data = DataVOC2012Train(data_file="data/VOC2012/train.tfrecord",
                                         number_classes=19, input_size=list([513, 513]),
                                         batch_size=1, random_scale=True, random_flip=True, ignore_label=255)
            self.train = TrainDeepLabV3(self.data, total_step=total_step, result_root="dist", name="DeepLabV3",
                                        summary_path="summary", model_path="model", model_name="model.ckpt")
            self.train.train(save_freq)
        elif run_type == 2:
            self.data = DataVOC2012Val(data_file="data/VOC2012/train.tfrecord",
                                       number_classes=19, input_size=list([513, 513]),
                                       batch_size=1, ignore_label=255)
            self.val = ValDeepLabV3(self.data, result_root="dist", name="DeepLabV3",
                                    model_path="model", model_name="model.ckpt")
            self.val.val()
        elif run_type == 3:
            data_path = "input/test"
            record_dir = os.path.join("input", "test.tfrecord")
            data_list = [os.path.join(data_path, data_file) for data_file in os.listdir(data_path)]
            ImageToTFRecordTest(data_list, record_dir).run()

            self.data = DataVOC2012Test(data_file=record_dir, number_classes=19, input_size=list([513, 513]), batch_size=1)
            self.test = TestDeepLabV3(self.data, result_root="dist", name="DeepLabV3", model_path="model",
                                      model_name="model.ckpt", output_path="output")
            self.test.test(self.data.save_result)
        pass
    pass


if __name__ == '__main__':
    main = Main(total_step=100, save_freq=5, run_type=1)
    pass
