import os
from Data import DataVOC2012Train
from Data import DataVOC2012Test
from Data import DataVOC2012Val
from Data import ImageToTFRecordTest
from TrainDeepLabV3 import TrainDeepLabV3
from TestDeepLabV3 import TestDeepLabV3
from ValDeepLabV3 import ValDeepLabV3
from TrainPSPNet import TrainPSPNet
from TestPSPNet import TestPSPNet
from ValPSPNet import ValPSPNet


class Main(object):

    def __init__(self, name, class_list, input_size, batch_size_list, number_classes,
                 total_step, save_freq, run_type,
                 result_root, model_path="model", model_name="model.ckpt", test_data_path="input/test"):
        if run_type == 1:
            self.data = DataVOC2012Train(data_file="data/VOC2012/train.tfrecord", number_classes=number_classes,
                                         input_size=input_size, batch_size=batch_size_list[0],
                                         random_scale=True, random_flip=True, random_adjust=True)
            self.train = class_list[0](self.data, total_step=total_step, result_root=result_root, name=name,
                                       summary_path="summary", model_path=model_path, model_name=model_name)
            self.train.train(save_freq)
        elif run_type == 2:
            self.data = DataVOC2012Val(data_file="data/VOC2012/val.tfrecord", number_classes=number_classes,
                                       input_size=input_size, batch_size=batch_size_list[1])
            self.val = class_list[1](self.data, result_root=result_root, name=name,
                                     model_path=model_path, model_name=model_name)
            self.val.val()
        elif run_type == 3:
            record_dir = test_data_path + ".tfrecord"
            data_list = [os.path.join(test_data_path, data_file) for data_file in os.listdir(test_data_path)]
            ImageToTFRecordTest(data_list, record_dir).run()

            self.data = DataVOC2012Test(data_file=record_dir, number_classes=number_classes,
                                        input_size=input_size, batch_size=batch_size_list[2])
            self.test = class_list[2](self.data, result_root=result_root, name=name, model_path=model_path,
                                      model_name=model_name, output_path="output")
            self.test.test(self.data.save_result)
        pass
    pass


if __name__ == '__main__':
    main = Main(name="PSPNet", class_list=list([TrainPSPNet, ValPSPNet, TestPSPNet]), input_size=list([720, 720]),
                batch_size_list=list([1, 2, 1]), number_classes=21, total_step=10000, save_freq=100, run_type=1,
                result_root="dist", model_path="model", model_name="model.ckpt", test_data_path="input/test")
    pass
