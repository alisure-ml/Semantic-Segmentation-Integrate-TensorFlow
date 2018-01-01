import os
import time
import tensorflow as tf
from Tools import Tools, ModelTools


class Test(object):

    def __init__(self, result_root, name, model_path, model_name, output_path):
        # 和保存模型相关的参数
        self.model_path = Tools.new_dir(os.path.join(result_root, name, model_path))
        self.checkpoint_path_and_name = os.path.join(self.model_path, model_name)

        # 结果目录
        self.output_path = Tools.new_dir(os.path.join(result_root, name, output_path))

        # 网络节点字典：所有可用节点都存储再这里
        self.net_op_dict = dict()

        # 构造网络
        self.build_net()

        # 检查是否满足条件
        self._check_op(self.net_op_dict)

        # 会话
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        pass

    # 子类需要重写改函数
    def build_net(self):

        pass

    @staticmethod
    def _check_op(net_op_dict):
        assert net_op_dict["image_batch"] is not None
        assert net_op_dict["logits_batch"] is not None
        assert net_op_dict["prediction_batch"] is not None
        assert net_op_dict["size_batch"] is not None
        assert net_op_dict["filename_batch"] is not None
        pass

    def _build_prediction(self, logits_batch, image_batch, size_batch, filename_batch):
        prediction_batch_op = tf.argmax(logits_batch, axis=3)
        self.net_op_dict["image_batch"] = image_batch
        self.net_op_dict["logits_batch"] = logits_batch
        self.net_op_dict["prediction_batch"] = prediction_batch_op
        self.net_op_dict["size_batch"] = size_batch
        self.net_op_dict["filename_batch"] = filename_batch
        pass

    # 通用节点集合
    def build_common(self, logits_output, image_batch, size_batch, filename_batch):
        self._build_prediction(logits_output, image_batch, size_batch, filename_batch)
        pass

    # 通用验证
    def test(self, fn_save_result):
        # 加载模型
        ModelTools.restore(self.sess, self.saver, self.model_path)

        coord = tf.train.Coordinator()
        # 线程队列
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        # 迭代训练
        count = 0
        while True:
            try:
                count += 1
                start_time = time.time()
                image_batch, prediction_batch, size_batch, filename_batch = (
                    self.sess.run([self.net_op_dict["image_batch"], self.net_op_dict["prediction_batch"],
                                   self.net_op_dict["size_batch"], self.net_op_dict["filename_batch"]]))
                duration = time.time() - start_time
                Tools.print('image {:d} ({:.3f} sec/step)'.format(count, duration))
                # 保存结果
                fn_save_result(prediction_batch, size_batch, filename_batch, result_path=self.output_path)
                pass
            except:
                Tools.print("Over")
                break
        coord.request_stop()
        coord.join(threads)
        pass

    pass
