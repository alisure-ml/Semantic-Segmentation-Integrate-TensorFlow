import os
import time
import tensorflow as tf


class Tools(object):

    @staticmethod
    def new_dir(path_or_file_name):
        path_name, file_name = os.path.split(path_or_file_name)
        if "." not in file_name:
            path_name = path_or_file_name
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        return path_or_file_name

    @staticmethod
    def print(info):
        print("{} {}".format(time.strftime("%H:%M:%S", time.localtime()), info))
        pass

    pass


class ModelTools(object):

    # 如果模型存在，恢复模型
    @staticmethod
    def restore(sess, saver, log_dir):
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            Tools.print("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
        else:
            Tools.print('No checkpoint file found.')
            pass
        pass

    pass

if __name__ == '__main__':
    Tools.new_dir("aa/a/ccc.x")
