import os
import numpy as np
from PIL import Image
import tensorflow as tf
from Tools import Tools


# 准备训练和验证的数据
class ImageToTFRecordVOC2012(object):

    def __init__(self, data_dir="C:\\ALISURE\\Data\\voc\\VOCdevkit\\VOC2012",
                 list_files=list(["../data/VOC2012/train.txt", "../data/VOC2012/val.txt"]),
                 record_dir="../data/VOC2012", image_format="JPEG", annotation_format="PNG"):
        self.data_dir = data_dir
        self.list_files = list_files

        self.record_dir = Tools.new_dir(record_dir)
        # 根据list和record_dir得到record结果的地址和文件名
        self.record_names = [os.path.splitext(os.path.split(list_file)[1])[0] for list_file in self.list_files]
        self.record_files = [self.record_dir + "/" + record_name + ".tfrecord" for record_name in self.record_names]

        self.image_format = image_format
        self.annotation_format = annotation_format
        pass

    @staticmethod
    def _int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _convert_to_example(self, filename, image_buffer, annotation_string, height, width):
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': self._int64_feature(height),
            'image/width': self._int64_feature(width),
            'image/filename': self._bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
            'image/format': self._bytes_feature(tf.compat.as_bytes(self.image_format)),
            'image/annotation/format': self._bytes_feature(tf.compat.as_bytes(self.annotation_format)),
            'image/encoded': self._bytes_feature(tf.compat.as_bytes(image_buffer)),
            'image/annotation/encoded': self._bytes_feature(tf.compat.as_bytes(annotation_string)),
        }))
        return example

    def _convert_to_tf_record(self):
        label_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_label = tf.image.encode_png(tf.expand_dims(label_placeholder, 2))
        sess = tf.Session()
        for list_file_index, list_file in enumerate(self.list_files):
            # 打开tf-record文件
            writer = tf.python_io.TFRecordWriter(self.record_files[list_file_index])
            # 打开list文件
            with open(list_file, 'r') as f:
                for image_file_index, image_file in enumerate(f):
                    # 得到要处理的图片名称
                    img_path, gt_path = image_file.strip().split()

                    # 读取image(jpeg)数据
                    with tf.gfile.FastGFile(self.data_dir + img_path, 'rb') as ff:
                        image_data = ff.read()

                    # 读取annotation(png)数据
                    im = Image.open(self.data_dir + gt_path)
                    annotation_string = sess.run(encoded_label, feed_dict={label_placeholder: np.array(im)})
                    # 得到样例
                    example = self._convert_to_example(img_path, image_data, annotation_string, im.size[1], im.size[0])

                    # 写入
                    writer.write(example.SerializeToString())
                pass
            print("OK {}".format(list_file))
            writer.close()
            pass
        sess.close()
        pass

    def run(self):
        if os.path.exists(self.record_files[0]):
            Tools.print("record file has existed...")
        else:
            Tools.print("begin to transform image to records")
            self._convert_to_tf_record()
            [Tools.print("result in {}".format(record_file)) for record_file in self.record_files]
            Tools.print("end to transform image to records")
        pass
    pass


# 训练Data
class DataVOC2012Train(object):

    def __init__(self, data_file, number_classes, input_size,
                 batch_size, random_scale=True, random_flip=True, ignore_label=255):

        # 类别数
        self.number_classes = number_classes
        # 输入大小
        self.input_size = input_size
        # 批次大小
        self.batch_size = batch_size

        # 数据文件
        self._data_file = data_file
        # 是否随机缩放
        self._random_scale = random_scale
        # 是否随机左右翻转
        self._random_flip = random_flip
        # 忽略得标签（padding时使用）
        self._ignore_label = ignore_label

        # 图像均值
        self._image_mean = np.array((103.939, 116.779, 123.68), dtype=np.float32)

        # color map
        self.label_colors = [(0, 0, 0),  # 0=background
                             (128, 0, 0), (0, 128, 0), (128, 128, 0),  # 1=aeroplane, 2=bicycle, 3=bird
                             (0, 0, 128), (128, 0, 128), (0, 128, 128),  # 4=boat, 5=bottle, 6=bus
                             (128, 128, 128), (64, 0, 0), (192, 0, 0),  # 7=car, 8=cat, 9=chair
                             (64, 128, 0), (192, 128, 0), (64, 0, 128),  # 10=cow, 11=diningtable, 12=dog
                             (192, 0, 128), (64, 128, 128), (192, 128, 128),  # 13=horse, 14=motorbike, 15=person
                             (0, 64, 0), (128, 64, 0), (0, 192, 0),   # 16=potted plant, 17=sheep, 18=sofa
                             (128, 192, 0), (0, 64, 128)]  # 19=train, 20=tv/monitor

        # 获取文件列表
        input_data_file = tf.train.match_filenames_once(data_file)
        # 输入队列
        input_data_queue = tf.train.string_input_producer(input_data_file, shuffle=True)
        # 从队列中读取数据
        self._image, self._annotation = self._read_from_tf_record(input_data_queue)
        # 处理数据：数据增强
        self._image, self._annotation = self._process_data(self._image, self._annotation, self.input_size)

        # 成批次读取数据
        self.image_batch, self.annotation_batch = self._get_batch_data(self._image, self._annotation, self.batch_size)
        pass

    # 从队列中读取数据
    @staticmethod
    def _read_from_tf_record(input_data_queue):
        # 从队列钟读取样例
        _, serialized_example = tf.TFRecordReader().read(input_data_queue)
        features = tf.parse_single_example(serialized_example, features={
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "image/annotation/encoded": tf.FixedLenFeature([], tf.string)
        })
        image = tf.image.decode_jpeg(features["image/encoded"], channels=3)
        annotation = tf.image.decode_png(features["image/annotation/encoded"], channels=1)
        return image, annotation

    # 处理数据：数据增强
    @staticmethod
    def _process_data(image, annotation, input_size):
        image = tf.image.resize_images(image, input_size, method=tf.image.ResizeMethod.BILINEAR)
        annotation = tf.image.resize_images(annotation, input_size, method=tf.image.ResizeMethod.BILINEAR)
        return image, annotation

    # 成批次读取数据
    @staticmethod
    def _get_batch_data(image, annotation, batch_size):
        capacity = 1000 + 3 * batch_size
        image_batch, annotation_batch = tf.train.shuffle_batch([image, annotation], batch_size, capacity, 100)
        return image_batch, annotation_batch

    # test
    def test(self):
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(5):
                image_batch_data, annotation_batch_data = sess.run([self.image_batch, self.annotation_batch])
                pass

            coord.request_stop()
            coord.join(threads)
        pass

    pass


# 验证Data
class DataVOC2012Val(object):

    def __init__(self, data_file, number_classes, input_size, batch_size, ignore_label=255):

        # 类别数
        self.number_classes = number_classes
        # 输入大小
        self.input_size = input_size
        # 批次大小
        self.batch_size = batch_size

        # 数据文件
        self._data_file = data_file
        # 忽略得标签（padding时使用）
        self._ignore_label = ignore_label

        # 图像均值
        self._image_mean = np.array((103.939, 116.779, 123.68), dtype=np.float32)

        # 获取文件列表
        input_data_file = tf.train.match_filenames_once(data_file)
        # 输入队列
        input_data_queue = tf.train.string_input_producer(input_data_file, num_epochs=1, shuffle=True)
        # 从队列中读取数据
        self._image, self._annotation = self._read_from_tf_record(input_data_queue)
        # 处理数据：数据增强
        self._image, self._annotation = self._process_data(self._image, self._annotation, self.input_size)

        # 成批次读取数据
        self.image_batch, self.annotation_batch = self._get_batch_data(self._image, self._annotation, self.batch_size)
        pass

    # 从队列中读取数据
    @staticmethod
    def _read_from_tf_record(input_data_queue):
        # 从队列钟读取样例
        _, serialized_example = tf.TFRecordReader().read(input_data_queue)
        features = tf.parse_single_example(serialized_example, features={
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "image/annotation/encoded": tf.FixedLenFeature([], tf.string)
        })
        image = tf.image.decode_jpeg(features["image/encoded"], channels=3)
        annotation = tf.image.decode_png(features["image/annotation/encoded"], channels=1)
        return image, annotation

    # 处理数据：数据增强
    @staticmethod
    def _process_data(image, annotation, input_size):
        image = tf.image.resize_images(image, input_size, method=tf.image.ResizeMethod.BILINEAR)
        annotation = tf.image.resize_images(annotation, input_size, method=tf.image.ResizeMethod.BILINEAR)
        return image, annotation

    # 成批次读取数据
    @staticmethod
    def _get_batch_data(image, annotation, batch_size):
        capacity = 1000 + 3 * batch_size
        image_batch, annotation_batch = tf.train.shuffle_batch([image, annotation], batch_size, capacity, 100)
        return image_batch, annotation_batch

    # test
    def test(self):
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(5):
                image_batch_data, annotation_batch_data = sess.run([self.image_batch, self.annotation_batch])
                pass

            coord.request_stop()
            coord.join(threads)
        pass

    pass


# 准备测试的数据
class ImageToTFRecordTest(object):

    def __init__(self, data_list, record_dir):
        self.data_list = data_list
        self.record_dir = record_dir
        pass

    @staticmethod
    def _int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _convert_to_example(self, filename, image_buffer, height, width):
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': self._int64_feature(height),
            'image/width': self._int64_feature(width),
            'image/filename': self._bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
            'image/encoded': self._bytes_feature(tf.compat.as_bytes(image_buffer))
        }))
        return example

    def _convert_to_tf_record(self):
        # 打开tf-record文件
        writer = tf.python_io.TFRecordWriter(self.record_dir)
        for image_file_index, image_file in enumerate(self.data_list):
            # 读取image(jpeg)数据
            with tf.gfile.FastGFile(image_file, 'rb') as ff:
                image_data = ff.read()
            im = Image.open(image_file)
            # 得到样例
            example = self._convert_to_example(os.path.basename(image_file), image_data, im.size[1], im.size[0])
            # 写入
            writer.write(example.SerializeToString())
        pass
        print("OK {}".format(self.record_dir))
        writer.close()
        pass

    def run(self):
        Tools.print("begin to transform image to records")
        self._convert_to_tf_record()
        Tools.print("end to transform image to records")
        pass
    pass


# 测试Data
class DataVOC2012Test(object):

    def __init__(self, data_file, number_classes, input_size, batch_size):
        # 类别数
        self.number_classes = number_classes
        # 输入大小
        self.input_size = input_size
        # 批次大小
        self.batch_size = batch_size

        # 数据文件
        self._data_file = data_file

        # 图像均值
        self._image_mean = np.array((103.939, 116.779, 123.68), dtype=np.float32)

        # color map
        self.label_colors = [(0, 0, 0),  # 0=background
                             (128, 0, 0), (0, 128, 0), (128, 128, 0),  # 1=aeroplane, 2=bicycle, 3=bird
                             (0, 0, 128), (128, 0, 128), (0, 128, 128),  # 4=boat, 5=bottle, 6=bus
                             (128, 128, 128), (64, 0, 0), (192, 0, 0),  # 7=car, 8=cat, 9=chair
                             (64, 128, 0), (192, 128, 0), (64, 0, 128),  # 10=cow, 11=diningtable, 12=dog
                             (192, 0, 128), (64, 128, 128), (192, 128, 128),  # 13=horse, 14=motorbike, 15=person
                             (0, 64, 0), (128, 64, 0), (0, 192, 0),   # 16=potted plant, 17=sheep, 18=sofa
                             (128, 192, 0), (0, 64, 128)]  # 19=train, 20=tv/monitor

        # 获取文件列表
        input_data_file = tf.train.match_filenames_once(data_file)
        # 输入队列
        input_data_queue = tf.train.string_input_producer(input_data_file, num_epochs=1, shuffle=True)
        # 从队列中读取数据
        self._image, self._size, self._filename = self._read_from_tf_record(input_data_queue)
        # 处理数据
        self._image = self._process_data(self._image, self.input_size)
        # 成批次读取数据
        self.image_batch, self.size_batch, self.filename_batch = tf.train.batch([self._image, self._size,
                                                                                 self._filename], self.batch_size)
        pass

    # 从队列中读取数据
    @staticmethod
    def _read_from_tf_record(input_data_queue):
        # 从队列钟读取样例
        _, serialized_example = tf.TFRecordReader().read(input_data_queue)
        features = tf.parse_single_example(serialized_example, features={
            "image/height": tf.FixedLenFeature([], tf.int64),
            "image/width": tf.FixedLenFeature([], tf.int64),
            "image/filename": tf.FixedLenFeature([], tf.string),
            "image/encoded": tf.FixedLenFeature([], tf.string)
        })
        image = tf.image.decode_jpeg(features["image/encoded"], channels=3)
        height = tf.cast(features["image/height"], tf.int32)
        width = tf.cast(features["image/width"], tf.int32)
        filename = tf.cast(features["image/filename"], tf.string)
        return image, [height, width], filename

    # 处理数据：数据增强
    @staticmethod
    def _process_data(image, input_size):
        image = tf.image.resize_images(image, input_size, method=tf.image.ResizeMethod.BILINEAR)
        return image

    # 保存结果
    def save_result(self, prediction_batch, size_batch, filename_batch, result_path):
        prediction_batch = np.asarray(prediction_batch, dtype=np.uint8)
        new_image = None
        for index, image in enumerate(prediction_batch):
            new_image = np.zeros(shape=[len(image), len(image[0]), 3], dtype=np.uint8) if new_image is None else new_image
            for x in range(len(image)):
                for y in range(len(image[0])):
                    val = list(self.label_colors[image[x][y]])
                    new_image[x, y, :] = val
                pass
            filename = str(filename_batch[index])
            filename = filename.split("'")[1] if "" in filename else filename
            im = Image.fromarray(new_image).convert("RGB").resize((size_batch[index][1], size_batch[index][0]))
            im.save(os.path.join(result_path, filename))
        pass

    # test
    def test(self):
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(5):
                image_data, size_data, filename_data = sess.run([self.image_batch, self.size_batch, self.filename_batch])
                self.save_result(np.squeeze(image_data[:, :, :, 0] % 20), size_data, filename_data,
                                 result_path="../dist/test/output")
                pass

            coord.request_stop()
            coord.join(threads)
        pass

    pass


if __name__ == '__main__':
    # 转换数据
    # ImageToTFRecordVOC2012().run()

    # 批次读取
    # DataVOC2012Train().test()

    # 转换数据
    # data_path = "../input"
    # record_dir = os.path.join(data_path, "test.tfrecord")
    # data_list = [os.path.join(data_path, data_file) for data_file in os.listdir(data_path)]
    # ImageToTFRecordTest(data_list, record_dir=record_dir).run()

    # 读取测试数据
    # DataVOC2012Test(data_file=record_dir, number_classes=19, input_size=list([513, 513]), batch_size=2).test()
    pass
