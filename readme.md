# Semantic Segmentation Integrate in TensorFlow
I want to integrate semantic segmentation net that encountered in my learning process.
I don't know whether this idea can be successful or not, however, I'm willing to give a try.


## Contains
- [ ] FCN
- [ ] SegNet
- [ ] UNet
- [ ] RefineNet
- [x] PSPNet
- [ ] Large Kernel Matters
- [ ] DeepLab V1
- [ ] DeepLab V2
- [x] DeepLab V3


## Files
* `data`
    * `VOC2012`
        * 将image和annotation转成tfrecord
    * `MSCOCO`

* `dist`
    * 运行过程中生成的所有文件都存放在dist

* `input`
    * 测试用的图片
        * 测试时会先将`指定文件夹`（如`input/test`）下的图片转成`tfrecord`（如`input/test.tfrecord`）格式，
        其中保存了`图片名称`、`图片大小`、`图片数据`等信息。

* `model`
    * 各个模型

* `pre_train`
    * 各路大神在各大数据集上的预训练

* `readme`
    * readme图片信息

* `runner`
    * 运行器，包括`Train`,`Val`,`Test`父类以及各个模型的子类

* `tools`
    * `Tools.py`：常用函数
    * `Data.py`：用于提供数据
        * `ImageToTFRecordVOC2012`：将VOC2012转成tfrecord
        * `DataAug`：专注于数据增强
        * `DataVOC2012Train`：提供训练数据，有`减均值`、`随机缩放`、`随机反转`等数据增强
        * `DataVOC2012Val`：提供验证数据，有`减均值`
        * `ImageToTFRecordTest`：将测试图片转成`tfrecord`
        * `DataVOC2012Test`：提供测试数据，配合`ImageToTFRecordTest`使用，有`减均值`

* `main.py`
    * 主函数


## Reference
1. FCN
    * [FCN](https://github.com/alisure-ml/Semantic-Segmentation-FCN)
    * [Paper](https://github.com/alisure-ml/Semantic-Segmentation-FCN)
