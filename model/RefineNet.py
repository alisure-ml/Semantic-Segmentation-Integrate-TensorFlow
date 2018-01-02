import tensorflow as tf
from tensorflow.contrib import slim
import RefineNet_resnet_v1 as resnet_v1

# 根据比例进行上采样：
# P5:This score map is then up-sampled to match the original image using bilinear interpolation.
def unpool(inputs, scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale])


# 基本的残差单元
def ResidualConvUnit(inputs, features=256, kernel_size=3):
    net = tf.nn.relu(inputs)
    net = slim.conv2d(net, features, kernel_size)
    net = tf.nn.relu(net)
    net = slim.conv2d(net, features, kernel_size)
    net = tf.add(net, inputs)
    return net


# 多分辨率融合
def MultiResolutionFusion(high_inputs=None, low_inputs=None, up0=2, up1=1, n_i=256):
    g0 = unpool(slim.conv2d(high_inputs, n_i, 3), scale=up0)

    if low_inputs is None:
        return g0

    g1 = unpool(slim.conv2d(low_inputs, n_i, 3), scale=up1)
    return tf.add(g0, g1)


# 链式残差池化
def ChainedResidualPooling(inputs, n_i=256):
    net_relu = tf.nn.relu(inputs)
    net = slim.max_pool2d(net_relu, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(net, n_i, 3)
    return tf.add(net, net_relu)


# RefineNet块
def RefineBlock(high_inputs=None, low_inputs=None):
    if low_inputs is not None:
        print(high_inputs.shape)
        rcu_high = ResidualConvUnit(high_inputs, features=256)
        rcu_low = ResidualConvUnit(low_inputs, features=256)
        fuse = MultiResolutionFusion(rcu_high, rcu_low, up0=2, up1=1, n_i=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_i=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output
    else:
        rcu_high = ResidualConvUnit(high_inputs, features=256)
        fuse = MultiResolutionFusion(rcu_high, low_inputs=None, up0=1, n_i=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_i=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output


# 模型
def RefineNet_model(images, weight_decay=1e-5, is_training=True):

    # 101层残差
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_101(images, is_training=is_training, scope='resnet_v1_101')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))

            g = [None, None, None, None]
            h = [None, None, None, None]

            for i in range(4):
                h[i] = slim.conv2d(f[i], 256, 1)
            for i in range(4):
                print('Shape of h_{} {}'.format(i, h[i].shape))

            g[0] = RefineBlock(h[0])
            g[1] = RefineBlock(g[0], h[1])
            g[2] = RefineBlock(g[1], h[2])
            g[3] = RefineBlock(g[2], h[3])

            g[3] = unpool(g[3], scale=4)
            f_score = slim.conv2d(g[3], 21, 1, activation_fn=tf.nn.relu, normalizer_fn=None)

    return f_score
