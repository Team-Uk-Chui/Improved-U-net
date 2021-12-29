import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

w_init = tf.random_normal_initializer(stddev=0.02)  # tf.truncated_normal_initializer(stddev=0.02) #
b_init = tf.constant_initializer(value=0.0)

def resblock(inputs, n_filter, num):
    res = Conv2d(inputs, n_filter, (3, 3), act=None, padding="SAME", W_init=w_init, b_init=b_init, name=f'res1_{n_filter}_{num}')
    res.outputs = tf.nn.relu(res.outputs)
    res = Conv2d(res, n_filter, (3, 3), act=None, padding="SAME", W_init=w_init, b_init=b_init, name=f'res2_{n_filter}_{num}')

    res = ElementwiseLayer([inputs, res],  combine_fn=tf.add, name=f'short1_{n_filter}_{num}')
    return res

def pre(inputs, num):
    pre0 = Conv2d(inputs, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                     name=f'pre0{num}')
    rpre0 = pre0
    rpre0.outputs = tf.nn.relu(rpre0.outputs)
    pre1 = Conv2d(rpre0, 16, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                     name=f'pre1{num}')
    rpre1 = pre1
    out_pre = ElementwiseLayer([rpre1, inputs], combine_fn=tf.add, name=f'out_pre{num}')
    return out_pre

def post(inputs, con_layer, n_filter, num):
    post1 = ConcatLayer([inputs, con_layer], concat_dim=3, name=f'post_concat{num}')  # 32
    rpost1 = post1
    p_conv1 = Conv2d(rpost1, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                     name=f'p_conv1{num}')
    rpconv1 = p_conv1
    rpconv1.outputs = tf.nn.relu(rpconv1.outputs)
    p_conv2 = Conv2d(rpconv1, n_filter, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                     name=f'p_conv2{num}')
    rp_conv2 = ElementwiseLayer([p_conv2, post1], combine_fn=tf.add, name='residual4')
    return rp_conv2


def MyUnet(t_image, reuse=False):

    nx = int(t_image._shape[1])
    ny = int(t_image._shape[2])
    nz = int(t_image._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    with tf.variable_scope("MyUnet", reuse=reuse) as vs:
        inputs = InputLayer(t_image, name='inputs')
        conv0_1 = Conv2d(inputs, 16, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='inputs')
        pre_1 = pre(conv0_1, 1)
        rpre_1 = pre_1
        pre_2 = pre(rpre_1, 2)
        rpre_2 = pre_2
        pre_3 = pre(rpre_2, 3)
        rpre_3 = pre_3

        ##  conv2
        rconv1_1 = rpre_3
        rconv1_1.outputs = tf.nn.relu(rconv1_1.outputs)
        conv2_1 = resblock(rconv1_1, 16, 0)
        print('conv2_1 : ', conv2_1.outputs)
        rconv2_1 = conv2_1
        rconv2_1.outputs = tf.nn.relu(rconv2_1.outputs)
        conv2_2 = Conv2d(rconv2_1, 32, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv2_2')
        print('conv2_2 : ', conv2_2.outputs)
        conv2_3 = conv2_2
        # print('conv2_3 : ', conv2_3.outputs)
        pool2 = MaxPool2d(conv2_3, (2, 2), padding='SAME', name='pool2')
        print('pool2 : ', pool2.outputs)

        ##  conv3
        rpool2 = pool2
        rpool2.outputs = tf.nn.relu(rpool2.outputs)
        conv3_1 = resblock(rpool2, 32, 1)
        print('conv3_1 : ', conv3_1.outputs)
        rconv3_1 = conv3_1
        rconv3_1.outputs = tf.nn.relu(rconv3_1.outputs)
        conv3_2 = Conv2d(rconv3_1, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv3_2')
        print('conv3_2 : ', conv3_2.outputs)
        conv3_3 = conv3_2
        print('conv3_3 : ', conv3_3.outputs)
        pool3 = MaxPool2d(conv3_3, (2, 2), padding='SAME', name='pool3')
        print('pool3 : ', pool3.outputs)

        ##  conv4
        rpool3 = pool3
        rpool3.outputs = tf.nn.relu(rpool3.outputs)
        conv4_1 = resblock(rpool3, 64, 1)
        print('conv4_1 : ', conv4_1.outputs)
        rconv4_1 = conv4_1
        rconv4_1.outputs = tf.nn.relu(rconv4_1.outputs)
        conv4_2 = Conv2d(rconv4_1, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv4_2')
        print('conv4_2 : ', conv4_2.outputs)
        conv4_3 = conv4_2
        print('conv4_3 : ', conv4_3.outputs)
        pool4 = MaxPool2d(conv4_3, (2, 2), padding='SAME', name='pool4')
        print('pool4 : ', pool4.outputs)

        ##  conv5
        rpool4 = pool4
        rpool4.outputs = tf.nn.relu(rpool4.outputs)
        conv5_1 = resblock(rpool4, 128, 1)
        print('conv5_1 : ', conv5_1.outputs)
        rconv5_1 = conv5_1
        rconv5_1.outputs = tf.nn.relu(rconv5_1.outputs)
        conv5_2 = Conv2d(rconv5_1, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='conv5_2')
        print('conv5_2 : ', conv5_2.outputs)
        conv5_3 = conv5_2
        print(" * After conv: %s" % conv5_2.outputs)
        pool5 = MaxPool2d(conv5_3, (2, 2), padding='SAME', name='pool5')
        print('pool5 : ', pool5.outputs)

        ##  Bridge
        rpool5 = pool5
        rpool5.outputs = tf.nn.relu(rpool5.outputs)
        Bridge1 = Conv2d(rpool5, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='Bridge1')
        print('conv5_1 : ', conv5_1.outputs)
        rBridge1 = Bridge1
        rBridge1.outputs = tf.nn.relu(rBridge1.outputs)
        Bridge2 = Conv2d(rBridge1, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                         name='Bridge2')
        print('conv5_2 : ', conv5_2.outputs)
        Bridge3 = Bridge2
        print(" * After conv: %s" % Bridge3.outputs)

        ##  up5
        rBridge3 = Bridge3
        rBridge3.outputs = tf.nn.relu(rBridge3.outputs)
        # up5_0 = UpSampling2dLayer(rBridge3, size=(2, 2), is_scale=True, method=1, name='up5_0')
        # up5_1 = Conv2d(up5_0, 256, (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
        #                  name='up5_1')
        up5_1 = DeConv2d(rBridge3, 256, (2, 2), strides=(2, 2),
                         padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up5_1')
        print('Deconv1 : ', up5_1.outputs)
        # print('up5 : ', up5_1.outputs)
        up5_2 = ConcatLayer([up5_1, conv5_3], concat_dim=3, name='up5_2')
        print('concat5 : ', up5_2.outputs)
        rup5_2 = up5_2
        rup5_2.outputs = tf.nn.relu(rup5_2.outputs)
        uconv5_1 = resblock(rup5_2, 512, 2)
        ruconv5_1 = uconv5_1
        ruconv5_1.outputs = tf.nn.relu(ruconv5_1.outputs)
        uconv5_2 = Conv2d(ruconv5_1, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv5_2')
        print('uconv5_2 : ', uconv5_2.outputs)
        uconv5_3 = uconv5_2
        print(uconv5_3.outputs)

        ##  up4
        ruconv5_3 = uconv5_3
        ruconv5_3.outputs = tf.nn.relu(ruconv5_3.outputs)
        # up4_0 = UpSampling2dLayer(ruconv5_3, size=(2, 2), is_scale=True, method=1, name='up4_0')
        # up4_1 = Conv2d(up4_0, 128, (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
        #                  name='up4_1')
        up4_1 = DeConv2d(ruconv5_3, 128, (2, 2), strides=(2, 2),
                         padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up4_1')
        # print('up4 : ', up4_0.outputs)
        up4_2 = ConcatLayer([up4_1, conv4_3], concat_dim=3, name='up4_2')
        print('concat4 : ', up4_2.outputs)
        rup4_2 = up4_2
        rup4_2.outputs = tf.nn.relu(rup4_2.outputs)
        uconv4_1 = resblock(rup4_2, 256, 2)
        ruconv4_1 = uconv4_1
        ruconv4_1.outputs = tf.nn.relu(ruconv4_1.outputs)
        uconv4_2 = Conv2d(ruconv4_1, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv4_2')
        print('uconv4_2 : ', uconv4_2.outputs)
        uconv4_3 = uconv4_2
        print(uconv4_3.outputs)

        ##  up3
        ruconv4_3 = uconv4_3
        ruconv4_3.outputs = tf.nn.relu(ruconv4_3.outputs)
        # up3_0 = UpSampling2dLayer(ruconv4_3, size=(2, 2), is_scale=True, method=1, name='up3_0')
        # up3_1 = Conv2d(up3_0, 64, (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
        #                  name='up3_1')
        up3_1 = DeConv2d(ruconv4_3, 64, (2, 2), strides=(2, 2), padding='SAME', act=None,
                         W_init=w_init, b_init=b_init, name='up3_1')
        # print('up3 : ', up3_0.outputs)
        up3_2 = ConcatLayer([up3_1, conv3_3], concat_dim=3, name='up3_2')
        print('concat3 : ', up3_2.outputs)
        rup3_2 = up3_2
        rup3_2.outputs = tf.nn.relu(rup3_2.outputs)
        uconv3_1 = resblock(rup3_2, 128, 2)
        print('uconv3_1 : ', uconv3_1.outputs)
        ruconv3_1 = uconv3_1
        ruconv3_1.outputs = tf.nn.relu(ruconv3_1.outputs)
        uconv3_2 = Conv2d(ruconv3_1, 32, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv3_2')
        print('uconv3_2 : ', uconv3_2.outputs)
        uconv3_3 = uconv3_2
        print(uconv3_3.outputs)

        ##  up2
        ruconv3_3 = uconv3_3
        ruconv3_3.outputs = tf.nn.relu(ruconv3_3.outputs)
        # up2_0 = UpSampling2dLayer(ruconv3_3, size=(2, 2), is_scale=True, method=1, name='up2_0')
        # up2_1 = Conv2d(up2_0, 32, (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
        #                  name='up2_1')
        up2_1 = DeConv2d(ruconv3_3, 32, (8, 8), strides=(2, 2),
                         padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up2_1')
        # print('up2 : ', up2_0.outputs)
        up2_2 = ConcatLayer([up2_1, conv2_3], concat_dim=3, name='up2_2')
        print('concat2 : ', up2_2.outputs)
        rup2_2 = up2_2
        rup2_2.outputs = tf.nn.relu(rup2_2.outputs)
        uconv2_1 = resblock(rup2_2, 64, 2)
        print('uconv2_1 : ', uconv2_1.outputs)
        ruconv2_1 = uconv2_1
        ruconv2_1.outputs = tf.nn.relu(ruconv2_1.outputs)
        uconv2_2 = Conv2d(ruconv2_1, 16, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='uconv2_2')
        print('uconv2_2 : ', uconv2_2.outputs)
        uconv2_3 = uconv2_2
        print(uconv2_3.outputs)

        post_1 = post(uconv2_3, pre_1, 32, 1)
        post_2 = post(post_1, pre_2, 48, 2)
        post_3 = post(post_2, pre_3, 64, 3)


        # Last stage 1 layer
        ruconv2_3 = post_3
        ruconv2_3.outputs = tf.nn.relu(ruconv2_3.outputs)
        uconv1_1 = Conv2d(ruconv2_3, 3, (3, 3), act=tf.nn.tanh, name='uconv1_1')
        print(" * Output: %s" % uconv1_1.outputs)
        return uconv1_1

        #resblock
