import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#配置神经网络的参数
INPUT_NODE = 1024
OUTPUT_NODE = 65

IMAGE_SIZE = 32
NUM_CHANNELS = 1
NUM_LABELS = 65

#第一个卷积层的尺寸和深度
CONV1_DEEP = 8
CONV1_SIZE = 2
#第二个卷积层的尺寸和深度
CONV2_DEEP = 16
CONV2_SIZE = 2
#第三个卷积层的尺寸和深度
CONV3_DEEP = 32
CONV3_SIZE = 2
#第四个卷积层的尺寸和深度
CONV4_DEEP = 64
CONV4_SIZE = 2
#第五个卷积层的尺寸和深度
CONV5_DEEP = 128
CONV5_SIZE = 2
#全连接层的节点个数
FCNN1_NODE = 260
FCNN2_NODE = 260

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weights',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        #if regularizer != None:
            #tf.add_to_collection('losses',regularizer(conv1_weights))
        conv1_biases = tf.get_variable('bias',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weights',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        #if regularizer != None:
            #tf.add_to_collection('losses',regularizer(conv2_weights))
        conv2_biases = tf.get_variable('bias',[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    with tf.variable_scope('layer5-conv3'):
        conv3_weights = tf.get_variable('weights',[CONV3_SIZE,CONV3_SIZE,CONV2_DEEP,CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        #if regularizer != None:
            #tf.add_to_collection('losses',regularizer(conv2_weights))
        conv3_biases = tf.get_variable('bias',[CONV3_DEEP],initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2,conv3_weights,strides=[1,1,1,1],padding='VALID')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))


    with tf.variable_scope('layer6-conv4'):
        conv4_weights = tf.get_variable('weights',[CONV4_SIZE,CONV4_SIZE,CONV3_DEEP,CONV4_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        #if regularizer != None:
            #tf.add_to_collection('losses',regularizer(conv2_weights))
        conv4_biases = tf.get_variable('bias',[CONV4_DEEP],initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3,conv4_weights,strides=[1,1,1,1],padding='VALID')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))

    with tf.variable_scope('layer7-conv5'):
        conv5_weights = tf.get_variable('weights',[CONV5_SIZE,CONV5_SIZE,CONV4_DEEP,CONV5_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        #if regularizer != None:
            #tf.add_to_collection('losses',regularizer(conv2_weights))
        conv5_biases = tf.get_variable('bias',[CONV5_DEEP],initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4,conv5_weights,strides=[1,1,1,1],padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5,conv5_biases))

    with tf.name_scope('layer8-pool3'):
        pool3 = tf.nn.max_pool(relu5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


        pool_shape = pool3.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool3,[-1,nodes])
    

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable('weights',[nodes,FCNN1_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[FCNN1_NODE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer9-fc2'):
        fc2_weights = tf.get_variable('weights',[FCNN1_NODE,FCNN2_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[FCNN2_NODE],initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weights)+fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2,0.5)

    with tf.variable_scope('layer10-fc3'):
        fc3_weights = tf.get_variable('weights',[FCNN2_NODE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2,fc3_weights) + fc3_biases

    return logit
