import tensorflow as tf
import numpy as np
import os
import inference

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_files = tf.train.match_filenames_once('/home/bupt/AlexNet/tfrecord1/train/output_*')
#定义parser方法从TFRecord中解析数据。
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64),
            'height':tf.FixedLenFeature([],tf.int64),
            'width':tf.FixedLenFeature([],tf.int64),
            'channels':tf.FixedLenFeature([],tf.int64),
        }
    )
    decoded_image = tf.decode_raw(features['image'],tf.uint8)
    decoded_image = tf.cast(decoded_image,tf.float32)

    decoded_image = tf.reshape(decoded_image,[32,16,1])
    # decoded_image.set_shape([32,32,1])
    #label = features['label']
    label = tf.cast(features['label'],tf.int32)
    return decoded_image,label

BATCH_SIZE = 128          #定义组合数据batch的大小
shuffle_buffer = 1000    #定义随机打乱顺序时buffer的大小

#定义读取训练数据的数据集
dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)
#对数据进行shuffle和batching操作
dataset = dataset.shuffle(shuffle_buffer).batch(BATCH_SIZE)

dataset = dataset.repeat(800)

#定义数据集迭代器
iterator = dataset.make_initializable_iterator()
image_batch,label_batch = iterator.get_next()

#配置神经网络的参数
#BATCH_SIZE = 10
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.001
#TRAINING_STEP = 20000
MOVINGING_AVERAGE_DECAY = 0.99

#CHINESE = 8402
#ENGLISH = 40962
DATASET = 49364
#模型保存的路径和文件名
MODEL_SAVE_PATH = '/home/bupt/AlexNet/model_small'
MODEL_NAME = 'model.ckpt'

def train():
    x = image_batch
    y_ = label_batch
    y_10 = tf.one_hot(y_,depth=65)
    # x = tf.placeholder(tf.float32,[BATCH_SIZE,digital_inference.IMAGE_SIZE,digital_inference.IMAGE_SIZE,digital_inference.NUM_CHANNELS],name='x-input')
    # y_ = tf.placeholder(tf.float32,[None,digital_inference.OUTPUT_NODE],name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #直接使用digital_inference中定义的前向传播过程
    y = inference.inference(x,train,regularizer)
    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVINGING_AVERAGE_DECAY,global_step)
    variables_average_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_10,1))

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_10)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,DATASET/BATCH_SIZE,LEARNING_RATE_DECAY)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_average_op]):
        train_op = tf.no_op(name='train')

    #初始化TensorFlow持久化类
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(),tf.local_variables_initializer()))

        sess.run(iterator.initializer)

    #     #在训练的过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
    #     for i in range(TRAINING_STEP):
    #         # xs,ys = image_batch,label_batch
    #         # xs = np.reshape(xs,(BATCH_SIZE,
    #         #                     digital_inference.IMAGE_SIZE,
    #         #                     digital_inference.IMAGE_SIZE,
    #         #                     digital_inference.NUM_CHANNELS))
    #         _,loss_value,step = sess.run([train_op,loss,global_step])
    
        while True:
            try:
                _,loss_value,step = sess.run([train_op,loss,global_step])
                if step % 380 == 0:
                    print('After %d training step(s),loss on training batch is %g' % (step,loss_value))

                    saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
            except tf.errors.OutOfRangeError as e:
                print(e)
                break
    #         #每1000轮保存一次模型
    #         if i % 100 == 0:
    #             print('After %d training step(s),loss on training batch is %g' % (step,loss_value))
    #             #sess.run(iterator.initializer)
    #         saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()


