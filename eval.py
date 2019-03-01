import time
import tensorflow as tf
import numpy as np
import os
import inference
import train
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

test_files = tf.train.match_filenames_once('/home/bupt/AlexNet/tfrecord1/test/output_*')
EVAL_INTERVAL_SECS = 50   #每60秒加载加载一次最新的模型，并在测试数据上测试最新模型的正确率

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
    #decoded_image.set_shape([features['height'],features['width'],features['channels']])
    #label = features['label']
    label = tf.cast(features['label'],tf.int32)
    return decoded_image,label
batch_size = 2695
test_dataset = tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(parser)
test_dataset = test_dataset.batch(batch_size)

#定义测试数据上的迭代器
test_iterator = test_dataset.make_initializable_iterator()
test_image_batch,test_label_batch = test_iterator.get_next()


def evaluate():
    #with tf.Graph().as_default() as g:
    x = test_image_batch
    y_ = test_label_batch
    y_10 = tf.one_hot(y_,depth=65)
    #直接通过调用封装好的函数来计算前向传播的结果
    y = inference.inference(x,None,None)
    #predictions = tf.argmax(y_10,axis=-1,output_type=tf.int64)

        
    #使用前向传播的结果计算正确率，
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_10,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #通过变量重命名的方式来加载模型
    variable_averages = tf.train.ExponentialMovingAverage(train.MOVINGING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    #每隔60秒调用一次计算正确率的过程以检验训练过程中正确率的变化
    while True:
        with tf.Session() as sess:
            sess.run((tf.global_variables_initializer(),tf.local_variables_initializer()))
            sess.run(test_iterator.initializer)

            test_results = []
            #tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                #加载模型
                saver.restore(sess,ckpt.model_checkpoint_path)
                #通过文件名得到模型保存时迭代的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy)
                print('After %s training step(s),test accuracy = %g' % (global_step,accuracy_score))
            else:
                print('No checkpoint file found')
                return
        time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    evaluate()

if __name__ == "__main__":
    tf.app.run()
