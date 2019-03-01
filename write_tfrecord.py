import tensorflow as tf 
import numpy as np 
import os

path = '/home/mitsui/AlexNet/dataset/test/浙'
LABEL = 64
HEIGHT = 32
WIDTH = 16
CHANNELS = 1

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

filename = '/home/mitsui/AlexNet/tfrecord1/test/output_64.tfrecords'

writer = tf.python_io.TFRecordWriter(filename)

dirs = os.listdir(path)

with tf.Session() as sess:
	for image in dirs:
		image1 = tf.gfile.FastGFile('/home/mitsui/AlexNet/dataset/test/浙/%s' % image,'rb').read()  
		image2 = tf.image.decode_jpeg(image1)
		image3 = sess.run(image2)  
		image4 = image3.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			'image':_bytes_feature(image4),
			'label':_int64_feature(LABEL),
			'height':_int64_feature(HEIGHT),
			'width':_int64_feature(WIDTH),
			'channels':_int64_feature(CHANNELS)}))
		writer.write(example.SerializeToString())
writer.close()
