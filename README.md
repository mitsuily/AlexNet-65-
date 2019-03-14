# 基于Tensorflow实现AlexNet

AlexNet卷积神经网络实现大陆车牌单个字符的65分类，提供训练和测试数据集的tfrecord文件。

inference文件是AlexNet网络的前向计算过程，其中有网络中卷积核大小、输入层图片大小等超参数，可以进行修改。

train文件是模型的训练文件，其中定义了batch_size，模型保存的路径等信息，训练集图片格式是tfrecord，在训练自己的模型之前应当将数据集转化成这种格式，其中的write_tfrecord文件提供了一个格式可以参考如何将输入图片转化成tfrecord格式。在训练时，直接python train.py即可。

eval文件时用于模型的准确率测试，在该文件中每隔60秒保存一次模型，在该模型上测试准确率。

recognition文件提供了一个接口，输入一个图片，加载模型，给出识别结果。

image是一个脚本文件，用于调整图片大小。

对于字符的tfrecord文件，陆续上传中。。。
