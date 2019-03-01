import tensorflow as tf
import numpy as np
from PIL import Image
import inference
import train
import math
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


e = math.e
# 接收的两个参数
# image_path = ''
# model = 0
#start = time.time()
def recognistion(image_path):
    tf.reset_default_graph()
    # 根据model参数值的不同来选择不同的模型
    # 得到测试图片
    image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read()
    x = tf.image.decode_jpeg(image_raw_data)
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, [-1, 32, 16, 1])
    # image = Image.open(image_path)
    # image = tf.cast(image,tf.float32)
    # x = tf.reshape(image,[32,16,1])
    # x = np.asarray(image.convert('L'))
    # x = np.asarray(Image.open(image_path).convert('L'))
    # 测试图片的前向传播过程
    y = inference.inference(x, None, None)

    # y_ = tf.array(y)
    # p = np.amax(y_, axis=0)
    # 得到置信度最高项的下标
    prediction = tf.argmax(y, 1)

    # 通过变量重命名的方式加载模型
    variable_averages = tf.train.ExponentialMovingAverage(train.MOVINGING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件
        ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # 加载模型
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 通过文件名得到模型保存时迭代的轮数
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            prediction,y_ = sess.run([prediction,y])
            p = math.pow(e, y_[0][prediction]) / (math.pow(e, y_[0][0])+math.pow(e, y_[0][1])+math.pow(e, y_[0][2])+math.pow(e, y_[0][3])+math.pow(e, y_[0][4])+math.pow(e, y_[0][5])
                                                    +math.pow(e, y_[0][6])+math.pow(e, y_[0][7])+math.pow(e, y_[0][8])+math.pow(e, y_[0][9])+math.pow(e, y_[0][10])+math.pow(e, y_[0][11])
                                                    +math.pow(e, y_[0][12])+math.pow(e, y_[0][13])+math.pow(e, y_[0][14])+math.pow(e, y_[0][15])+math.pow(e, y_[0][16])+math.pow(e, y_[0][17])
                                                    +math.pow(e, y_[0][18])+math.pow(e, y_[0][19])+math.pow(e, y_[0][20])+math.pow(e, y_[0][21])+math.pow(e, y_[0][22])+math.pow(e, y_[0][23])
                                                    +math.pow(e, y_[0][24])+math.pow(e, y_[0][25])+math.pow(e, y_[0][26])+math.pow(e, y_[0][27])+math.pow(e, y_[0][28])+math.pow(e, y_[0][29])
                                                    +math.pow(e, y_[0][30])+math.pow(e, y_[0][31])+math.pow(e, y_[0][32])+math.pow(e, y_[0][33])+math.pow(e, y_[0][34])+math.pow(e, y_[0][35])
                                                    +math.pow(e, y_[0][36])+math.pow(e, y_[0][37])+math.pow(e, y_[0][38])+math.pow(e, y_[0][39])+math.pow(e, y_[0][40])+math.pow(e, y_[0][41])
                                                    +math.pow(e, y_[0][42])+math.pow(e, y_[0][43])+math.pow(e, y_[0][44])+math.pow(e, y_[0][45])+math.pow(e, y_[0][46])+math.pow(e, y_[0][47])
                                                    +math.pow(e, y_[0][48])+math.pow(e, y_[0][49])+math.pow(e, y_[0][50])+math.pow(e, y_[0][51])+math.pow(e, y_[0][52])+math.pow(e, y_[0][53])
                                                    +math.pow(e, y_[0][54])+math.pow(e, y_[0][55])+math.pow(e, y_[0][56])+math.pow(e, y_[0][57])+math.pow(e, y_[0][58])+math.pow(e, y_[0][59])
                                                    +math.pow(e, y_[0][60])+math.pow(e, y_[0][61])+math.pow(e, y_[0][62])+math.pow(e, y_[0][63])+math.pow(e, y_[0][64]))
            if prediction == 0:
                prediction = '0'
            elif prediction == 1:
                prediction = '1'
            elif prediction == 2:
                prediction = '2'
            elif prediction == 3:
                prediction = '3'
            elif prediction == 4:
                prediction = '4'
            elif prediction == 5:
                prediction = '5'
            elif prediction == 6:
                prediction = '6'
            elif prediction == 7:
                prediction = '7'
            elif prediction == 8:
                prediction = '8'
            elif prediction == 9:
                prediction = '9'
            elif prediction == 10:
                prediction = 'A'
            elif prediction == 11:
                prediction = 'B'
            elif prediction == 12:
                prediction = 'C'
            elif prediction == 13:
                prediction = 'D'
            elif prediction == 14:
                prediction = 'E'
            elif prediction == 15:
                prediction = 'F'
            elif prediction == 16:
                prediction = 'G'
            elif prediction == 17:
                prediction = 'H'
            elif prediction == 18:
                prediction = 'J'
            elif prediction == 19:
                prediction = 'K'
            elif prediction == 20:
                prediction = 'L'
            elif prediction == 21:
                prediction = 'M'
            elif prediction == 22:
                prediction = 'N'
            elif prediction == 23:
                prediction = 'P'
            elif prediction == 24:
                prediction = 'Q'
            elif prediction == 25:
                prediction = 'R'
            elif prediction == 26:
                prediction = 'S'
            elif prediction == 27:
                prediction = 'T'
            elif prediction == 28:
                prediction = 'U'
            elif prediction == 29:
                prediction = 'V'
            elif prediction == 30:
                prediction = 'W'
            elif prediction == 31:
                prediction = 'X'
            elif prediction == 32:
                prediction = 'Y'
            elif prediction == 33:
                prediction = 'Z'
            elif prediction == 34:
                prediction = '藏'
            elif prediction == 35:
                prediction = '川'
            elif prediction == 36:
                prediction = '鄂'
            elif prediction == 37:
                prediction = '甘'
            elif prediction == 38:
                prediction = '赣'
            elif prediction == 39:
                prediction = '桂'
            elif prediction == 40:
                prediction = '贵'
            elif prediction == 41:
                prediction = '黑'
            elif prediction == 42:
                prediction = '沪'
            elif prediction == 43:
                prediction = '吉'
            elif prediction == 44:
                prediction = '冀'
            elif prediction == 45:
                prediction = '津'
            elif prediction == 46:
                prediction = '晋'
            elif prediction == 47:
                prediction = '京'
            elif prediction == 48:
                prediction = '辽'
            elif prediction == 49:
                prediction = '鲁'
            elif prediction == 50:
                prediction = '蒙'
            elif prediction == 51:
                prediction = '闽'
            elif prediction == 52:
                prediction = '宁'
            elif prediction == 53:
                prediction = '青'
            elif prediction == 54:
                prediction = '琼'
            elif prediction == 55:
                prediction = '陕'
            elif prediction == 56:
                prediction = '苏'
            elif prediction == 57:
                prediction = '皖'
            elif prediction == 58:
                prediction = '湘'
            elif prediction == 59:
                prediction = '新'
            elif prediction == 60:
                prediction = '渝'
            elif prediction == 61:
                prediction = '豫'
            elif prediction == 62:
                prediction = '粤'
            elif prediction == 63:
                prediction = '云'
            elif prediction == 64:
                prediction = '浙'
            #print('字符为：%s' % (prediction))
            return prediction,p
            # tf.reset_default_graph()

        else:
            print('No checkpoint file found')
            # tf.reset_default_graph()
            return
#end = time.time()

def main(argv=None):
    start = time.time()
    res=recognistion('/home/bupt/AlexNet/test.jpg')
    end = time.time()
    print(res)
    print(end-start)

if __name__ == "__main__":
    tf.app.run()
