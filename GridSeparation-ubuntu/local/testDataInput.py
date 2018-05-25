import fnmatch
import os
from io_funcs.signal_processing import stft
import tensorflow as tf
import numpy as np
import csv


def gen_feats(dir,file_name,type):
    #STFT 暂未使用
    # mix_stft = stft(mix_wav, time_dim=0, size=256, shift=128)

    file=dir + '/' + file_name
    featCSV = open(file, "r")
    Input_reder = csv.reader(featCSV)
    i=0
    write_index=0
    count=0#一年中的周计数
    countH=0#一周中的假期树
    for line in Input_reder:
        if i==0:
            i =i+1
        else:
            if(write_index==0):#新的一周开始
                countH=0
                mix_input = np.zeros([96*7,2])#母线
                single1 = np.zeros([96*7,1])#支线1
                single2 = np.zeros([96*7, 1])  # 支线2
                single3 = np.zeros([96 * 7, 1])  # 支线3
                single4 = np.zeros([96 * 7, 1])  # 支线4
                single5 = np.zeros([96 * 7, 1])  # 支线5
            #######################################
            if line[1]=='1.0' :
                countH =countH+1;

            mix_input[write_index,0]=line[2]
            mix_input[write_index, 1] = line[3]
            # mix_input[write_index, 2] = line[2]
            # mix_input[write_index, 3] = line[3]
            #######################################
            # single1[write_index, 0] = line[2]
            single1[write_index, 0] = line[4]
            # single1[write_index, 2] = line[2]
            # single1[write_index, 3] = line[4]
            #######################################
            # single2[write_index, 0] = line[2]
            single2[write_index, 0] = line[5]
            # single2[write_index, 2] = line[2]
            # single2[write_index, 3] = line[5]
            # #######################################
            # single3[write_index, 0] = line[2]
            single3[write_index, 0] = line[6]
            # #######################################
            # single4[write_index, 0] = line[2]
            single4[write_index, 0] = line[7]
            # #######################################
            # single5[write_index, 0] = line[2]
            single5[write_index, 0] = line[8]

            if (write_index == (96*7-1) and countH // 96<3):#一周输入完成,写入tfrecords文件
                # if countH > 2:#去除含有特殊假日的数据
                #     break
                tf_name=file_name+str(count)
                count += 1
                if type =='tr':
                    tfrecords_name = 'D:/Grid/data/input-5D-min-b/tr/' + tf_name + '.tfrecords'
                else:
                    tfrecords_name = 'D:/Grid/data/input-5D-min-b/cv/' + tf_name + '.tfrecords'
                print(tfrecords_name)
                with tf.python_io.TFRecordWriter(tfrecords_name) as writer:
                    inputs = np.concatenate((mix_input, mix_input), axis=1)
                    labels = np.concatenate((single1, single2, single3, single4, single5), axis=1)#
                    print(labels.shape)
                    ex = make_sequence_example(inputs, labels)
                    writer.write(ex.SerializeToString())

            write_index +=1
            write_index %=(96*7)

    print(count)

def make_sequence_example(inputs, labels):
    """Returns a SequenceExample for the given inputs, labels and genders
    Args:
        inputs: A list of input vectors. Each input vector is a list of floats.
        labels: A list of label vectors. Each label vector is a list of floats.
        genders: A 1*2 vector [0, 1], [0,1], [1,1], [0, 0]
    Returns:
        A tf.train.SequenceExample containing inputs and labels(optional).
    """
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]
    label_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=label))
        for label in labels]
    # gender_features = [ tf.train.Feature(float_list=tf.train.FloatList(value=genders)) ]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features)#,
         # 'genders': tf.train.FeatureList(feature=gender_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)

def get_file_list(data_dir):
    Stra=data_dir
    Strb='*.csv'
    a_list = fnmatch.filter(os.listdir(Stra), Strb)
    for i in range(len(a_list)):
        # a_list[i] = Stra + '/' + a_list[i]
        if i>len(a_list)*0.8:
            gen_feats(Stra , a_list[i],'cv')
        else:
            gen_feats(Stra , a_list[i],'tr')


if __name__ == "__main__":
    # name='011a010a.wav'
    # gen_feats(name)
    # name='440c040j.wav'
    # gen_feats(name)
    data_dir = 'D:/组会/电力/相关代码/csvData/5_min/clean'
    get_file_list(data_dir)