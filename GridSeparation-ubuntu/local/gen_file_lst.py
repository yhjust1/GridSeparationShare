import fnmatch
import os
import pandas as pd
import numpy as np
import sys

import tensorflow as tf
# InputStra = sys.argv[1]
# InputStrb = sys.argv[2]

# 参数：
#     Stra:tfrecords类型的数据文件存放目录
#     Strb:文件类型
#     type tr or cv
#     data_dir:数据的存放目录
#     save_dir:数据文件名列表的存放目录，应与run_lstm.py中的lists_dir参数保持一致
#
def ReadSaveAddr(Stra,Strb,type,data_dir,save_dir):

    a_list = fnmatch.filter(os.listdir(Stra),Strb)
    if (type == 'tr'):
        file_dir = data_dir+'tr'
    else:
        file_dir = data_dir+'cv'
    for i in range(len(a_list)):
        a_list[i] = file_dir+'/'+a_list[i]
    print("Find = ",len(a_list))
    df = pd.DataFrame(np.array(a_list).reshape((len(a_list),1)),columns=['Addr'])#
    # df.Addr = a_list
    #print(df.head())
    if(type=='tr'):
        file_name = save_dir + '/tr.lst'
    else:
        file_name = save_dir + '/cv.lst'
    df.to_csv(file_name,columns=['Addr'],index=False,header=False)


if __name__ == "__main__":
    save_dir='D:/Grid'
    data_dir='D:/Grid/data/input-2D/'
    InputStra='D:/Grid/data/input-2D/tr'
    InputStrb='*.tfrecords'
    ReadSaveAddr(InputStra,InputStrb,'tr',data_dir,save_dir)
    InputStra = 'D:/Grid/data/input-2D/cv'
    InputStrb = '*.tfrecords'
    ReadSaveAddr(InputStra, InputStrb, 'cv',data_dir,save_dir)