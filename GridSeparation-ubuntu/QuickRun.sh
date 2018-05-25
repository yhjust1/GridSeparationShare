#! /bin/bash
step=0  #程序运行行为控制
lists_dir=~/data #记录所有数据地址的列表文件的存放地址
num_threads=12
tfrecords_dir=~/data/input-2D   #数据的保存地址，即cv,tr文件夹父级目录
if [ $step -le 0 ]; then
   #读取数据
   echo "Start generate data list."
   python3 ./local/gen_file_lst.py --lists_dir=$lists_dir --tfrecords_dir=$tfrecords_dir
fi

min_epochs=0    #最小训练次数
max_epochs=30   #最大训练次数
output_dir=~/Grid/data/2/output-2D #最终结果的保存位置

gpu_id='0'  #GPU的ID，
TF_CPP_MIN_LOG_LEVEL=1  #默认为1，为屏幕中的提示
rnn_num_layers=3    #网络层数
batch_size=32    #批处理的个数，即一次性训练使用样本个数

input_size=2        #输入数据的维度
output_size=1       #输出数据的维度

cost_right=0.001

rnn_size=512        #每层网络节点数
keep_prob=0.8       #LSTM设置
learning_rate=0.000005 #学习步长
end_halving_impr=0.01
halving_factor=0.7
model_type=BLSTM    #网络类型，

prefix=GridSeparation
assignment=def
output_dir=${output_dir}_${rnn_num_layers}_${rnn_size}_${cost_right}
name=${prefix}_${model_type}_${rnn_num_layers}_${rnn_size}_ReLU
save_dir=exp/$name/
data_dir=data/separated/${name}_${assignment}/
resume_training=false

# tfrecords are stored in data/input/{tr, cv}/

if [ $step -le 1 ]; then
    
    echo "Start Traing RNN(LSTM or BLSTM) model."
    decode=0
    #python3 run_lstm.py --lists_dir=$lists_dir --rnn_num_layers=$rnn_num_layers --batch_size=$tr_batch_size --rnn_size=$rnn_size --min_epochs=$min_epochs --max_epochs=$max_epochs --output_dir=$output_dir
    tr_cmd="python3 run_lstm.py \
    --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --min_epochs=$min_epochs --max_epochs=$max_epochs --output_dir=$output_dir \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --halving_factor=$halving_factor --end_halving_impr=$end_halving_impr \
    --cost_right=$cost_right"
    echo $tr_cmd
    #GPU版本使用下面的代码执行Python文件
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd
    cost_right=0.01
    output_dir=${output_dir}_${rnn_num_layers}_${rnn_size}_${cost_right}
    tr_cmd="python3 run_lstm.py \
    --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --min_epochs=$min_epochs --max_epochs=$max_epochs --output_dir=$output_dir \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --halving_factor=$halving_factor --end_halving_impr=$end_halving_impr \
    --cost_right=$cost_right"
    echo $tr_cmd
    #GPU版本使用下面的代码执行Python文件
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd
    cost_right=0.1
    output_dir=${output_dir}_${rnn_num_layers}_${rnn_size}_${cost_right}
    tr_cmd="python3 run_lstm.py \
    --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --min_epochs=$min_epochs --max_epochs=$max_epochs --output_dir=$output_dir \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --halving_factor=$halving_factor --end_halving_impr=$end_halving_impr \
    --cost_right=$cost_right"
    echo $tr_cmd
    #GPU版本使用下面的代码执行Python文件
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd

    cost_right=1
    output_dir=${output_dir}_${rnn_num_layers}_${rnn_size}_${cost_right}
    tr_cmd="python3 run_lstm.py \
    --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --min_epochs=$min_epochs --max_epochs=$max_epochs --output_dir=$output_dir \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --halving_factor=$halving_factor --end_halving_impr=$end_halving_impr \
    --cost_right=$cost_right"
    echo $tr_cmd
    #GPU版本使用下面的代码执行Python文件
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd

    cost_right=10
    output_dir=${output_dir}_${rnn_num_layers}_${rnn_size}_${cost_right}
    tr_cmd="python3 run_lstm.py \
    --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --min_epochs=$min_epochs --max_epochs=$max_epochs --output_dir=$output_dir \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --halving_factor=$halving_factor --end_halving_impr=$end_halving_impr \
    --cost_right=$cost_right"
    echo $tr_cmd
    #GPU版本使用下面的代码执行Python文件
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd

    cost_right=100
    output_dir=${output_dir}_${rnn_num_layers}_${rnn_size}_${cost_right}
    tr_cmd="python3 run_lstm.py \
    --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --min_epochs=$min_epochs --max_epochs=$max_epochs --output_dir=$output_dir \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --halving_factor=$halving_factor --end_halving_impr=$end_halving_impr \
    --cost_right=$cost_right"
    echo $tr_cmd
    #GPU版本使用下面的代码执行Python文件
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd

    cost_right=1000
    output_dir=${output_dir}_${rnn_num_layers}_${rnn_size}_${cost_right}
    tr_cmd="python3 run_lstm.py \
    --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size --rnn_size=$rnn_size \
    --min_epochs=$min_epochs --max_epochs=$max_epochs --output_dir=$output_dir \
    --decode=$decode --learning_rate=$learning_rate --save_dir=$save_dir --data_dir=$data_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training \
    --model_type=$model_type --halving_factor=$halving_factor --end_halving_impr=$end_halving_impr \
    --cost_right=$cost_right"
    echo $tr_cmd
    #GPU版本使用下面的代码执行Python文件
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd
fi


