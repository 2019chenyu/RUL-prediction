# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
import tensorflow as tf
from numpy.random import RandomState
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense,Flatten,CuDNNLSTM,Dropout,Activation,Bidirectional
from keras.layers import Conv1D, Convolution1D, MaxPooling1D
from keras.callbacks import ReduceLROnPlateau  #学习率自动变化
from keras.callbacks import EarlyStopping
import sys
sys.path.append("../")
from utils.tcn import TCN, compiled_tcn
#使用GPU
import keras.backend.tensorflow_backend as KTF 
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
#config.gpu_options.per_process_gpu_memory_fraction = 0.3

import time
import os
import argparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#matplotlib inline
plt.rcParams['figure.figsize'] = 16,6
plt.rcParams['xtick.color'] = 'w'  
plt.rcParams['ytick.color'] = 'w'  
mpl.style.use('ggplot')
font1 = {'weight' : 'normal','size': 23}
font2 = {'weight' : 'normal','size': 18}

#主要的超参数
parser = argparse.ArgumentParser(description='1dcnn+blstm+fc')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--sequence_length', default=110, type=int, help='sequence_length')
parser.add_argument('--epochs', default=70, type=int, help='epochs')
parser.add_argument('--batch_size', default=5, type=int, help='batch_size')
parser.add_argument('--lr', default=2e-2, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
parser.add_argument('--test', default='0', type=str, help='input model file name, then it will be tested')
parser.add_argument('--reduce_lr', default=10 , type=int, help='')
parser.add_argument('--early_stopping', default=30 , type=int, help='')
## tcn hyperparameters
parser.add_argument('--nb_filters', default=64 , type=int, help='')
parser.add_argument('--nb_stacks', default=1 , type=int, help='')
parser.add_argument('--dropout_rate', default=0 , type=float, help='')
parser.add_argument('--stacked_tcn', default=0, type=int, help='')
parser.add_argument('--optimizer', default='adam', type=str, help='')


args = parser.parse_args()
print(args)
# seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

t = time.localtime()
ft = "%Y%m%d%H%M%S"
datetime = time.strftime(ft, t)

# ## 数据导入和处理
train_df_path = r'../data/processed_data/train_df.csv'
test_df_path = r'../data/processed_data/test_df.csv'
data_master_thesis_path = r'../../data/data_master_thesis'
if not os.path.exists(data_master_thesis_path):
    os.makedirs(data_master_thesis_path)

train_df = pd.read_csv(train_df_path,index_col=0) #第一列作为index
test_df = pd.read_csv(test_df_path,index_col=0)

#print("train_df shape: {}".format(train_df.shape))
#print("test_df shape: {}".format(test_df.shape))


# ## 定义X_train, y_train, X_test, y_train


X_train = train_df.iloc[:,5:26]
y_train = train_df.iloc[:,-2]

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = test_df.iloc[:,5:26]
y_test = test_df.iloc[:,-1]#注意，test_df中RUL列是最后一列，
                           #但是，测试集的RUL不是test_df的RUL对应的列

X_test = np.array(X_test)
y_test = np.array(y_test)

#print("X_train.shape: {}, y_train.shape: {}, X_test.shape: {}, y_test.shape: {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))


# ## 采用时间窗分割的方式改变数据的维度


# 将数据格式变为(样本循环次数, 时间窗大小：30, 特征数)
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(
            range(0, num_elements - seq_length), range(seq_length,
                                                       num_elements)):
        yield data_array[start:stop, :]


# 选择特征列
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# seq_array为用上函数生成的数组，其形状为(15631, 30, 25)
seq_gen = (list(
    gen_sequence(train_df[train_df['id'] == id], args.sequence_length,
                 sequence_cols)) for id in train_df['id'].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)


# 对应数据格式生成标签
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


# 标签的形状为(15631, 1)
label_gen = [
    gen_labels(train_df[train_df['id'] == id], args.sequence_length, ['RUL'])
    for id in train_df['id'].unique()
]
label_array = np.concatenate(label_gen).astype(np.float32)

# 生成test数据的最后一个序列，形状为(93, 50, 25)，不足100是因为有些测试集小于50
seq_array_test_last = [
    test_df[test_df['id'] == id][sequence_cols].values[-args.sequence_length:]
    for id in test_df['id'].unique()
    if len(test_df[test_df['id'] == id]) >= args.sequence_length
]
seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

# 对应生成test的label，形状为(93, 1)
y_mask = [
    len(test_df[test_df['id'] == id]) >= args.sequence_length
    for id in test_df['id'].unique()
]
label_array_test_last = test_df.groupby('id')['RUL'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(
    label_array_test_last.shape[0], 1).astype(np.float32)

#nb_features = seq_array.shape[2]
# nb_features == 25
#nb_out = label_array.shape[1]
# nb_out ==1

# print("seq_array shape: {}".format(seq_array.shape))
# print("label_array shape: {}".format(label_array.shape))
# print("seq_array_test_last shape: {}".format(seq_array_test_last.shape))
# print("label_array_test_last shape: {}".format(label_array_test_last.shape))

X_train = seq_array
y_train = label_array
X_test = seq_array_test_last
y_test = label_array_test_last

print("X_train.shape: {}, y_train.shape: {}, X_test.shape: {}, y_test.shape: {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# 现在X_train, y_train, X_test, y_test已经准备好了
def get_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def model_train_visualization(path, logname, model):
    # model architechture
    name = logname.split('.')[0] + '.png'
    path = path
    path = os.path.join(path, name)
    keras.utils.plot_model(model, to_file=path)

    # tensorboard
    

def main():
    ####模型构建#################################
    i = Input(batch_shape=(None, X_train.shape[1], X_train.shape[2]))
    if args.stacked_tcn == 0:
        o = TCN(nb_filters=args.nb_filters,
                nb_stacks=args.nb_stacks,
                dropout_rate=args.dropout_rate,
                return_sequences=False)(i)  # The TCN layers are here.
    else:
        o = TCN(nb_filters=args.nb_filters,
                nb_stacks=args.nb_stacks,
                dropout_rate=args.dropout_rate,
                return_sequences=True)(i)
        if args.tcn_num >= 3:
            for j in range(args.tcn_num - 2):
                o = TCN(return_sequences=True, dropout_rate=args.dropout1)(o)  
        o = TCN(return_sequences=False)(o)
    o = Dense(1)(o)

    model = Model(inputs=[i], outputs=[o])
    if args.optimizer == 'sgd':
        sgd = keras.optimizers.SGD(lr=args.lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd)
    elif args.optimizer == 'adam':
        adam = keras.optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999,
                                     epsilon=None, decay=args.weight_decay, amsgrad=False)
    
        model.compile(loss='mse', optimizer=adam)

    # 模型训练
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=args.reduce_lr, verbose=1, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=args.early_stopping, verbose=1)
    
    ## tensorboard logging
    tb_callback = keras.callbacks.TensorBoard(log_dir = os.path.join(data_master_thesis_path, 'tensorboard'),
                                              histogram_freq=1,
                                              write_graph=True,
                                              write_images=True)
    best_model_name = 'best' + datetime + '.h5'
    best_model_path = os.path.join(os.path.join(data_master_thesis_path, 'model_saved'), best_model_name)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_loss',mode='auto' ,save_best_only='True')
    start_time = time.clock()
    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        #validation_split=0.3,
        validation_data = (X_test, y_test),
        callbacks=[reduce_lr, early_stopping, tb_callback, checkpoint],
        verbose=2,
        shuffle=True)
    end_time = time.clock()
    print("Training time: {:.4} minutes".format((end_time - start_time) / 60))

    # 模型评估
    model = keras.models.load_model(best_model_path)
    y_pred = model.predict(X_test)

    rmse = get_rmse(y_pred, y_test)

    # 将以best开头的模型名称改为以rmse开头
    model_name = str(int(rmse*100)) + datetime + '.h5'
    model_path = os.path.join(os.path.join(data_master_thesis_path, 'model_saved'), model_name)
    os.system("mv {} {}".format(best_model_path, model_path))

    # 画出预测和实际点的对比图
    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(y_test)), y_test, 'o-', color='black', label='labels')
    plt.plot(np.arange(len(y_pred)), y_pred, 'o-', color='red', label='predictions')
    plt.legend(prop=font2)
    plt.xlabel("index")
    plt.ylabel("rul")
    figname = str(int(rmse * 100)) + datetime + '_pred_and_label.png'
    figpath = os.path.join('../model_saved/images', figname)
    plt.savefig(figpath)

    # 保存预测值和实际值到文件
    preds_labels = np.concatenate((y_pred, y_test), axis=1)
    filename = str(int(rmse * 100)) + datetime + '_pred_and_label.txt'
    filepath = os.path.join('../model_saved/preds_labels', filename)
    np.savetxt(filepath, preds_labels)
    
    # 保存loss和val_loss以及其他信息到logs文件夹
    log = np.concatenate((np.array(history.history['loss'])[:, np.newaxis],
                           np.array(history.history['val_loss'])[:,np.newaxis]), axis=1)
    logpath = str(int(rmse*100)) + datetime + '.txt'
    logpath = os.path.join('../model_saved/logs', logpath)
    np.savetxt(logpath, log)
    with open(logpath, 'a', encoding='utf-8') as f:
        #f.write(model.get_config()) # TypeError: write() argument must be str, not dict
        #print(model.get_config(), file=f)
        f.write(str(args))
        model.summary(print_fn=lambda x: f.write(x + '\n')) # it is the easist way
                                                            # to log model.summary()

        plt.figure(figsize=(20, 10))
        plt.plot(history.history['loss'], 'o-', color='red', label='train')
        plt.plot(history.history['val_loss'], 'o-', color='blue', label='val')
        plt.legend(prop=font2)
        plt.xlabel('epoch', font1)
        plt.ylabel('loss', font1)
        figname = str(int(rmse * 100)) + datetime + '.png'
        figpath = os.path.join('../model_saved/images', figname)
        plt.savefig(figpath)
        #plt.show()
    # model architechture
    arch_name = str(int(rmse*100)) + datetime + '.png'
    arch_path = '../model_saved/model_architechture'
    path = os.path.join(arch_path, arch_name)
    try:
        keras.utils.plot_model(model, to_file=path)
    except Exception as e:
        print(e)
    

    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    print("train loss:")
    print(train_loss)
    print("val loss:")
    print(val_loss)
    print()
    print(model.summary())
    print()
    print("model_name: ", model_name)
    print("time_step: {}".format(args.sequence_length))
    print("set epoch_num: {}".format(args.epochs))
    print("batch_size: {}".format(args.batch_size))
    print("lr: {}".format(args.lr))
    print("nb_filters: ", args.nb_filters)
    print("nb_stacks: ", args.nb_stacks)
    print("dropout_rate: ", args.dropout_rate)
    print("*****最佳模型表现***************************")
    print("Epoch:", np.where(val_loss==min(val_loss)))
    print("Train rmse: {}".format(np.sqrt(train_loss[val_loss==min(val_loss)])))
    print("Test rmse: {}".format(np.sqrt(val_loss[val_loss==min(val_loss)])))


def test(model_name):
    model_path = os.path.join('../model_saved', model_name)
    model = keras.models.load_model(model_path)
    y_pred = model.predict(X_test)
    rmse = get_rmse(y_pred, y_test)
    print('rmse: {:.2f}'.format(rmse))

    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(y_test)), y_test, 'o-', color='black', label='label')
    plt.plot(np.arange(len(y_pred)), y_pred, 'o-', color='red', label='predict')
    plt.legend(prop=font2)
    figname = str(int(rmse * 100)) + datetime + '_pred_and_label.png'
    figpath = os.path.join('../model_saved/images', figname)
    plt.savefig(figpath)

    preds_labels = np.concatenate((y_pred, y_test), axis=1)
    filename = str(int(rmse * 100)) + datetime + '_pred_and_label.txt'
    filepath = os.path.join('../model_saved/preds_labels', filename)
    np.savetxt(filepath, preds_labels)

if __name__ == '__main__':
    if args.test == '0':
        main()
    else:
        test(args.test)





