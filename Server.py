# -*- coding: utf-8 -*-

# Import the necessary modules
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
from keras.utils import np_utils
# from tensorflow import plot_model
from keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Model_init as MI

def load_data(filepath, input_shape=20):
    df = pd.read_csv(filepath)

    # 标签及词汇表
    labels, vocabulary = list(df['label'].unique()), list(df['evaluation'].unique())

    # 构造字符级别的特征
    string = ''
    for word in vocabulary:
        string += word

    vocabulary = set(string)

    # 字典列表
    word_dictionary = {word: i+1 for i, word in enumerate(vocabulary)}
    with open('word_dict.pk', 'wb') as f:
        pickle.dump(word_dictionary, f)
    inverse_word_dictionary = {i+1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}
    with open('label_dict.pk', 'wb') as f:
        pickle.dump(label_dictionary, f)
    output_dictionary = {i: labels for i, labels in enumerate(labels)}

    vocab_size = len(word_dictionary.keys()) # 词汇表大小
    label_size = len(label_dictionary.keys()) # 标签类别数量

    # 序列填充，按input_shape填充，长度不足的按0补充
    x = [[word_dictionary[word] for word in sent] for sent in df['evaluation']]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[sent]] for sent in df['label']]
    y = [np_utils.to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(_[0]) for _ in y])

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary


# 创建深度学习模型， Embedding + LSTM + Softmax.
def create_LSTM(n_units, input_shape, output_dim, filepath):
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath)
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=output_dim,
                        input_length=input_shape, mask_zero=True))
    model.add(LSTM(n_units, input_shape=(x.shape[0], x.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(label_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # plot_model(model, to_file='./model_lstm.png', show_shapes=True)
    model.summary()

    return model






if __name__ == '__main__':
    filepath = './online_shopping_cloth2.csv'
    input_shape = 57

    # 导入字典
    with open('word_dict.pk', 'rb') as f:
        word_dictionary = pickle.load(f)
    with open('label_dict.pk', 'rb') as f:
        output_dictionary = pickle.load(f)

    # 得到初始化模型
    MI.model_init()
    lstm_model = load_model('./cloth_model_0.h5')
    vars_init = lstm_model.trainable_variables


    # 载入数据分配
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath, input_shape)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=42)

    vars_list = []

    # 模型输入参数，需要自己根据需要调整
    n_units = 100
    batch_size = 32
    epochs = 1
    output_dim = 20

    l = int(len(train_x)/4)
    for i in range(4):
        lstm_model0 = create_LSTM(n_units, input_shape, output_dim, filepath)
        lstm_model0.set_weights(vars_init)
        train_xx = train_x[i * l:(i + 1) * l]
        train_yy = train_y[i * l:(i + 1) * l]
        lstm_model0.fit(train_xx, train_yy)
        vars_list.append(lstm_model0.trainable_variables)

        # 保存新的模型
        lstm_model0.save('./cloth_model_'+str(i+1)+'.h5')

    global_vars = []

    for i in range(len(vars_list[0])):
        local_vars = vars_list[0][i]
        for j in range(len(vars_list)-1):
            local_vars = local_vars + vars_list[j+1][i]

        global_vars.append(local_vars / 4)

    # 测试初始模型的准确率
    N = test_x.shape[0]  # 测试的条数
    predict = []
    label = []
    for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
        sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model.predict(test_x[start:end])
        label_predict = output_dictionary[np.argmax(y_predict[0])]
        label_true = output_dictionary[np.argmax(test_y[start:end])]
        # print(''.join(sentence), label_true, label_predict) # 输出预测结果
        predict.append(label_predict)
        label.append(label_true)

    acc = accuracy_score(predict, label)  # 预测准确率
    print('初始模型在测试集上的准确率为: %s.' % acc)


    # 测试新的模型的准确率
    lstm_model_new = create_LSTM(n_units, input_shape, output_dim, filepath)
    lstm_model_new.set_weights(global_vars)

    N = test_x.shape[0]  # 测试的条数
    predict = []
    label = []
    for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
        sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model_new.predict(test_x[start:end])
        label_predict = output_dictionary[np.argmax(y_predict[0])]
        label_true = output_dictionary[np.argmax(test_y[start:end])]
        # print(''.join(sentence), label_true, label_predict) # 输出预测结果
        predict.append(label_predict)
        label.append(label_true)

    acc = accuracy_score(predict, label)  # 预测准确率
    print('联邦模型在测试集上的准确率为: %s.' % acc)





