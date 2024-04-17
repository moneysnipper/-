import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn import metrics
from keras.models import Sequential, model_from_json

# 读取数据
def pre_process():
  """
    预处理数据，将数据标准化
  """
  # 导入数据
  df = pd.read_csv("D:\\深度学习实验\\deep learning\\xgboost实验\\lstm\\train.csv", engine='python', encoding='UTF-8',
                   index_col=[0])

  # 划分训练集和测试集
  test_split = round(len(df) * 0.20)
  df_train = df[:-test_split]  # -test_split
  df_test = df[-test_split:]
  print(df_train.shape)
  print(df_test.shape)

  # 数据标准化
  scaler = MinMaxScaler(feature_range=(0, 1))
  df_train_scaled = scaler.fit_transform(df_train)
  df_test_scaled = scaler.transform(df_test)

  X_train, Y_train = createXY(df_train_scaled, 60)
  X_test, Y_test = createXY(df_test_scaled, 60)
  return df_train,df_test,scaler,X_train, Y_train,X_test, Y_test



# 数据窗口划分
def createXY(dataset,n_past):  #N_past是我们在预测下一个目标值时将在过去查看的步骤数。
  """
    将数据集dataset分为输入X和输出Y。n_past是我们在预测下一个目标值时将在过去查看的步骤数。
  """
  dataX = []
  dataY = []
  for i in range(n_past, len(dataset)):
          dataX.append(dataset[i - n_past:i, [0,3,4,5,6]])  #0:dataset.shape[1]表示列数
          dataY.append(dataset[i,0])
  return np.array(dataX),np.array(dataY)
 


# 构建模型
def build_model():
  """
   构建LSTM模型
  """
  grid_model = Sequential()
  grid_model.add(LSTM(50,return_sequences=True,input_shape=(60,5)))  # 添加LSTM层，设置输入形状
  grid_model.add(LSTM(50))  # 添加LSTM层
  grid_model.add(Dropout(0.4))  # 添加Dropout层，防止过拟合
  grid_model.add(Dense(1))  # 添加Dense层，设置输出形状为1
  grid_model.compile(loss='mse',optimizer='Adam')  # 编译模型
  return grid_model

def save_model(model):
    # saving model
    json_model = model.model.to_json()
    open('model_architecture.json', 'w').write(json_model)
    # saving weights
    model.model.save_weights('model_weights.h5', overwrite=True)

def load_model():
  # loading model
  model = model_from_json(open('./model_architecture.json').read())
  model.load_weights('./model_weights.h5')
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  return model

def forward_model(X,Y):
  pass

def train(X_train=None, Y_train=None, X_test=None, Y_test=None):
    grid_model = KerasRegressor(build_fn=build_model, verbose=1)  # 使用KerasRegressor包装器来包装模型
    batch_size = 256  # 批处理大小
    epochs = 5  # 迭代次数
    model = grid_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs)
    # 拟合模型并返回训练历史记录，model变量包含训练历史记录

    # 查看训练过程信息
    history_dict = model.history  # 获取训练的数据字典
    train_loss = history_dict['loss']  # 训练集损失
    val_loss = history_dict['val_loss']  # 验证集损失
    # 绘制训练损失和验证损失
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')  # 训练集损失
    plt.plot(range(epochs), val_loss, label='val_loss')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    return model

def predict(model, scaler=None, ):
    # 模型评价
    pred = model.predict()  # 预测值
    prediction_copies_array = np.repeat(pred, 7, axis=-1)  # 复制7次，axis=-1表示复制到最后一维
    pred_actual = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(pred), 7)))[:, 0]  # 逆缩放
    # 保存预测结果
    np.savetxt("./train预测结果.csv", pred_actual, delimiter=',')
    return pred_actual

def plot():
    # 模型评价指标
    print('MSE:', metrics.mean_squared_error(original, pred_actual))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(original, pred_actual)))
    print('MAE:', metrics.mean_absolute_error(original, pred_actual))
    print('R2-score:',metrics.r2_score(original, pred_actual))

    # 测试集结果可视化
    plt.figure()
    plt.plot(original, color='red', label='真实值')
    plt.plot(pred_actual, color='blue', label='预测值')
    plt.title('温度预测')
    plt.xlabel('样本数')
    plt.ylabel('温度')
    plt.legend()
    plt.show()


# # 实际故障时段预测
# #故障1
#
# df1 = pd.read_csv("D:\\深度学习实验\\deep learning\\xgboost实验\\lstm\\B18_2107fault.xls.csv",engine='python',index_col=[0])
# df1_scaled = scaler.fit_transform(df1)
# df1_X,df1_Y = createXY(df1_scaled,60)
# pred1 = model.predict(df1_X) #标准化的预测值
# prediction1_copies_array = np.repeat(pred1,7, axis=-1)
# pred1_actual = scaler.inverse_transform(np.reshape(prediction1_copies_array,(len(pred1),7)))[:,0]
# np.savetxt("D:\\深度学习实验\\deep learning\\xgboost实验\\lstm\\B18_2107fault.xls预测结果.csv",pred1_actual, delimiter=',')
#
# #故障2
# df2 = pd.read_csv("D:\\深度学习实验\deep learning\\xgboost实验\\lstm\\B18_2112fault.xls.csv",engine='python',index_col=[0])
# df2_scaled = scaler.fit_transform(df2)
# df2_X,df2_Y = createXY(df2_scaled,60)
# pred2 = model.predict(df2_X) #标准化的预测值
# prediction2_copies_array = np.repeat(pred2,7, axis=-1)
# pred2_actual = scaler.inverse_transform(np.reshape(prediction2_copies_array,(len(pred2),7)))[:,0]
# np.savetxt("D:\\深度学习实验\\deep learning\\xgboost实验\\lstm\\B18_2112fault.xls预测结果.csv",pred2_actual, delimiter=',')
#
# #故障3
# df3 = pd.read_csv("D:\\深度学习实验\\deep learning\\xgboost实验\\lstm\\B18_2203fault.xls.csv",engine='python',index_col=[0])
# df3_scaled = scaler.fit_transform(df3)
# df3_X,df3_Y = createXY(df3_scaled,60)
# pred3 = model.predict(df3_X) #标准化的预测值
# prediction3_copies_array = np.repeat(pred3,7, axis=-1)
# pred3_actual = scaler.inverse_transform(np.reshape(prediction3_copies_array,(len(pred3),7)))[:,0]
# np.savetxt("D:\\深度学习实验\\deep learning\\xgboost实验\\lstm\\B18_2203fault.xls预测结果.csv",pred3_actual, delimiter=',')

if __name__ == "__main__":
    pre_process()
    createXY()
    build_model()
    train()
    # train()
    # plot()
