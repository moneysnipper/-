import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from logging import log

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

def load_saved_model(model_path):
    """
    从 HDF5 文件加载 Keras 模型。

    参数:
        model_path (str): 保存的模型文件的路径（包含文件扩展名）。

    返回:
        loaded_model: 加载的 Keras 模型实例。
    """
    # 加载并返回模型
    loaded_model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return loaded_model


# 设置页面
st.set_page_config(page_title="Data Exploration and Prediction Interface", layout="wide")

# 侧边栏选择上传文件
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")



# if uploaded_file is not None:
#     # 读取数据
#     df = pd.read_csv(uploaded_file)
#
#     # 显示数据框架
#     if st.sidebar.checkbox("Show dataframe"):
#         st.write(df)
#
#     # 分析数据 （这里放上你的数据分析代码）
#
#     # 已保存的 LSTM 模型加载
#     model = None
#     if st.sidebar.button("Load LSTM model"):
#         model = load_saved_model(model_path="my_lstm_model.h5")
#         st.sidebar.text(f"Model loaded successfully! \t {model}")
#
#     if st.sidebar.button("Predict"):
#         st.sidebar.text(f"Model loaded successfully! \t {model}")
#
#     # # 预测
#     # if model and st.sidebar.button("Predict"):
#     #     log.info(f"{model} Predicting...")
#     #     try:
#     #         # LSTM模型的输入数据预处理部分（实际情况可能有所不同）
#     #         # 此处我们假设数据已经被预处理并且整齐
#     #         # 数据标准化
#     #         log.info(f"read for predicting...")
#     #         scaler = MinMaxScaler(feature_range=(0, 1))
#     #         df1_scaled = scaler.fit_transform(df)
#     #         print(df1_scaled[0])
#     #         df1_X, df1_Y = createXY(df1_scaled, 60)
#     #         pred1 = model.predict(df1_X)
#     #         prediction1_copies_array = np.repeat(pred1, 7, axis=-1)
#     #         pred1_actual = scaler.inverse_transform(np.reshape(prediction1_copies_array, (len(pred1), 7)))[:, 0]
#     #         # 预测结果可视化
#     #         st.write("预测结果:")
#     #         for i, pred in enumerate(pred1_actual):
#     #             st.write(f"预测结果 {i + 1}: {pred}")
#     #     except Exception as e:
#     #         st.sidebar.write("An error occurred during prediction.")
#     #         st.sidebar.write(e)
#
# # 添加说明
# st.write("""
# # Data Exploration and Prediction Interface
#
# 上传CSV文件，查看数据，并使用预训练的LSTM模型进行预测。
# """)



if __name__ == '__main__':

    model = None
    st.markdown("""
    ## 使用说明

    1. 通过侧边栏上传CSV文件。
    2. 查看并分析数据，如果需要的话。
    3. 加载预训练的LSTM模型。
    4. 点击预测按钮以生成预测。
    """)
    # 读取CSV文件
    if 'uploaded_file' in st.session_state:
        uploaded_file = st.session_state['uploaded_file']
    else:
        uploaded_file = st.sidebar.file_uploader("请选择CSV文件", type=["csv"])
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file
            # 读取数据
            df = pd.read_csv(uploaded_file, index_col=[0])
            st.session_state['df'] = df

    # 如果已上传文件，则显示可用列并在主页面绘制走势图
    if 'df' in st.session_state:
        df = st.session_state['df']
        # 显示可选项
        if 'available_columns' in st.session_state:
            available_columns = st.session_state['available_columns']
        else:
            available_columns = df.columns.tolist()
            st.session_state['available_columns'] = available_columns

        # 选择列
        selected_column = st.sidebar.selectbox("选择要查看的列", available_columns)
        st.session_state['selected_column'] = selected_column
        # 绘制走势图
        if st.session_state.get('selected_column'):
            st.line_chart(df[st.session_state['selected_column']])
        else:
            st.warning("请选择要查看的列")




        # 检查是否已经加载了模型，如果没有，则初始化状态
        if 'model' not in st.session_state:
            st.session_state['model'] = None

        # 已保存的 LSTM 模型加载
        if st.sidebar.button("Load LSTM model"):
            st.session_state['model'] = load_saved_model(model_path="my_lstm_model.h5")
            st.sidebar.text(f"Model loaded successfully! \t {st.session_state['model']}")

        # 预测
        if st.session_state['model'] and st.sidebar.button("Predict"):

            # log.info(f"{model} Predicting...")
            #获取session_state中保存的模型
            model = st.session_state['model']
            try:
                # LSTM模型的输入数据预处理部分（实际情况可能有所不同）
                # 此处我们假设数据已经被预处理并且整齐
                # 数据标准化
                #log.info(f"read for predicting...")
                scaler = MinMaxScaler(feature_range=(0, 1))
                df1_scaled = scaler.fit_transform(df)
                #print(df1_scaled[0])
                df1_X, df1_Y = createXY(df1_scaled, 60)
                pred1 = model.predict(df1_X)
                prediction1_copies_array = np.repeat(pred1, 7, axis=-1)
                pred1_actual = scaler.inverse_transform(np.reshape(prediction1_copies_array, (len(pred1), 7)))[:, 0]
                print(f'行数: {pred1_actual.shape[0]}')
                np.savetxt("./B18_2107fault.xls预测结果.csv",
                           pred1_actual, delimiter=',')

                # # 预测结果可视化
                # st.write("预测结果:")
                # for i, pred in enumerate(pred1_actual):
                #     st.write(f"预测结果 {i + 1}: {pred}")

                # 设置Matplotlib字体以支持中文显示
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种常见的中文字体
                plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

                start_index = 60
                end_index = start_index + len(pred1_actual)

                # 获取对齐后的实际值
                actual_aligned = df["FN-FV"][start_index:end_index].reset_index(drop=True)
                # 计算残差
                residuals = actual_aligned - pred1_actual
                # 创建新的matplotlib图和轴
                fig, ax = plt.subplots()
                # 绘制预测结果，这里假设pred1_actual是包含预测结果的np.array 或 list
                ax.plot(np.arange(start_index, end_index),pred1_actual, color='tab:red', label='Predicted')
                ax.plot(np.arange(start_index, end_index),df["FN-FV"][start_index:end_index], color='tab:blue', label='Actual')

                # 绘制残差
                ax.plot(residuals, color='tab:green', label='Residual (Actual - Predicted)')
                # 可以设置更多的图形属性，如标题、轴标签等
                ax.set_title("Predicted vs Actual Values")
                ax.set_xlabel("时间步")
                ax.set_ylabel("温度")

                # 添加图例，位置为右上方
                ax.legend(loc='upper right')
                # 使用st.pyplot()展示绘制的图
                st.pyplot(fig)
            except Exception as e:
                st.sidebar.write("An error occurred during prediction.")
                st.sidebar.write(e)

    # 添加说明
    st.write("""
    # Data Exploration and Prediction Interface

    
    """)