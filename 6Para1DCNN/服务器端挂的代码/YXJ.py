from torch.utils.data import random_split
import numpy as np
import torch

# ---------------------------------将数据集划分为训练集和测试集-----------------------------------------
def split_dataset(dataset, split_ratio=0.8):
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

### --------------------------------------- 归一化---------------------------------------------
def normalization(data):
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data

### --------------------------------------- upsampling---------------------------------------------
def upsampling(targetdata, thisdata):
    upsampled_data = np.interp(np.arange(len(targetdata)), np.arange(len(thisdata)), thisdata)
    return upsampled_data

def upsampling(targetdatalen, thisdata):
    upsampled_data = np.interp(np.arange(targetdatalen), np.arange(len(thisdata)), thisdata)
    return upsampled_data

### ------------------------------------等分分割数据生成多个样本---------------------------------------------
def Split2Samples(X_data, width,dim=0):
    num_samples = X_data.shape[0] // width    # 计算样本数量
    X_data = X_data[:num_samples * width]     # 在样本长度上截断以确保样本之间的连续性
    chunked_tensors = torch.chunk(X_data, chunks=num_samples, dim=dim)    # 按照第0维度分割，分割成num_samples份
    X_data = torch.stack(chunked_tensors, dim=1)   
    return X_data  


### ------------------------------------计算卷积操作后输出张量的形状---------------------------------------------
def calc_conv_outputshape(input_shape, kernel_size, stride, padding):
    N = input_shape   # 获取输入张量的尺寸
    K, S = kernel_size, stride   # 获取卷积核的尺寸和步长
    O = (N - K + 2*padding) // S + 1   # 计算输出张量的尺寸
    O= int(O)
    return O



### --------------------------------------------训练模型---------------------------------------------
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    losses = []  # 记录每个batch的损失
    for epoch in range(num_epochs):  # 遍历数据集多次
        epoch_loss = 0.0  # 记录每个epoch的损失
        for x_batch, y_batch in train_loader:  # 遍历训练集中的所有batch
            x_batch,y_batch = x_batch.clone().requires_grad_(True),y_batch.clone().detach() 
            # clone是将数据备份，requires_grad是要求计算梯度    标签detach是将数据从计算图中分离出来，不参与梯度计算

            optimizer.zero_grad()  # 梯度清零
            y_pred = model(x_batch)  # 前向传播
            loss = criterion(y_pred, y_batch)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度下降

            epoch_loss += loss.item()  # 累加损失
            losses.append(loss.item())  # 将当前的损失值添加到列表中
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")  # 打印损失
    return model,losses


### --------------------------------------------测试模型---------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def calculate_metrics(model, data_loader):
    with torch.no_grad():
        true_labels = []
        predicted_labels = []
        for inputs, labels in data_loader:
            # 将数据放入模型中进行前向传播并计算精度
            outputs = model(inputs)               # 前向传播
            predicted = (outputs > 0.5).float()   # 将输出值转换为0或1
            true_labels.extend(labels.cpu().numpy())        # 将标签转换为numpy数组并添加到列表中
            predicted_labels.extend(predicted.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        cm=confusion_matrix(true_labels, predicted_labels)
        
        return accuracy, precision, recall, f1,cm
    
###------------------------------------------- 推理 ----------------------------------------------------------
def predict(new_model,model_path, data):
# 加载保存的模型参数
    checkpoint = torch.load(model_path) 
    new_model.load_state_dict(checkpoint) 
    with torch.no_grad():
        predict = new_model(data)
    return predict

### --------------------------------------------绘制 confusion matrix---------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm(cm):
    # 可视化混淆矩阵
    sns.set(font_scale=1.4) # 设置字体大小
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues') # 设置参数
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion matrix')
    plt.show()

### -------------------------------------------------绘制loss函数---------------------------------------------
import matplotlib.pyplot as plt
def draw_loss(losses):
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()














'''
# 对upsampled_longitude、upsampled_altitude、upsampled_latitude进行滑动窗口操作
window_size = 10  # 窗口大小为10
stride = 5  # 步长为5

def Window(data,window_size,stride):
    Windowed_data = np.array([data[i:i+window_size] for i in range(0, len(data)-window_size+1, stride)])
    # 将二维数组压缩为一维数组
    Windowed_data = np.reshape(Windowed_data, [-1])
    return Windowed_data

### 所有要用到的维度的数据都窗口滑动一下
x = Window(acceleration_x,window_size,stride)
y = Window(acceleration_y,window_size,stride)
z = Window(acceleration_y,window_size,stride)
lo = Window(upsampled_longitude,window_size,stride)
a = Window(upsampled_altitude,window_size,stride)
la = Window(upsampled_latitude,window_size,stride)
rolling_isSmoke = Window(isSmoke,window_size,stride)

print(x.shape)



ang_velocity_x = df2["AngularVelocity-X"].to_numpy()
ang_velocity_y = df2["AngularVelocity-Y"].to_numpy()
ang_velocity_z = df2["AngularVelocity-Z"].to_numpy()
'''