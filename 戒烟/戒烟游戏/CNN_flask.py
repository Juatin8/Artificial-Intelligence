from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import YXJ


app = Flask(__name__)         # 创建一个Flask实例
app.config['DEBUG'] = True     # 开启调试模式

### -------------------------------------------定义模型结构-------------------------------------------------
out_channels =64
kernel_size = [3,2]
padding=1
stride=1
layers_num = 6
width =100
class ParallelConvModel(nn.Module):   #nn.Module是PyTorch中用于构建神经网络的基类
    ## 定义神经网络的各个层的情况
    def __init__(self):  
        super(ParallelConvModel, self).__init__()  #这是一个构造函数，被调用后可以实例化你自定义的这个神经网络   super()函数来调用基类的构造函数或方法  self相当于C#中的this
        self.conv_layers = nn.ModuleList()
        for i in range(layers_num):
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size[0], padding=padding,stride=stride),     
                #in_channels=1 输入的特征数量 #out_channels=64 可以理解成要提取多少个特征    #kernel_size 卷积核的大小  #stride默认为1
                # 1D卷积层需要的是形状为(batch_size,输入通道, 序列长度)
                # 每个卷积层的输出形状为 (batch_size, out_channels, sequence_length)
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=kernel_size[1]), # 每次池化操作后，序列长度会减半
                nn.Flatten() # 除了 batch_size 会被保留外，其余的维度压平到一个维度上 （batch_size(32), out_features*sequence_length(64*20/2)）
            )
            self.conv_layers.append(conv_layer)
        outputshape = int(YXJ.calc_conv_outputshape(width, kernel_size[0], stride, padding)*out_channels*0.5*layers_num) # 计算卷积层的输出数量
        self.dense_layer = nn.Sequential(
            nn.Linear(outputshape, 30),  # 输入数量，输出数量）
            nn.Linear(30, 1),   
            nn.Sigmoid()  # 二分类问题，最终输出的是0-1之间的值
        )

    ## 定义向前传播过程
    def forward(self, x):
        conv_outputs = []  # 用于保存所有卷积层的输出
        for i in range(6):
            feature_i = x[:, i:i+1, :]                     # 沿着feature维度分割输入张量，选择第i个feature  分割后的形状(batch_size, 1, sequence_length)
            conv_output_i = self.conv_layers[i](feature_i) # 将第i个feature通过第i层卷积层
            conv_outputs.append(conv_output_i)             # 将第i层卷积层的输出添加到卷积层输出列表
        conv_output = torch.cat(conv_outputs, dim=1)       # 沿着feature维度将所有卷积层输出拼接起来
        dense_output = self.dense_layer(conv_output)       # 将拼接的卷积层输出通过全连接层
        return dense_output  
    

### --------------------------------------------------推理/预测-------------------------------------------------------------
model_path = 'model.pth'
new_model = ParallelConvModel() #实例化模型

### ---------------------------------------------------定义路由 -------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def make_prediction():
    input = request.get_json('input')                  # 获取输入
    input = torch.tensor([input]).float()              # 将输入转换为 PyTorch 张量,并且是浮点型
    output = YXJ.predict(new_model, model_path, input) # 调用预测函数
    output = output.tolist()                           # 将 PyTorch 张量转换为 Python 列表
    return jsonify({'result': output})                 # 将计算结果转换为 JSON 格式并返回

### ------------------------------------------------- 启动应用程序 ----------------------------------------------------------
if __name__ == '__main__':
    app.run('0.0.0.0',port=4399) #端口号在矩池云那里租赁机器的时候自定义，0.0.0.0是那边要求这样写才能访问