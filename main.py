import math
import torch
import torchvision
import torch.utils.data
import torch.nn.functional
import numpy as np
from tensorboardX import SummaryWriter

# 1. 下载MNIST训练集和测试集，
train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 2. 定义初始参数
learning_rate = 0.01  # 学习率
Epoch = 5  # 训练轮次数
Batch_size = 64 # batch_size设置了每批装载的数据图片为64个
# 数据装载,装载时随机乱序
train_dataset = torch.utils.data.DataLoader(train_set, batch_size=Batch_size, shuffle=True)
test_dataset = torch.utils.data.DataLoader(test_set, batch_size=Batch_size, shuffle=True)
device = torch.device('cuda:0')
writer = SummaryWriter('log')

# 3. 定义MLP类
class MLP():
    # 3.1. 构建网络模型
    input_size = 784  # 因为图片都是28*28的1维图片，28*28=784
    hidden1_size = 390
    hidden2_size = 180
    hidden3_size = 90
    output_size = 10
    weight = []  # weight
    bias = []  # bias
    temp_in = []  # 作为层的输入
    temp_out = []  # 作为层的输出（例如output层在接收到层的输入后经过softmax（）处理得到的内容）

    d_w = []  # 经过后向传播后的weight
    d_b = []  # 经过后向传播后的bias

    # 给定一个self参数,定义sigmoid函数，s(t) = 1/1+e^-t
    # 是神经网络中的激活函数，其作用就是引入非线性
    def sig(self, x):
        return 1/(1 + torch.exp(-x))
    # 也是激活函数，将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    def ReLU(self, x):
        y = (torch.clamp(x, min=0)).float()
        return y
    # 反向传播中用到的激活函数 得出0 或 1
    def d_ReLU(self, x):
        y = (x > 0).float()
        return y

    # 3.2. 初始化参数（初始化weight & bias） randn生成随机数字的tensor，这些随机数字满足标准正态分布（0，1）
    # 通过除的方式，将权重初始化为比较小的值。 或者直接乘以0.01 避免出现梯度爆炸问题
    def init_parameters(self):
        self.weight.append(
            torch.randn(self.hidden1_size, self.input_size) /
            math.sqrt(28 * 28))
        self.weight.append(
            torch.randn(self.hidden2_size, self.hidden1_size) /
            math.sqrt(self.hidden1_size * self.hidden1_size))
        self.weight.append(
            torch.randn(self.hidden3_size, self.hidden2_size) /
            math.sqrt(self.hidden2_size * self.hidden2_size))
        self.weight.append(
            torch.randn(self.output_size, self.hidden3_size) /
            math.sqrt(self.hidden3_size * self.hidden3_size))
        self.bias.append(torch.randn(self.hidden1_size, 1) / math.sqrt(28 * 28))
        self.bias.append(torch.randn(self.hidden2_size, 1) / math.sqrt(self.hidden1_size * self.hidden1_size))
        self.bias.append(torch.randn(self.hidden3_size, 1) / math.sqrt(self.hidden2_size * self.hidden2_size))
        self.bias.append(torch.randn(self.output_size, 1) / math.sqrt(self.hidden3_size * self.hidden3_size))

    # 3.3.前向传播（从前往后走）
    def forward_pro(self, x):
        # 首先清空各层的输入和输出
        self.temp_in.clear()
        self.temp_out.clear()
        # 计算的是本层的值和下层的输入值
        self.temp_in.append((self.weight[0].mm(x) + self.bias[0]))
        self.temp_out.append(torch.relu(self.temp_in[0]))

        self.temp_in.append(self.weight[1].mm(self.temp_out[0]) + self.bias[1])
        self.temp_out.append(torch.relu(self.temp_in[1]))

        self.temp_in.append(self.weight[2].mm(self.temp_out[1]) + self.bias[2])
        self.temp_out.append(torch.relu(self.temp_in[2]))

        # output层要进行softmax()函数操作 它可以将任一实数xs转换成0到1之间的一个概率P(xs)
        self.temp_in.append(self.weight[3].mm(self.temp_out[2]) + self.bias[3])
        # exp(x)之和不是所有的exp(x)之和，是单个样本的exp(x)之和
        self.temp_out.append(torch.nn.functional.softmax(self.temp_in[3], dim=0))

        return self.temp_out[3]

    # 3.4 定义损失函数 计算损失 _class应该可以改？？？？log e
    def loss_cal(self, y_hat, label_value):
        # print(y_hat)
        return -1 * torch.log(y_hat[label_value])

    # 3.5反向传播求梯度（这里可以改一下 改成GA）
    # x, t 含义待确定？？？？
    def backward_pro(self, x, t):
        # 反向传播后的weight和bias
        self.d_w.clear()
        self.d_b.clear()

        # delta 是 loss 对 z 的导数
        # 输出层，delta求比较特殊
        delta = self.temp_out[3] - t
        self.d_w.append(delta.mm(self.temp_out[2].T))
        self.d_b.append(delta)
        tmp_db = torch.zeros(self.output_size, 1)
        for i in range(self.output_size):
            tmp_db[i] = delta[i].sum()
        self.d_b.append(tmp_db)  # 注意b的维数，批处理每个样本相加

        # 第三隐藏层
        # delta = ((self.W[3].T).mm(delta)) * (self.a[2] - self.a[2] * self.a[2])
        delta = ((self.weight[3].T).mm(delta)) * self.d_ReLU(self.temp_in[2])
        self.d_w.append(delta.mm(self.temp_out[1].T))
        tmp_db = torch.zeros(self.hidden3_size, 1)
        for i in range(self.hidden3_size):
            tmp_db[i] = delta[i].sum()
        self.d_b.append(tmp_db)  # 注意b的维数，批处理每个样本相加

        # 第二隐藏层
        # delta = ((self.W[2].T).mm(delta)) * (self.a[1] - self.a[1] * self.a[1])
        delta = ((self.weight[2].T).mm(delta)) * self.d_ReLU(self.temp_in[1])
        self.d_w.append(delta.mm(self.temp_out[0].T))
        tmp_db = torch.zeros(self.hidden2_size, 1)
        for i in range(self.hidden2_size):
            tmp_db[i] = delta[i].sum()
        self.d_b.append(tmp_db)  # 注意b的维数，批处理每个样本相加

        # 第一隐藏层
        # delta = ((self.W[1].T).mm(delta)) * (self.a[0] - self.a[0] * self.a[0])
        delta = ((self.weight[1].T).mm(delta)) * self.d_ReLU(self.temp_in[0])
        self.d_w.append(delta.mm((x.T)))
        tmp_db = torch.zeros(self.hidden1_size, 1)
        for i in range(self.hidden1_size):
            tmp_db[i] = delta[i].sum()
        self.d_b.append(tmp_db)  # 注意b的维数，批处理每个样本相加

        self.d_w.reverse()
        self.d_b.reverse()

    # 3.6 权重更新
    def optimize(self):
        for i in range(4):
            self.weight[i] = self.weight[i] - learning_rate * self.d_w[i]
            self.bias[i] = self.bias[i] - learning_rate * self.d_b[i]

# 3.7正确性计算 用于后面绘制图表 表现训练过程中正确性的变化状态
# pred 预测数值  label 真实数值
def AccuarcyCompute(pred, label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    # np.argmax返回沿轴axis最大值的索引。 此处0代表列 表示取各列最大值的索引（从0开始）????????
    test_np = (np.argmax(pred, 0) == label)
    # 单精度浮点数，包括：1 个符号位，8 个指数位，23 个尾数位
    test_np = np.float32(test_np)
    return np.sum(test_np)

# MLP开始
model = MLP()
# 导入参数
model.init_parameters()
# 更改打印设置 np.inf表示正无穷大 此参数是指定tensor的数目超过多少时开始显示进行折叠。默认为1000。也就是说不折叠
torch.set_printoptions(threshold=np.inf)
# 记录当前训练batch的次数
n_iter = 0

# 在训练轮次数内循环
for x in range(Epoch):
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for i, data in enumerate(train_dataset):
        # 标签：所预测的东西实际是什么（可理解为结论），如分类问题中图片中是0或是1还是..9。
        (inputs, labels) = data
        # T意为转置矩阵 定义输入层
        inputs = inputs.view(-1, 28 * 28).T
        # torch.zeros 10行 * 列 t用来存储什么
        t = torch.zeros(10, min(Batch_size, labels.size()[0]))
        # 遍历各列 也就是各个标签
        for j in range(min(Batch_size, labels.size()[0])):
            t[labels[j]][j] = 1
        # 开始前向传播 得到一轮前向传播后的
        y_hat = model.forward_pro(inputs)
        # 初始化loss值为0
        loss = 0
        for j in range(labels.size()[0]):
            # 计算损失
            loss += model.loss_cal((y_hat.T)[j], labels[j])
        loss = loss / labels.size()[0]
        # 这个值+1 ，代表一次前向过程结束
        n_iter += 1
        # 进入反向传播过程
        inputs
        model.backward_pro(inputs, t)
        # 更新权重
        model.optimize()
        # 每100个batch绘制一个点
        if (n_iter % 100 == 0):
            print(x, end=" x\n")
            print("loss : %.5f" % loss, end=" ")
            print("train : %.3f" %
                  (AccuarcyCompute(y_hat, labels) / labels.size()[0]),
                  end=" ")

            writer.add_scalar(
                'Train/Accuracy',
                (AccuarcyCompute(y_hat, labels) / labels.size()[0]), n_iter)
            writer.add_scalar('Train/Loss', loss, n_iter)

            accuarcy_list = []
            tmp = 0
            for j, (inputs, labels) in enumerate(test_dataset):
                inputs = inputs.view(-1, 28 * 28).T
                outputs = model.forward_pro(inputs)
                tmp += labels.size()[0]
                accuarcy_list.append(AccuarcyCompute(outputs, labels))
                loss_test = 0
                for k in range(labels.size()[0]):
                    loss += model.loss_cal((outputs.T)[k], labels[k])
                loss = loss / labels.size()[0]
            writer.add_scalar('Test/Accuracy',
                              sum(accuarcy_list) / tmp, n_iter)
            writer.add_scalar('Test/Loss', loss, n_iter)
            print("test : ",
                  format(sum(accuarcy_list) / tmp, '.3f'),
                  end=" \n")















