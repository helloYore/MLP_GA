import torch.nn as nn

# 定义MLP类
class MLP(nn.Module):
    def __init__(self, neurons, activator_id, optimizer_id):
        super(MLP, self).__init__()
        self.neurons = neurons  # track number of neurons in each layer
        self.activator_id = activator_id  # track activation function
        self.optimizer_id = optimizer_id  # track optimizer id, can be one of the following: Adadelta, Adagrad, Adam, Adamax, ASGD, RMSprop, Rprop, SGD

        # 随机初始值，这里也为GA运算中突变和交叉作准备
        if (activator_id == 0):
            self.activator = nn.ELU()
        elif (activator_id == 1):
            self.activator = nn.Hardshrink()
        elif (activator_id == 2):
            self.activator = nn.LeakyReLU()
        elif (activator_id == 3):
            self.activator = nn.LogSigmoid()
        # 构建网络模型（层数固定），其中中间各层的神经元都是基因算法中随机生成的
        self.layers = nn.Sequential(
            nn.Linear(784, self.neurons[0]),
            self.activator,
            nn.Linear(self.neurons[0], self.neurons[1]),
            self.activator,
            nn.Linear(self.neurons[1], self.neurons[2]),
            self.activator,
            nn.Linear(self.neurons[2], self.neurons[3]),
            self.activator,
            nn.Linear(self.neurons[3], 10),
            self.activator)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        return self.layers(out)



