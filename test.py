    # 3.5反向传播求梯度（这里可以改一下 改成GA）
    #  x, t 含义待确定？？？？
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
