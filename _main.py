
import numpy as np
import ga

# 暂时用不到
# 5.正确性计算 用于后面绘制图表 表现训练过程中正确性的变化状态
# # pred 预测数值  label 真实数值
# def AccuarcyCompute(pred, label):
#     pred = pred.data.numpy()
#     label = label.data.numpy()
#     # np.argmax返回沿轴axis最大值的索引。 此处0代表列 表示取各列最大值的索引（从0开始）????????
#     test_np = (np.argmax(pred, 0) == label)
#     # 单精度浮点数，包括：1 个符号位，8 个指数位，23 个尾数位
#     test_np = np.float32(test_np)
#     return np.sum(test_np)


# 调用ga方法，开始计算并绘制图标 （绘制图表部分未完成）！！！！！！！！！！！！
# pop_size, num_gens, chance_of_mutation
ga.evolve(20, 10, 0.01)
