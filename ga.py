import random
import torch
import newMLP
import torchvision
import torch.utils.data


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

# 3. 配置测试环境
if(torch.cuda.is_available()):
	device = torch.device('cuda:0')
else:
	device = torch.device('cpu')



# create a NN with random parameters
def create_MLP():
    # pick neurons in each layer
    neurons = list()
    # number of layers - 1
    for i in range(4):
        neurons.append(random.randint(80, 1000))
    # pick activation function
    actf_id = random.randint(0, 3)

    # pick random optimizer
    optim_id = random.randint(0, 3)

    # create neural network and return
    return newMLP.MLP(neurons, actf_id, optim_id)


# trains an individual model 训练集的表现
def train(model):

    # set up optimizer 根据不同的优化id进行不同的优化操作
    if (model.optimizer_id == 0):
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif (model.optimizer_id == 1):
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif (model.optimizer_id == 2):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif (model.optimizer_id == 3):
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

    # set up loss function 定义loss函数
    loss_func = newMLP.nn.CrossEntropyLoss()

    # training
    for epoch_id in range(Epoch):
        for batch_id, (inps, vals) in enumerate(train_dataset):
            # format inps and vals
            inps = inps.to(device)
            vals = vals.to(device)

            # forward pass
            outs = model(inps)
            loss = loss_func(outs, vals)

            # backward and optimize
            # 将模型的参数梯度初始化为0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# tests individual model performance 测试集的表现
def test_model_performance(model):

    model.eval()

    # testing
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_id, (inps, vals) in enumerate(test_dataset):
            # format inps and vals
            inps = inps.to(device)
            vals = vals.to(device)

            # forward pass
            outs = model(inps)
            pred = outs.data.max(1)[1]

            # increment accordingly
            total += 1
            if (pred[0].item() == vals[0].item()):
                correct += 1
    # 返回正确率
    return correct / total


# mutates one parameter of the model 基因突变运算
def mutate(model):
    key1 = random.random()
    # complete mutation 10% of the itme
    if (key1 <= 0.1):
        return create_MLP()
    # otherwise just mutate one parameter
    else:
        key2 = random.randint(0, 2)
        if (key2 == 0):
            # change number of neurons in a random layer
            layer_num = random.randint(0, 3)
            model.neurons[layer_num] = random.randint(80, 1000)
            return newMLP.MLP(model.neurons, model.activator_id, model.optimizer_id)
        elif (key2 == 1):
            # change activator
            return newMLP.MLP(model.neurons, random.randint(0, 3), model.optimizer_id)
        elif (key2 == 2):
            # change optimizer
            return newMLP.MLP(model.neurons, model.activator_id, random.randint(0, 3))

# creates child using parameters from two parents 基因交叉运算
def breed(m1, m2):
    # choose keys to determine which parent to take parameter from
    key1 = random.randint(0, 1)
    key2 = random.randint(0, 1)
    key3 = random.randint(0, 1)

    if (key1 == 0):
        neurons = m1.neurons
    else:
        neurons = m2.neurons
    if (key2 == 0):
        activator_id = m1.activator_id
    else:
        activator_id = m2.activator_id
    if (key3 == 0):
        optimizer_id = m1.optimizer_id
    else:
        optimizer_id = m2.optimizer_id

    return newMLP.MLP(neurons, activator_id, optimizer_id)


# creates initial population for training创建初始种群数
def initial_pop(pop_num):
    # store population in list
    pop = list()

    for i in range(pop_num):
        pop.append(create_MLP())

    return pop


# prevent double training 防止训练两次
def reset_pop(pop):
    # new population without any training
    fresh_pop = list()

    # create new NN for each member of population
    for mem in pop:
        fresh_pop.append(newMLP.MLP(mem.neurons, mem.activator_id, mem.optimizer_id))

    return fresh_pop


# actual evolution  实际的进化
def evolve(pop_size, num_gens, chance_of_mutation):
    # create initial population
    pop = initial_pop(pop_size)

    # evolution through generations
    for gen_id in range(num_gens):
        print("Generation {}".format(gen_id))
        # train each of the members of the generation
        for mem_id in range(pop_size):
            print("training member {} ({}, {}, {})".format(mem_id, pop[mem_id].neurons,
                                                               pop[mem_id].activator_id, pop[mem_id].optimizer_id))
            pop[mem_id] = train(pop[mem_id])

        # get performance from each of the members and remove those below averge
        # store performance score in list
        perf_scores = list()
        for mem_id in range(len(pop)):
            print("testing member {}".format(mem_id))
            perf_scores.append(test_model_performance(pop[mem_id]))

        # print progress of current population
        print("Performance Summary of Generation {}:".format(gen_id))
        for i in range(pop_size):
            print("Member {} achieved accuracy of {} with parameters ({}, {}, {})"
                  .format(i, perf_scores[i], pop[i].neurons, pop[i].activator_id, pop[i].optimizer_id))

        # don't do the following for the last generation
        if (gen_id != num_gens - 1):
            # find average and keep those above it
            average_perf = sum(perf_scores) / pop_size
            print("Average accuracy was {}".format(average_perf))

            new_pop = list()
            surv_inds = list()

            print("Surviving members:")
            for i in range(pop_size):
                if (perf_scores[i] >= average_perf):
                    new_pop.append(pop[i])
                    # track indices of surviving members for printing
                    surv_inds.append(i)
                    print("member {} survived".format(i))

            pop = new_pop.copy()

            # fill in remaining population members by breeding members of population
            while (len(pop) < pop_size):
                # pick two random parents
                m1_id = random.randint(0, len(new_pop) - 1)
                m2_id = random.randint(0, len(new_pop) - 1)

                # breed together if parents are not the same member
                if (m1_id != m2_id):
                    new_MLP = breed(new_pop[m1_id], new_pop[m2_id])

                    # chance to mutate new child
                    key = random.random()
                    if (key <= chance_of_mutation):
                        new_MLP = mutate(new_MLP)
                        print(
                            "bred together members {} and {}, child mutated".format(surv_inds[m1_id], surv_inds[m2_id]))
                    else:
                        print("bred together members {} and {}, child not mutated".format(surv_inds[m1_id],
                                                                                          surv_inds[m2_id]))
                    pop.append(new_MLP)

            # reset populations
            pop = reset_pop(pop)
            print()
        else:
            # save best performer after all generations
            best_perf_score = -1
            best_perf_id = -1

            for i in range(pop_size):
                if (perf_scores[i] > best_perf_score):
                    best_perf_score = perf_scores[i]
                    best_perf_id = i

                print("Best performer was member {} with an accuracy of {}".format(best_perf_id, best_perf_score))
                return pop[best_perf_id]
