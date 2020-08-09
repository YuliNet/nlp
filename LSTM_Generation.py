import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 设置随机种子
torch.manual_seed(1)

# 加载训练好的 word_embding model
wordmodel = word2vec.Word2Vec.load("vectors.model")
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
sentences = [sens.split() for sens in sentences]
targets = []
for sens in sentences:
    tmp = []
    for words in sens:
        tmp.append(wordmodel[words])
    targets.append(tmp)



##################################################

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # 输入单词用一个维度为3的向量表示, 隐藏层的一个维度3，仅有一层的神经元，
# 记住就是神经元，这个时候神经层的详细结构还没确定，仅仅是说这个网络可以接受[seq_len,batch_size,3]的数据输入
print(lstm.all_weights)

inputs = [torch.randn(1, 3) for _ in range(5)]
# 构造一个由5个单单词组成的句子 构造出来的形状是 [5,1,3]也就是明确告诉网络结构我一个句子由5个单词组成，
# 每个单词由一个1X3的向量组成，就是这个样子[1,2,3]
# 同时确定了网络结构，每个批次只输入一个句子，其中第二维的batch_size很容易迷惑人
# 对整个这层来说，是一个批次输入多少个句子，具体但每个神经元，就是一次性喂给神经元多少个单词。
print('Inputs:',inputs)

# 初始化隐藏状态
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
print('Hidden:',hidden)
for i in inputs:
    # 将序列的元素逐个输入到LSTM，这里的View是把输入放到第三维，看起来有点古怪，
    # 回头看看上面的关于LSTM输入的描述，这是固定的格式，以后无论你什么形式的数据，
    # 都必须放到这个维度。就是在原Tensor的基础之上增加一个序列维和MiniBatch维，
    # 这里可能还会有迷惑，前面的1是什么意思啊，就是一次把这个输入处理完，
    # 在输入的过程中不会输出中间结果，这里注意输入的数据的形状一定要和LSTM定义的输入形状一致。
    # 经过每步操作,hidden 的值包含了隐藏状态的信息
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    print('out1:',out)
    print('hidden2:',hidden)
# 另外, 我们还可以一次对整个序列进行训练. LSTM 返回的第一个值表示所有时刻的隐状态值,
# 第二个值表示最近的隐状态值 (因此下面的 "out"的最后一个值和 "hidden" 的值是一样的).
# 之所以这样设计, 是为了通过 "out" 的值来获取所有的隐状态值, 而用 "hidden" 的值来
# 进行序列的反向传播运算, 具体方式就是将它作为参数传入后面的 LSTM 网络.

# 增加额外的第二个维度
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print('out2',out)
print('hidden3',hidden)
'''
运行输出：
out2 tensor([[[-0.0187,  0.1713, -0.2944]],

        [[-0.3521,  0.1026, -0.2971]],

        [[-0.3191,  0.0781, -0.1957]],

        [[-0.1634,  0.0941, -0.1637]],

        [[-0.3368,  0.0959, -0.0538]]])
hidden3 (tensor([[[-0.3368,  0.0959, -0.0538]]]), tensor([[[-0.9825,  0.4715, -0.0633]]]))
'''