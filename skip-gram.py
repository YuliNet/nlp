import torch
import torch.nn as nn
import torch.nn.functional as F

# 语料库
corpus = ['he is a king',
          'she is a queen',
          'he is a man',
          'she is a woman',
          'warsaw is poland capital',
          'berlin is germany capital',
          'paris is france capital']

# 构建分词
corpus_list = [sentences.split() for sentences in corpus]
word2ix = {}
# 构造词典
for sentences in corpus:
    for words in sentences.split():
        if words not in word2ix:
            word2ix[words] = len(word2ix)
ix2word = {v:k for k, v in word2ix.items()}

# 构造训练对
WINDOWS = 2 # 记录窗口的大小
pairs = []  # 记录训练对
for sentences in corpus_list:
    for sentences_word_index in range(len(sentences)):
        center_word_ix = word2ix[sentences[sentences_word_index]]
        for win in range(-WINDOWS, WINDOWS + 1):
            context_word_index = sentences_word_index + win
            if 0 <= context_word_index < len(sentences) and context_word_index != sentences_word_index:
                context_word_ix = word2ix[sentences[context_word_index]]
                pairs.append((center_word_ix, context_word_ix))


class SkipGram(nn.Module):
    def __init__(self, emb_dim, vec_dim):
        super(SkipGram, self).__init__()
        # 初始化参数
        # 这两行的意思是要把词嵌入矩阵和隐藏层到输出层的矩阵变成可学习的矩阵
        self.embding_vector = nn.Parameter(torch.FloatTensor(emb_dim, vec_dim))
        self.W = nn.Parameter(torch.FloatTensor(vec_dim, emb_dim))
        # 用Xavier初始化方法，初始化参数, 因为网络涉及tanh或者sigmoid 而Relu 需要He方法进行初始化
        nn.init.xavier_normal_(self.embding_vector, gain=1)
        nn.init.xavier_normal_(self.embding_vector, gain=1)

    # 这里的x是需要训练的数据
    def forward(self, x):
        emb = torch.matmul(self.embding_vector, x)
        h = torch.matmul(self.W, emb)
        logSoftmax = F.log_softmax(h, dim=0)
        return logSoftmax

# 提前设置超参数
lr = 0.01
emb_dim = 5
epoch = 30

# 定义模型，优化器，损失函数
model = SkipGram(emb_dim, len(word2ix))
optim = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.NLLLoss()

# 获得ont-hot vector
def get_onehot_vector(ix):
    one_hot_vector = torch.zeros(len(word2ix)).float()
    one_hot_vector[ix] = 1.0
    return one_hot_vector

# 开始训练
for e in range(epoch):
    e_loss = 0.0
    optim.zero_grad()
    for i, (center_ix, context_ix) in enumerate(pairs):
        # 处理相应的数据结构
        x_onehot_vector = get_onehot_vector(context_ix)
        y = torch.Tensor([context_ix]).long()
        # forward pass
        y_hat = model(x_onehot_vector)
        closs = loss(y_hat.view(1, -1), y)
        # backprop pass
        closs.backward()
        e_loss += closs.data.item()
        # 梯度更新
        optim.step()
    print("The step %d loss is: %f" % (e, e_loss))
