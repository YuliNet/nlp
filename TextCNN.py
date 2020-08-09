from gensim.models import word2vec
import torch
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 首先加载训练好的 word embding ,并将此次语料库中的文字再次进行预训练
model = word2vec.Word2Vec.load("vectors.model")
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.
sentences = [items.split() for items in sentences]
'''
model.build_vocab(sentences, update=True)
model.train(sentences, total_examples=model.corpus_count, epochs=10)
model.save("vectors.model")
'''
# 转换训练语料数据

dtype = torch.FloatTensor
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 使用GPU进行训练


# 处理相应数据
def make_data(inputDataSets, labels, models):
    dataSets = []
    for sens in inputDataSets:
        tmp = []
        for words in sens:
            tmp.append(models[words])
        dataSets.append(tmp)
    targets = torch.LongTensor(labels)
    return torch.FloatTensor(dataSets), targets

input_batch, target_batch = make_data(sentences, labels, model)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, 3, True)


class TextCNN(torch.nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        # 初始化各种参数
        self.embding_sizes = 100
        self.batch_sizes = 3
        output_channels = 3
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_channels, (2, self.embding_sizes)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 1))
        )
        self.fc = torch.nn.Linear(output_channels, 2)

    def forward(self, inputDataSets):
        inputDataSets = torch.unsqueeze(inputDataSets, 1)
        conv = self.conv(inputDataSets)
        flatten = conv.view(self.batch_sizes, -1)
        output = self.fc(flatten)
        return output

cnnModels = TextCNN()
lossf = torch.nn.CrossEntropyLoss()
optims = optim.Adam(cnnModels.parameters(), lr=0.01)

epoches = 100
for indices in range(epoches):
    for batch_x, batch_y in loader:
        pred = cnnModels(batch_x)
        ls = lossf(pred, batch_y)
        print('Epoch:', '%04d' % (indices + 1), 'loss =', '{:.6f}'.format(ls))

        cnnModels.zero_grad()
        ls.backward()
        optims.step()




