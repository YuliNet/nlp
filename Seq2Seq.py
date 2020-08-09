import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import spacy
import numpy as np
import torch.utils.data as Data
import time
import math


# 加载英语和德语的语言模型
en = spacy.load("en_core_web_sm")
de = spacy.load("de_core_news_sm")


# 定义Encoder部分
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, batch_size, dropout, n_layers):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim

    def forward(self, src):
        emb = self.dropout(self.emb(src))
        outputs, (hidden, cell) = self.lstm(emb)
        return hidden, cell


# 定义 Decoder部分
class Decoder(nn.Module):
    def __init__(self, emb_dim, out_dim, input_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.emb = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hid_dim, out_dim)

    # 这里要加载新一次的input(其实就是EOS)
    def forward(self, input_src, hidden, cell):
        input_src = input_src.unsqueeze(0)
        embs = self.dropout(self.emb(input_src))
        outputs, (hidden, cell) = self.lstm(embs, (hidden, cell))
        predictions = self.out(outputs.squeeze(0))
        return predictions, hidden, cell


# 定义Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, src, trg):
        max_len = trg.shape[0]
        batch_size = trg.shape[1]
        inputs = trg[0, :]
        trg_vocab_size = self.decoder.out_dim
        hidden, cell = self.encoder(src)
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        for i in range(1, max_len):
            output, hidden, cell = self.decoder(inputs, hidden, cell)
            outputs[i] = output
            inputs = trg[i]
        return outputs



# 构建词表
def build_vocab(SRC):
    word2ix = {}
    for sentences in SRC:
        for words in sentences:
            if words not in word2ix:
                word2ix[words] = len(word2ix)
    word2ix[" "] = len(word2ix)
    ix2word = {v: k for k, v in word2ix.items()}
    return word2ix, ix2word


def build_onehot(words, lens):
    one_hot_return = []
    for i in range(len(words)):
        tmp = []
        for j in range(len(words[0])):
            one_hot = [0 for i in range(lens)]
            one_hot[words[i][j]] = 1
            tmp.append(one_hot)
        one_hot_return.append(tmp)
    one_hot_return = np.array(one_hot_return)
    return torch.FloatTensor(one_hot_return)


def predealData(datapath):
    doc = open(datapath)
    document_src = []
    while True:
        line = doc.readline()
        if not line:
            break
        else:
            document_src.append(line)
    max_len = 0
    SRC = []
    for tokens in en.pipe(document_src):
        SRC.append(list(tokens))
    for sen in SRC:
        max_len = max(max_len, len(sen))

    # 构建词表z
    word2ix, ix2word = build_vocab(SRC)
    for sents in SRC:
        if len(sents) < max_len:
            for i in range(max_len-len(sents)):
                sents.append(" ")
    for i in range(len(SRC)):
        for j in range(len(SRC[0])):
            SRC[i][j] = word2ix[SRC[i][j]]
    '''
    final_src = []
    for sents in SRC:
        tmp = []
        for words in sents:
            h = build_onehot(words, len(word2ix))
            tmp.append(h)
        final_src.append(tmp)
    '''
    return np.array(SRC), word2ix, ix2word


# 初始化模型参数
def init_param(m):
    for param in m.parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


# 定义消耗时间函数
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



# 定义各种参数
SRC, eword2ix, eix2word = predealData("./multi30k/train.en")
STG, dword2ix, dix2word = predealData("./multi30k/train.de")
INPUT_DIM = len(eword2ix)
OUTPUT_DIM = len(dword2ix)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 20
CLIP = 1

# 处理数据
SRC = torch.LongTensor(torch.from_numpy(SRC))
STG = torch.LongTensor(torch.from_numpy(STG))
dataset = Data.TensorDataset(SRC, STG)
loader = Data.DataLoader(dataset, 20, True)
'''
for indices in range(1):
    for batch_x, batch_y in loader:
        print(batch_x.shape, batch_y.shape)

'''
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, BATCH_SIZE, ENC_DROPOUT, N_LAYERS)
dec = Decoder(DEC_EMB_DIM, OUTPUT_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

# 加载seq2seq 模型
model = Seq2Seq(enc, dec)
model.apply(init_param)
optimizor = optim.Adam(model.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss(ignore_index=len(dword2ix)-1)


# 定义训练函数 clip是为了防止梯度爆炸
def train(model, lossf, optims, srcloader, clip):
    # 启动dropout和batchnormalization
    model.train()
    epoches = 8
    for indices in range(epoches):
        batch_loss = 0
        start_time = time.time()
        for batch_x, batch_y in srcloader:
            optims.zero_grad()
            output = model(batch_x.view(23, 20), batch_y.view(26, 20))
            new_batch_y = batch_y.view(-1)
            new_output = output.view(-1, output.shape[-1])
            loss = lossf(new_output, new_batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optims.step()
            batch_loss += loss.item()
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        batch_loss = batch_loss / len(srcloader)
        print(f'Epoch: {indices + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {batch_loss:.3f} | Train PPL: {math.exp(batch_loss):7.3f}')


train(model, loss, optimizor, loader, CLIP)
