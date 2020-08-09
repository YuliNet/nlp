from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 语料库
corpus_list = word2vec.Text8Corpus("/home/yuli/下载/text8.txt")
# train the models
model = word2vec.Word2Vec(corpus_list, min_count=1, size=100, workers=1)
model.save("vectors.model")