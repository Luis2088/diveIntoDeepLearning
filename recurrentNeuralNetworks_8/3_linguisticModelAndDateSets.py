import collections
import random

import numpy as np
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
# 遍历tokens
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

# 词频图
freqs = [freq for token, freq in vocab.token_freqs]
#  x轴为freqs索引，y轴为对应的频率。 取对数，数值越小，变化越大
d2l.plot(freqs, xlabel='token :x', ylabel='frequency: n(x)', xscale='log', yscale='log')
d2l.plt.show()

# 二元语法频率
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]  # 错开一个位置拼接，每一项为连续两个单词的元组。
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

# 三元语法频率
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

# 对比三种模型的词元频率
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='idx', ylabel='frequency',
         xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
d2l.plt.show()


# 读取长序列数据
# 在随机采样中，每个样本都是在原始的长序列上任意捕获的子序列。
# 对于语言建模，目标是基于到目前为止看到的词元来预测下一个词元， 因此标签是移位了一个词元的原始序列。
# 使用随机抽样生成一个小批量子序列
# num_steps:时间步数 (可以理解为马尔卡夫假设的tau),  num_subseqs子序列的长度
def seq_data_iter_random(corpus, batch_size, num_steps):
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps - 1
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps

    # 长度为num_subseqs的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


# 用0到34的序列进行测试
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X:', X, '\nY:', Y)


# 顺序分区
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = np.array(corpus[offset:offset + num_tokens])
    Ys = np.array(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random()
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential()
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


