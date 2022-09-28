import collections
import re

import numpy
import torch
from d2l import torch as d2l

# 1.读取数据集
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():
    # 将时间机器数据集加载到文本行的列表中
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


lines = read_time_machine()
print(f'# 文本总行数:{len(lines)}')


# print(numpy.array(lines).reshape(-1, 1)[:10])


# 词元化
# 将文本行拆分为单词或者字符
def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('wrong token')


tokens = tokenize(lines, token='word')


# for i in range(10):
#     print(tokens[i])


# 构建词表
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        # couter.items() == [('the', 2261), ('i', 1267), ('and', 1245), ('of', 1) ...]
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


# 统计词元频率
def count_corpus(tokens):
    # tokens  1 or 2 dim
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        # 先执行 for line in tokens 返回第一行 , 再执行 for token in line 对第一行遍历
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# test
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])


# for i in range(12):
#     print('文本', tokens[i])
#     print('索引', vocab[tokens[i]])


def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
