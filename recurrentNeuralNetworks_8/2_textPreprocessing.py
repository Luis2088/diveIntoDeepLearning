import collections
import re
from d2l import torch as d2l

# 1.读取数据集
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])


# 列表中的每个元素是一个文本序列,每个文本序列被拆分成一个词元列表，词元（token）是文本的基本单位。
def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


# tokens 还是二维的,第一个纬度是文本行，第二纬度是由词元组成的字符串列表
tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


# 词表
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        # counter为集合实例，每一个item是一个dict类型的键值对。此处k为
        counter = count_corpus(tokens)

        # return : [('the', 2260), ('i', 1266), ('and', 1245) ....
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 未知词元的索引为 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            # _token_freqs中的freq是从大到小的，如果小于minfreq就终止循环
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    # 此处的tokens 可以是一个子序列，获取对应的索引。子序列的类型可以是列表或元组,可以无限递归（套娃）,具体问题具体分析
    # vocab[] 快捷调用
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    # 索引转对应的词元
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


# 统计词元的频率
def count_corpus(tokens):
    # tokens是1D列表或2D列表 ,将以词元为单位平铺
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


vocab = Vocab(tokens)
print(vocab.token_freqs)
print(f'self.idx_to_token{vocab.idx_to_token}')
print(f'self.token_to_idx{vocab.token_to_idx}')
print(f'self.len{len(vocab)}')
a = [('the', 'and'), ('was'), ['was'], [('the'), ['the']]]
print(f'self.item{vocab.__getitem__(a)}')
indices = [0, 1, 2, 100]
print(f'self.to_token{vocab.to_tokens(indices)}')
print(f'token_to_idx类型 === {type(vocab.token_to_idx)}')
print(f'=================\n')
# 前10行文本转换成索引列表
for i in range(10):
    print(f'文本行类型', type(tokens[i]))
    print(f'第{i}行', tokens[i])
    print(f'索引', vocab[tokens[i]])  # vocab['the'] 等价于调用getitem方法


# 整合所有功能
def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus,vocab


corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab),vocab.idx_to_token)
