import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json

# 使用GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置填充和起始标志
PAD_token = 0
SOS_token = 1
EOS_token = 2


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        # 词 -> 索引
        self.word2index = {}
        # 索引 -> 词
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        # 词 -> 统计出现次数
        self.word2count = {}
        self.num_words = 3  # 统计已出现的词数，已有3种标志

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    # 去除出现次数过少数据
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []  # 保留词汇

        new_word2count = {}
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
                new_word2count[k] = v

        print('词汇保留情况如下 {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index),
                                                         len(keep_words) / len(self.word2index)))

        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

        self.word2count = new_word2count


# vocab = Voc("test")
# vocab.addSentence("hello world hello python python python test ok ok ok")
# print(vocab.word2index)
# print(vocab.word2count)
# vocab.trim(0)
# print(vocab)
# print(vocab.word2index)
# print(vocab.word2count)


MAX_LENGTH = 10  # 最多仅考虑10个词


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())  # 字母小写
    s = re.sub(r"([.!?])", r" \1", s)  # 确保标点和字母间存在空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 保留字母和部分标点
    s = re.sub(r"\s+", r" ", s).strip()  # 去除多余空格
    return s


# input_string = " Hello, World! This is an example sentence... "
# normalized_string = normalizeString(input_string)
# print(normalized_string)


def readVocs(datafile, corpus_name):
    print("读取中...")
    lines = open(datafile, encoding='utf-8'). \
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


def filterPair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("开始准备训练数据 ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("读取 {!s} 对句子对".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("筛选出 {!s} 对句子对".format(len(pairs)))
    print("统计中...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("单词总计:", voc.num_words)
    return voc, pairs


save_dir = os.path.join("data", "save")
corpus_name = "movie-corpus"
corpus = os.path.join("data", corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

MIN_COUNT = 3


def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("从 {} 对筛选到 {}，总计保留了 {:.4f} ".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


pairs = trimRareWords(voc, pairs, MIN_COUNT)
