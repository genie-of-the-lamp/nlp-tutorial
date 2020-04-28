#-*- coding:utf-8 -*-

# Byte Pair Encoding (BPE), 단어 분리 하기
# : OOV문제를 완화하는 대표적 단어 분리 토크나이저

import re, collections

num_merges = 10  # BPE 수행 횟수
vocab = {'l o w </w>': 5,
         'l o w e r </w>': 2,
         'n e w e s t </w>': 6,
         'w i d e s t </w>': 3}


# BPE code
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) -1):
            pairs[symbols[i], symbols[i+1]] += freq;
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)
print(vocab)
