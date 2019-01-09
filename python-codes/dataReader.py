import time
from collections import namedtuple
import os
import numpy as np
import tensorflow as tf
fileName = "anna.txt"
filePath = os.getcwd()[:os.getcwd().rfind("/")+1]+fileName
# print(fileName)

with open(filePath,'r') as f:
    text = f.read()
    # print(text)
vocab = sorted(set(text))
vocab_to_num = {value:key for key,value in enumerate(vocab)}
num_to_vocab = {key:value for key,value in enumerate(vocab)}
encoded = np.array([vocab_to_num[c] for c in text], dtype=np.int32)
# print(vocab_to_num)
# print(num_to_vocab)
# print(encoded)
# print(text[:100])
print(len(vocab))
