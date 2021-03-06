import time
from collections import namedtuple
import os
import numpy as np
import tensorflow as tf

fileName = "anna.txt"
filePath = os.getcwd()[:os.getcwd().rfind("/") + 1] + fileName
# print(fileName)

with open(filePath, 'r') as f:
    text = f.read()
    # print(text)
vocab = sorted(set(text))
vocab_to_num = {value: key for key, value in enumerate(vocab)}
num_to_vocab = {key: value for key, value in enumerate(vocab)}
encoded = np.array([vocab_to_num[c] for c in text], dtype=np.int32)
# print(vocab_to_num)
# print(num_to_vocab)
# print(encoded)
# print(text[:100])
print(len(vocab))


def getBatches(arr, batchSize, nSteps):
    # number of character per batch. batch is a grid of dimension batchSize * nSteps
    charsPerBatch = batchSize * nSteps
    # total number of batch so that each batch has the same size // indicates Integer division
    totalBatch = len(arr) // charsPerBatch
    arr = arr[0:totalBatch * charsPerBatch]
    arr = np.reshape(arr, (batchSize, -1))

    # print(arr.shape[1])
    # print(totalBatch)
    # print(len(text))

    for n in range(0, arr.shape[1], nSteps):
        # input values for all rows (indicated by first :) from n to n+nSteps
        x = arr[:, n:n + nSteps]
        # target values for all rows (indicated by first :) from n+1 (that is the next char) to n+1+nSteps
        yTemp = arr[:, n + 1:n + nSteps + 1]

        # For the very last batch, y will be one character short at the end of
        # the sequences which breaks things. To get around this, I'll make an
        # array of the appropriate size first, of all zeros, then add the targets.
        # This will introduce a small artifact in the last batch, but it won't matter.
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:, :yTemp.shape[1]] = yTemp

        # same as return. but works only for generator function
        # more details on generator function
        # https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
        yield x, y


batches = getBatches(encoded, 10, 50)
x, y = next(batches)

print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])


def buildInput(batchSize, nSteps):
    # tensorflow placeholder for input
    input = tf.placeholder(tf.int32, [batchSize, nSteps], "input")
    # tensorflow placeholder for target
    target = tf.placeholder(tf.int32, [batchSize, nSteps], "target")
    # tensorflow placeholder for dropout wrapper with 0-d tensor
    keepProbForDropOut = tf.placeholder(tf.int32, name="keepProbForDropOut")
    return input, target, keepProbForDropOut


def buildCell(lstmSize, keepProb):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstmSize)
    dropOutWrapper = tf.contrib.rnn.DropoutWrapper(lstm)
    return dropOutWrapper


def buildLSTMNetwork(lstmSize, numOfLayer, batchSize, keepProb):
    dropoutWrapper = buildCell(lstmSize, keepProb)
    cell = tf.contrib.rnn.MultiRNNCell([(buildCell(lstmSize, keepProb)) for _ in range(numOfLayer)])
    initialState = cell.zero_state(batchSize, tf.int32)

    return cell, initialState

def buildOutput(lstmOutput,inputSize,outputSize):
    # concatenate the lstm output along y (indicated by axis=1) axis
    # as the lstmOutput is a long list
    sequenceOutput = tf.concat(lstmOutput,axis=1)
    x = tf.reshape(sequenceOutput,[-1,inputSize])

    with tf.variable_scope("softmax"):
        softmax_w = tf.Variable(tf.truncated_normal(inputSize,outputSize),stddev=0.01)
        softmax_b = tf.Variable(tf.zeros(outputSize))

    logits = tf.matmul(x,softmax_w) + softmax_b

    output = tf.nn.softmax(logits,name="prediction")

    return logits,output


def buildLoss(logits,targets,lstmSize,numOfClasses):
    targetsOneHot = tf.one_hot(targets,numOfClasses)
    targetsReshaped = tf.reshape(targetsOneHot,logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targetsReshaped)
    loss = tf.reduce_mean(loss)

    return loss

def buildOptimizer(loss,learningRate,gradientClip):
    trainableVariables = tf.trainable_variables()
    gradients, _ = tf.clip_by_global_norm(tf.gradients(loss,trainableVariables),gradientClip)

    trainingOptimizer = tf.train.AdamOptimizer(learningRate)
    optimizer = trainingOptimizer.apply_gradients(zip(gradients,trainableVariables))

    return optimizer






