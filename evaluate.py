from __future__ import print_function
import model
import preprocess
import textGen
import json
import numpy as np
import keras.backend as K

model = model.generate_model(maxContextLen, maxQuestionLen)
model.load_weights('checkpoints/qa-weights-improvement-06-9.8426.hdf5')

TRAIN_DIR = "data/train.json"
TEST_DIR = "data/test.json"
GLOVE_DIR = "glove/glove.6B.300d.txt"
FILTER = ''' !"#$%&()*+,-./:;<=>?@[]^_{|}~\  '''

train_data, test_data = preprocess.load_data(TRAIN_DIR, TEST_DIR)
glove = preprocess.load_glove(GLOVE_DIR)

a, b , c , d, m, maxContextLen, maxQuestionLen = preprocess.preprocess_data(train_data, glove, FILTER)
Xc, Xq, Ys, Ye, mapper, maxContextLen, maxQuestionLen = preprocess.preprocess_data(test_data, glove, FILTER)

gen_test = textGen.generate_data(Xc, Xq, Ys, Ye, mappert, maxContextLen, maxQuestionLen, 128)

YsTest, YeTest = model.predict_generator(gen_test, steps=(Xq_test.shape[0] // 128))

def transformYtoText(Ys, Ye):
  dic = {}
  i = 0
  for article in test_data:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        for q in qa['question']:
          clear_paragraph = paragraph['context'].translate(translator).split(' ')
          string_answer = ""
          answer = clear_paragraph[np.argmax(Ys[i]) : np.argmax(Ye[i])]
          for b in answer:
            string_answer += b
            string_answer += ' '
          string_answer = string_answer[: -1]
          dic.update({qa['id'] : string_answer})
          i += 1
          if i >= Xq_test.shape[0] // 128:
            break
  return dic

pred = transformYtoText(YsTest, YeTest)

""" Official evaluation script for v1.1 of the SQuAD dataset. """
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

print(evaluate(test_data, pred))