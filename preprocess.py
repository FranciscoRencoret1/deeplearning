from keras.preprocessing.text import text_to_word_sequence
import os
import json
import numpy as np
from keras.utils import to_categorical

# text = 'Architecturally, the school has a Catholic character. Atop the Main Buildings gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
# result = text_to_word_sequence(text, filters="^'", lower=True, split='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ')
# print(result)

# Load Dataset and preprocess
# Define train and test files
TRAIN_DATA_DIR = 'data/train.json'
TEST_DATA_DIR = 'data/test.json'
GLOVE_DIR = 'glove/glove.6B.300d.txt'
FILTER_CHAR = '''!#%&()*+,-./:;<=>?@[\]^_`'{|}~"°£₹¢₥’—‘$'''

def load_data(train_data_dir, test_data_dir):
    with open(train_data_dir) as f:
        train_data = json.load(f)['data']

    with open(test_data_dir) as f:
        test_data = json.load(f)['data']

    return train_data, test_data

def load_glove(glove_dir):
    # Create dictionary to map word to glove vectors
    glove = {}
    with open(glove_dir) as file:
        for line in file:
            line = line.strip().split(' ')
            word = line[0]
            glove_vector = line[1: ]
            glove.update({word: glove_vector})
    return glove

def get_glove_repr(text, glove):
    # Create array for glove reprentation of each word of the Text
    text_glove_repr = []
    # First, we clean the context filtering by puntuation marks 
    clean_text = text_to_word_sequence(text, filters=FILTER_CHAR, lower=True, split=' ')
    for word in clean_text:
        # Try finding word in glove dic, create vector of 0s if word is not found in dic
        if word in glove.keys(): word_glove_repr = glove[word]
        else: word_glove_repr = np.zeros((300))
        text_glove_repr.append(word_glove_repr)
    return text_glove_repr, clean_text

def preprocess_data(dataset, glove): 
    # Create array for Context
    Xc = []
    # Create array for Questions
    Xq = []
    # Create array for start index on answer
    Ys = []
    # Create array for end index on answer
    Ye = []
    # Create mapping array for Question and Context. There are many Questions per Context, so we create
    # this mapping array to avoid loading repeated Contexts in memory.
    mapper = []

    # Iterate over the JSON with the data to obtain Contexts, Questions, Start Index and End Index
    for data in dataset:
        # Convert Context to glove vectors
        for paragraph in data['paragraphs']:
            # Get glove representation of Context
            context_glove_repr, clean_context = get_glove_repr(paragraph['context'], glove)
            # print(clean_context)
            # print(context_glove_repr)
            # Append context to Xc, the array of contexts
            Xc.append(context_glove_repr)
        
            #Convert question and answer to glove vectors
            for qa in paragraph['qas']:
                # Get glove representation of Question
                question_glove_repr, clean_question = get_glove_repr(qa['question'], glove)

                # Work on answer. We must get the answer's end index of the context. We iterate because there
                # could be several answes for the same question
                for answer in qa['answers']:
                    # Filter answer
                    clean_answer = text_to_word_sequence(answer['text'], filters=FILTER_CHAR, lower=True, split=' ')
                    start_point_char = answer['answer_start']
                    end_point_char = start_point_char + len(answer['text'])
                    complete_answer = text_to_word_sequence(paragraph['context'][start_point_char : end_point_char], filters=FILTER_CHAR, lower=True, split=' ')

                    # TEMP Clean some bad data
                    if len(complete_answer) > 0:
                        indices = [i for i, x in enumerate(clean_context) if x == complete_answer[-1]]
                        for i in indices:
                            if clean_context[i - (len(complete_answer) - 1)] == complete_answer[0]:
                                start_point = i - (len(complete_answer) - 1)
                                end_point = i
                                Ys.append(start_point)
                                Ye.append(end_point)
                                # Append question to Xq, the array of questions
                                Xq.append(question_glove_repr)
                                mapper.append(data['paragraphs'].index(paragraph))
                                break
    maxPassageLen = max(len(l) for l in Xc)
    maxQuestionLen = max(len(l) for l in Xq)
    return np.array(Xc).T, np.array(Xq).T, to_categorical(Ys, num_classes=maxPassageLen), to_categorical(Ye, num_classes=maxPassageLen), mapper, maxPassageLen, maxQuestionLen

train_data, test_data = load_data(TRAIN_DATA_DIR, TEST_DATA_DIR)
glove = load_glove(GLOVE_DIR)

Xc, Xq, Ys, Ye, mapper, maxPassageLen, maxQuestionLen = preprocess_data(train_data, glove)
# print(Xc[0])
print('Len Xc ' + str(len(Xc)))
# print(Xq[0])
print('Len Xq ' + str(len(Xq)))
# print(Ys[0])
print('Len Ys ' + str(len(Ys)))
# print(Ye[0])
print('Len Ye ' + str(len(Ye)))
# print(mapper[:50])
print('Len mapper ' + str(len(mapper)))
print(maxPassageLen, maxQuestionLen)
# preprocess_data(train_data, glove)





