import model
import preprocess
import textGen
import json
import pickle
from keras.callbacks import History, ModelCheckpoint, EarlyStopping

TRAIN_DIR = "data/train.json"
TEST_DIR = "data/test.json"
GLOVE_DIR = "glove/glove.6B.300d.txt"
FILTER = ''' !"#$%&()*+,-./:;<=>?@[]^_{|}~\ '''
WEIGHTS_FILEPATH = "model.hdf5"

train_data, test_data = preprocess.load_data(TRAIN_DIR, TEST_DIR)
glove = preprocess.load_glove(GLOVE_DIR)

Xc, Xq, Ys, Ye, mapper, maxContextLen, maxQuestionLen = preprocess.preprocess_data(train_data, glove, FILTER)
Xct, Xqt, Yst, Yet, mappert, maxContextLent, maxQuestionLent = preprocess.preprocess_data(test_data, glove, FILTER)

gen_train = textGen.generate_data(Xc, Xq, Ys, Ye, mapper, maxContextLen, maxQuestionLen, 256)
gen_test = textGen.generate_data(Xct, Xqt, Yst, Yet, mappert, maxContextLen, maxQuestionLen, 256)

filepath="checkpoints/qa-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history = History()
callbacks_list = [checkpoint, history]

print("Max Question Length: {}".format(maxQuestionLen))
print("Max Context Length: {}".format(maxContextLen))

print("test data len {}".format(len(test_data)))

model = model.generate_model(maxContextLen, maxQuestionLen)
model.fit_generator(gen_train, steps_per_epoch = len(train_data)//256, epochs = 50, callbacks=callbacks_list, validation_data = gen_test, validation_steps = len(test_data) // 256)
model.save_weights(WEIGHTS_FILEPATH)
with open('history.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
