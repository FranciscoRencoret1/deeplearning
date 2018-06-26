from keras.models import Model
from keras.layers import Input, Dense, Dropout, CuDNNGRU, concatenate, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.regularizers import l2

#Generate model and return model compiled, ready for fitting
def generate_model(maxPassageLen, maxQuestionLen):
  # Process question input and feed to bidirectional layer
  inputQuestion = Input(shape=(maxQuestionLen,300))
  bidirectionalQuestion1 = Bidirectional(CuDNNGRU(units=128, return_sequences=True))(inputQuestion)
  dropoutQuestion1 = Dropout(0.2)(bidirectionalQuestion1)
  bidirectionalQuestion2 = Bidirectional(CuDNNGRU(units=128, return_sequences=True),
                                       merge_mode = 'concat')(dropoutQuestion1)
  dropoutQuestion2 = Dropout(0.2)(bidirectionalQuestion2)
  
  # Process context input and feed to bidirectional layer
  inputPassage = Input(shape=(maxPassageLen,300))
  bidirectionalPassage1 = Bidirectional(CuDNNGRU(units=128, return_sequences=True))(inputPassage)
  dropoutPassage1 = Dropout(0.2)(bidirectionalPassage1)
  bidirectionalPassage2 = Bidirectional(CuDNNGRU(units=128, return_sequences=True),
                                       merge_mode = 'concat')(dropoutPassage1)
  dropoutPassage2 = Dropout(0.2)(bidirectionalPassage2)

  # Concatenate the output of the two bidirectionals and feed to FC layer
  concat = concatenate([dropoutQuestion2, dropoutPassage2], axis = 1)
  flatten = Flatten()(concat)
  fc1 = Dense(500, activation='relu', kernel_regularizer = l2(0.025))(flatten)
  dropout = Dropout(0.5)(fc1)
  output1 = Dense(maxPassageLen, activation='softmax')(dropout)
  output2 = Dense(maxPassageLen, activation='softmax')(dropout)

  model = Model(inputs=[inputQuestion,inputPassage], outputs=[output1, output2])
  model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
  return model