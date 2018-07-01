from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, Flatten, Lambda, Activation, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.regularizers import l2
import keras.backend as K
from keras.engine.topology import Layer
from keras import optimizers

#Generate model and return model compiled, ready for fitting
def generate_model(maxContextLen, maxQuestionLen):
  # Process question input and feed to bidirectional layer
  inputQuestion = Input(shape=(maxQuestionLen,300))

  bidirectionalQuestion1 = Bidirectional(GRU(units=128, return_sequences=True))(inputQuestion)
  dropoutQuestion1 = Dropout(0.2)(bidirectionalQuestion1)
  bidirectionalQuestion2 = Bidirectional(GRU(units=128, return_sequences=True),
                                       merge_mode = 'concat')(dropoutQuestion1)
  dropoutQuestion2 = Dropout(0.2)(bidirectionalQuestion2)
  
  # Process context input and feed to bidirectional layer
  inputContext = Input(shape=(maxContextLen,300))
  bidirectionalContext1 = Bidirectional(GRU(units=128, return_sequences=True))(inputContext)
  dropoutContext1 = Dropout(0.2)(bidirectionalContext1)
  bidirectionalContext2 = Bidirectional(GRU(units=128, return_sequences=True),
                                       merge_mode = 'concat')(dropoutContext1)
  dropoutContext2 = Dropout(0.2)(bidirectionalContext2)

  # Atention Layer
  similarityMatrix = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,2]), name='ContextDotQuestion')([dropoutContext2, dropoutQuestion2])
  softmaxMatrix = Activation('softmax')(similarityMatrix)
  contextToQuery = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]), name='SoftmaxDotQuestion')([softmaxMatrix, dropoutQuestion2])
  attentionConcatContext = Concatenate()([dropoutContext2, contextToQuery])
  dense = Dense(maxContextLen, activation='softmax')(attentionConcatContext)
  
  
  
  fc1 = Dense(500, activation='relu', kernel_regularizer = l2(0.025))(attentionConcatContext)
  dropout = Dropout(0.5)(fc1)
  output1 = Dense(maxContextLen, activation='softmax')(dropout)
  output2 = Dense(maxContextLen, activation='softmax')(dropout)

  model = Model(inputs=[inputQuestion,inputContext], outputs=[output1, output2])
  model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
  return model