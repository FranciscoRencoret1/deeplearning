from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, Flatten, Lambda, Activation, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.regularizers import l2
import keras.backend as K
from keras.engine.topology import Layer
from keras import optimizers, initializers
from keras.engine import InputSpec
from keras.activations import tanh, softmax

class PointerGRU(GRU):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape
        self.input_length = []
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        initialization_seed = initializers.get('orthogonal')
        self.W1 = initialization_seed((self.hidden_shape, 1))
        self.W2 = initialization_seed((self.hidden_shape, 1))
        self.vt = initialization_seed((input_shape[1], 1))
        self.trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1]-1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_states(x_input)

        constants = super().get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                            initial_states,
                                            go_backwards=self.go_backwards,
                                            constants=constants,
                                            input_length=input_shape[1])
        return outputs

    def step(self, x_input, states):
        #print "x_input:", x_input, x_input.shape
        # <TensorType(float32, matrix)>
        
        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super().step(x_input, states[:-1])

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = TimeDistributed(Dense(en_seq, self.W1, output_dim=1))
        Dij = TimeDistributed(Dense(dec_seq, self.W2, output_dim=1))
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])








#Generate model and return model compiled, ready for fitting
def generate_model(maxContextLen, maxQuestionLen):
    # Process question input and feed to bidirectional layer
    inputQuestion = Input(shape=(maxQuestionLen,300))

    bidirectionalQuestion1 = Bidirectional(GRU(units=128, return_sequences=True))(inputQuestion)
    dropoutQuestion1 = Dropout(0.2)(bidirectionalQuestion1)
    bidirectionalQuestion2 = Bidirectional(GRU(units=128, return_sequences=True),
                                       merge_mode = 'concat')(dropoutQuestion1)
    dropoutQuestion2 = Dropout(0.2, name='QuestionRNNOutput')(bidirectionalQuestion2)

    # Process context input and feed to bidirectional layer
    inputContext = Input(shape=(maxContextLen,300))
    bidirectionalContext1 = Bidirectional(GRU(units=128, return_sequences=True))(inputContext)
    dropoutContext1 = Dropout(0.2)(bidirectionalContext1)
    bidirectionalContext2 = Bidirectional(GRU(units=128, return_sequences=True),
                                       merge_mode = 'concat')(dropoutContext1)
    dropoutContext2 = Dropout(0.2, name='ContextRNNOutput')(bidirectionalContext2)

    # Atention Layer
    similarityMatrix = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,2]), name='ContextDotQuestion')([dropoutContext2, dropoutQuestion2])
    softmaxMatrix = Activation('softmax', name='SoftmaxMatrix')(similarityMatrix)
    contextToQuery = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]), name='ContextToQuery')([softmaxMatrix, dropoutQuestion2])
    attentionConcatContext = Concatenate(axis=2)([dropoutContext2, contextToQuery])
    attentionContextCoeficients = Dense(maxQuestionLen, activation='sigmoid', name='Atention')(attentionConcatContext)
    
    # hp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[]))(dropoutContext2, attentionContextCoeficients)


    # Pointer Networks Layer
    pointerLayer = PointerGRU(128, output_dim=128, initial_state_provided=True, name='PointerLayer')([dropoutContext2, attentionContextCoeficients])

  
    fc1 = Dense(500, activation='relu', kernel_regularizer = l2(0.025))(attentionConcatContext)
    dropout = Dropout(0.5)(fc1)
    output1 = Dense(maxContextLen, activation='softmax')(dropout)
    output2 = Dense(maxContextLen, activation='softmax')(dropout)

    model = Model(inputs=[inputQuestion,inputContext], outputs=[output1, output2])
    model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
    return model