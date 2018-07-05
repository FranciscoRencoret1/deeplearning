from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, Flatten, Lambda, Activation, CuDNNGRU, GRU,  Add
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.regularizers import l2
import keras.backend as K
from keras.engine.topology import Layer
from keras import optimizers, initializers
from keras.engine import InputSpec
from keras.activations import tanh, softmax
import tensorflow as tf
BATCH_SIZE= 128


class PointerGRU(GRU):
    def __init__(self, hidden_shape, *args, **kwargs):
        # kwargs['implementation'] = kwargs.get('implementation', 2)
        self.hidden_shape = hidden_shape
        self.input_length = []
        print(111)
        self.input_spec = [None] * 256 # TODO TODO TODO
        for arg in (args):
          print("arg: {}".format(arg))
        print("pipi{}".format(kwargs))
        super().__init__(*args, **kwargs)
        print(2222)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        B, Q, H = input_shape
        return (B, H)
    
    def build(self, input_shape):
        input_shape = input_shape[0]
        print(3333)
        print("INPUT SHAPE {}".format(input_shape))
        self.input_spec = [InputSpec(shape=input_shape)]
        self.GRU_input_spec = self.input_spec
        super().build(input_shape)
        initialization_seed = initializers.get('orthogonal')
        # self.W1 = self.add_weight(name='W1',
        #                           shape=(BATCH_SIZE, 256, 1),
        #                           initializer=initialization_seed,
        #                           trainable=True)
        # self.W2 = self.add_weight(name='W2',
        #                           shape=(BATCH_SIZE, 256, 1),
        #                           initializer=initialization_seed,
        #                           trainable=True)
        # self.vt = self.add_weight(name='vt',
        #                           shape=(BATCH_SIZE, input_shape[0][1], 1),
        #                           initializer=initialization_seed,
        #                           trainable=True)
        self.W1 = tf.convert_to_tensor(initialization_seed((256, 1)), dtype=tf.float32)
        self.W2 = tf.convert_to_tensor(initialization_seed((256, 1)), dtype=tf.float32)
        print('asfadfad{}'.format(input_shape))
        self.vt = tf.convert_to_tensor(initialization_seed((input_shape[1], 1)), dtype=tf.float32)
        self.trainable_weights.append(self.W1)
        self.trainable_weights.append(self.W2)
        self.trainable_weights.append(self.vt)
        print(self.trainable_weights)

    def call(self, x, mask=None):
        print(44444)
        x = x[0]
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

class QuestionPooling(Layer):

    def __init__(self, hidden_shape, **kwargs):
        super(QuestionPooling, self).__init__(**kwargs)
        self.hidden_shape = hidden_shape

    def compute_output_shape(self, input_shape):
        B, Q, H = input_shape
        
        return (B, H)

    def build(self, input_shape):
        initialization_seed = initializers.get('orthogonal')
        self.Wu = tf.convert_to_tensor(initialization_seed((self.hidden_shape, self.hidden_shape//2)), dtype=tf.float32)
        self.Wv = tf.convert_to_tensor(initialization_seed((self.hidden_shape, self.hidden_shape//2)), dtype=tf.float32)
        self.Vr = tf.convert_to_tensor(initialization_seed((self.hidden_shape//2, self.hidden_shape//2)), dtype=tf.float32)
        self.vt = tf.convert_to_tensor(initialization_seed((self.hidden_shape//2, 1)), dtype=tf.float32)
        self.trainable_weights += [self.Wu, self.Wv, self.Vr, self.vt]

    def call(self, inputs, mask=None):
        uQ = inputs
        ones = K.ones_like(K.sum(uQ, axis=1, keepdims=True))




        s_hat1 = K.dot(uQ, self.Wu)
        s_hat2 = K.dot(ones, K.dot(self.Wv, self.Vr))
        s_hat = K.tanh(Add()([s_hat1, s_hat2]))
        s = K.dot(s_hat, self.vt)
        s = K.batch_flatten(s)

        a = softmax(s, axis=1)

        rQ = K.batch_dot(uQ, a, axes=[1, 1])

        return rQ







#Generate model and return model compiled, ready for fitting
def generate_model(maxContextLen, maxQuestionLen):
    # Process question input and feed to bidirectional layer
    inputQuestion = Input(shape=(maxQuestionLen,300))

    bidirectionalQuestion1 = Bidirectional(CuDNNGRU(units=128, return_sequences=True))(inputQuestion)
    dropoutQuestion1 = Dropout(0.2)(bidirectionalQuestion1)
    bidirectionalQuestion2 = Bidirectional(CuDNNGRU(units=128, return_sequences=True),
                                       merge_mode = 'concat')(dropoutQuestion1)
    dropoutQuestion2 = Dropout(0.2, name='QuestionRNNOutput')(bidirectionalQuestion2)

    # Process context input and feed to bidirectional layer
    inputContext = Input(shape=(maxContextLen,300))
    bidirectionalContext1 = Bidirectional(CuDNNGRU(units=128, return_sequences=True))(inputContext)
    dropoutContext1 = Dropout(0.2)(bidirectionalContext1)
    bidirectionalContext2 = Bidirectional(CuDNNGRU(units=128, return_sequences=True),
                                       merge_mode = 'concat')(dropoutContext1)
    dropoutContext2 = Dropout(0.2, name='ContextRNNOutput')(bidirectionalContext2)

    # Atention Layer
    similarityMatrix = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,2]), name='similarityMatrix')([dropoutContext2, dropoutQuestion2])
    softmaxMatrix = Activation('softmax', name='SoftmaxSimilarity')(similarityMatrix)
    contextToQuery = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]), name='ContextToQuery')([softmaxMatrix, dropoutQuestion2])
    attentionConcatContext = Concatenate(axis=2)([dropoutContext2, contextToQuery])
    attentionContextCoeficients = Dense(256, activation='sigmoid', name='Atention')(attentionConcatContext)
    
    # permuteDims = Permute((2, 1), input_shape=(800, 256))(attentionConcatContext)
    # attentionContextCoeficients = Dense(maxContextLen, activation='sigmoid', name='Atention')(permuteDims)
    # embeddingDotAttention = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 1]), name='ContextDotAttention')([dropoutContext2, attentionContextCoeficients])


    #QuestionAware passage encoding GRU
    bidirectionalQAwarePassage = Bidirectional(CuDNNGRU(units=128, return_sequences=True))(attentionContextCoeficients)

    #QuestionPool
    rQ = QuestionPooling(256)(dropoutQuestion2)

    # Pointer Networks Layer
    # print("Dimentions: {}".format(bidirectionalQAwarePassage.get_shape()))
    # pointerLayer = PointerGRU(maxContextLen, output_dim=256, name='PointerLayer', return_sequences=True)(inputs=[bidirectionalQAwarePassage], initial_state =[rQ])


    # Flatten
    flat = Flatten()(bidirectionalQAwarePassage)

    preDense = Concatenate()([flat, rQ])
    fc1 = Dense(500, activation='relu', kernel_regularizer = l2(0.05))(preDense)
    dropout = Dropout(0.5)(fc1)
    fc2 = Dense(1000, activation='relu', kernel_regularizer = l2(0.05))(dropout)
    dropout2 = Dropout(0.5)(fc2)
    output1 = Dense(maxContextLen, activation='softmax')(dropout2)
    preOutput2 = Concatenate()([dropout2, output1])
    output2 = Dense(maxContextLen, activation='softmax')(preOutput2)

    model = Model(inputs=[inputQuestion,inputContext], outputs=[output1, output2])
    model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
  model = generate_model(400,100)
  print(model.summary())