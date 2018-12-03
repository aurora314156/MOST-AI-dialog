import os, sys, logging, gc
import tensorflow as tf
import numpy as np
import time, math
from numpy import array, argmax
from keras.models import Sequential, Model
from keras.layers import LSTM, CuDNNGRU, Dense, RepeatVector, TimeDistributed, Input, GRU
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AttentionWithGRU():
    def __init__(self, qW, sW, aL):
        self.gru_units = 50
        self.model_fit_epochs = 2
        self.hops = 2
        self.questionWordList = qW
        self.storyWordList = sW
        self.answerList = aL
        
    def AttentionWithGRUMain(self):
    
        eachTime = time.time()
        # QuestionBidirectionalGRU input Vector
        forwardV, backwardV = self.OneHotEncoding(self.questionWordList), self.OneHotEncoding(list(reversed(self.questionWordList)))
        #print("forV len:",len(forwardV))
        # StoryBidirectionalGRU 
        storyV = self.BidirectionalStoryGRU(self.storyWordList)
        #print("storyVector len:", len(storyV))
        # QuestionBidirectionalGRU
        questionV = self.BidirectionalGRU(forwardV, backwardV)
        #print(questionV.shape)
        # hops for n iteration
        for h in range(self.hops):
            print("Start processing hops summed.")
            # AttentionValue
            #print("storyVector len:", len(storyV))
            attentionValueV = self.AttentionValue(storyV, questionV)
            #print("attentionValueVector length:",len(attentionValueV))
            # WordLevelAttetion
            storyWordLevelV = self.WordLevelAttention(storyV, attentionValueV)
            #print(storyWordLevelV.shape)
            # hops, VQn and VSn+1 summed to form a new question Vector VQn+1
            if len(questionV)> len(storyWordLevelV):
                summend_len = len(storyWordLevelV)
            else:
                summend_len = len(questionV)

            for j in range(summend_len):
                storyWordLevelV[j] += questionV[j]
            # use final attention VS vector as next VQ vector
            forwardV, backwardV = storyWordLevelV, np.flip(storyWordLevelV, axis = 0)
            # QuestionBidirectionalGRU
            questionV = self.BidirectionalGRU(forwardV, backwardV)

            print("Finished {} hops summed!".format(h+1))

        # guess answer
        print("Start calculate answer vector.")
        highestScoreAnswer = 0
        guessAnswer = 1
        ind = 1
        for a in self.answerList:
            # AnswerBidirectionalGRU input Vector
            ansForwardV, ansBackwardV = self.OneHotEncoding(a), self.OneHotEncoding(list(reversed(a)))
            # AnswerBidirectionalGRU
            answerV = self.BidirectionalGRU(ansForwardV, ansBackwardV)
            # use final attention VS vector as FINAL VQ vector
            # guess answer by calculate cosine value between storyV and answerV
            #tempScoreAnswer = cosine(questionV, answerV)
            tempScoreAnswer = cosine_similarity(questionV.reshape(1,-1), answerV.reshape(1,-1))
            if highestScoreAnswer < tempScoreAnswer:
                highestScoreAnswer = tempScoreAnswer
                guessAnswer = ind
            #print("CurrentAnswer score",tempScoreAnswer)
            #print("HighestScoreAnswer score",highestScoreAnswer)
            ind += 1
    
        #gc.collect()
        print("GuessAnswer: ", guessAnswer)
        print("This epoch took: %.2fs" % (time.time()-eachTime))
    
        return guessAnswer

    def WordLevelAttention(self, storyVector, attentionValueVector):

        storyVector = np.ravel(array(storyVector))
        storyVector = storyVector.tolist()
        wordLevelStoryVector = np.array([(storyVector[i] + storyVector[i+1]) * attentionValueVector[i] for i in range(len(attentionValueVector))])
        del storyVector
        return wordLevelStoryVector

    def AttentionValue(self, storyVector, questionVector):
        # calculate AttentionValue, using cosine similarity between storyVector and questionVector^2
        # transpose question vector length to match up storyVector for calculate cosine similarity
        attentionValue = []
        for index in range(len(storyVector)):
            storyVectorElem = storyVector[index].reshape(1,-1)
            questionVector = np.square(questionVector).reshape(1, -1)

            if math.isnan(cosine_similarity(storyVectorElem, questionVector)):
                attentionValue.append(0)
            else:
                attentionValue.append(cosine_similarity(storyVectorElem, questionVector))
        # AttentionValue normalization (actually is softmax in this paper...)
        exps = [np.exp(i) for i in attentionValue]
        sum_of_exps = sum(exps)
        attentionValue_softmax = [j/sum_of_exps for j in exps]

        return attentionValue_softmax

    def BidirectionalGRU(self, forwardV, backwardV):
        # forward vector hidden state
        f_all_hidden_state, f_final_hidden_state = self.GRU(forwardV)
        # backward vector hidden state
        b_all_hidden_state, b_final_hidden_state = self.GRU(backwardV)
        # concat forward vector and backward vector
        forwardVector, backwardVector = f_final_hidden_state, b_final_hidden_state
        # print(forwardVector.shape)
        # print(backwardVector.shape)
        quesitonVector = np.concatenate((forwardVector,backwardVector), axis=None)
        # print(quesitonVector.shape)
        
        return quesitonVector

    def BidirectionalStoryGRU(self, storyWordList):
        # forward vector
        forwardV = self.OneHotEncoding(storyWordList)
        f_all_hidden_state, f_final_hidden_state = self.GRU(forwardV)
        # print(f_all_hidden_state.shape)
        # backward vector
        backwardV = self.OneHotEncoding(list(reversed(storyWordList)))
        b_all_hidden_state, b_final_hidden_state = self.GRU(backwardV)
        # print(b_all_hidden_state.shape)
        # The word vector representation of the t-th word St is constructed 
        # by concatenating the hidden layer outputs of forward and backward GRU networks
        storyVector = []
        for index in range(len(f_all_hidden_state[0])):
           storyVector.append(np.concatenate((f_all_hidden_state[0][index],b_all_hidden_state[0][index]), axis=None))
        
        return storyVector

    def GRU(self, inputV):
        # define timesteps
        seqlen = len(inputV)
        # define model, save GRU all hidden state and final hidden state for question vector representation
        inputs = Input(shape=(seqlen,1))
        # for cpu version GRU
        #temp_all_hidden_state, temp_final_hidden_state = GRU(self.gru_units, return_sequences=True, return_state=True)(inputs)
        temp_all_hidden_state, temp_final_hidden_state = CuDNNGRU(units=self.gru_units, return_sequences=True, return_state=True)(inputs)
        model = Model(inputs=inputs, outputs=[temp_all_hidden_state, temp_final_hidden_state])
        # define input data
        data = inputV.reshape((1,seqlen,1))
        # train model using encoder method
        model.compile(optimizer='adam', loss='mean_squared_error')
        # train model
        #model.fit(data, data, epochs = self.model_fit_epochs)
        # 
        mp = model.predict(data, verbose = 1)
        
        all_hidden_state, final_hidden_state = mp[0], mp[1]

        return all_hidden_state, final_hidden_state

    def GRUModelEvalute(self, inputV):
        # keras GRU cell
        # reshape input into [samples, timesteps, features]
        train_x = inputV
        print(inputV.shape)
        n_in = len(inputV)
        train_x = train_x.reshape((1, n_in, 1))
        # define model
        model = Sequential()
        model.add(GRU(self.gru_units, activation='relu', input_shape=(n_in,1)))
        model.add(RepeatVector(n_in))
        model.add(GRU(self.gru_units, activation='relu', return_sequences=True, return_state=True))
        model.add(TimeDistributed(Dense(1, activation='relu')))
        model.compile(optimizer='adam', loss='mean_squared_error')

        print(model.summary())
        #history = model.fit(train_x, train_x, self.model_fit_epochs)
        print()
        # Plot training & validation loss values
        # plt.plot(history.history['loss'])
        # #plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        print()

    def OneHotEncoding(self, WordList):
        # dict transfer to array
        values = array(WordList)
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        # create one-dim oneHotVector, only need to take maximum number building one-hot vector 
        # from array then merge all one hot encodeing vector to one-dim vector.
        oneHotV = np.zeros((len(integer_encoded), integer_encoded.max()+1))
        oneHotV[np.arange(len(integer_encoded)), integer_encoded] = 1
        oneHotV = oneHotV.ravel()
        
        return oneHotV



# Plot training & validation accuracy values
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()