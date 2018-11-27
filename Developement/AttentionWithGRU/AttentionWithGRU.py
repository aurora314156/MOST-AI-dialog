import os, sys, logging
import tensorflow as tf
import numpy as np
import time
from numpy import array, argmax
from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Dense, RepeatVector, TimeDistributed, Input
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import gc
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AttentionWithGRU():
    def __init__(self, CQADataSet, tqn):
        self.gru_units = 30
        self.model_fit_epochs = 2
        self.hops = 2
        self.CQADataSet = CQADataSet
        self.tqn = tqn
        
    def AttentionWithGRUMain(self):
        
        # final answer list
        guessAnsList = []
        for i in range(self.tqn):
            print("Processing number: ", i)
            # corpus content initial
            questionWordList = self.CQADataSet[i].getQuestion()
            storyWordList = self.CQADataSet[i].getCorpus()
            answerList = self.CQADataSet[i].getAnswer()
            sTime = time.time()
            # QuestionBidirectionalGRU input Vector
            forwardV, backwardV = self.OneHotEncoding(questionWordList), self.OneHotEncoding(list(reversed(questionWordList)))
            # StoryBidirectionalGRU 
            storyV = self.BidirectionalStoryGRU(storyWordList)
            #print("storyVector len:", len(storyV))
            # QuestionBidirectionalGRU
            questionV = self.BidirectionalGRU(forwardV, backwardV)
            #print("questionVector len:", len(questionV))
            # hops for n iteration
            for h in range(self.hops):
                # AttentionValue
                attentionValueV = self.AttentionValue(storyV, questionV)
                #print("attentionValueVector length:",len(attentionValueV))
                # WordLevelAttetion
                storyWordLevelV = self.WordLevelAttention(storyV, attentionValueV)
                #print("storyWordLevelV Shape:",storyWordLevelV.shape)
                # hops, VQn and VSn+1 summed to form a new question Vector VQn+1
                for j in range(len(questionV)):
                    storyWordLevelV[j] += questionV[j]
                
                # use final attention VS vector as next VQ vector
                questionV = storyWordLevelV
                forwardV, backwardV = questionV, np.flip(questionV, axis = 0)
                # QuestionBidirectionalGRU
                questionV = self.BidirectionalGRU(forwardV, backwardV)
                print("Finished {} hops summed!".format(h+1))
                # avoid tensorflow error
                gc.collect()
            
            # guess answer
            highestScoreAnswer = 0
            guessAnswer = 1
            ind = 1
            for a in answerList:
                # AnswerBidirectionalGRU input Vector
                ansForwardV, ansBackwardV = self.OneHotEncoding(a), self.OneHotEncoding(list(reversed(a)))
                # AnswerBidirectionalGRU
                answerV = self.BidirectionalGRU(ansForwardV, ansBackwardV)

                # use final attention VS vector as FINAL VQ vector
                forwardV, backwardV = questionV, np.flip(questionV, axis = 0)
                # FINAL GRU QuestionBidirectionalGRU
                questionV = self.BidirectionalGRU(forwardV, backwardV)

                # guess answer by calculate cosine value between storyV and answerV
                tempScoreAnswer = cosine(questionV, answerV)
                if highestScoreAnswer < tempScoreAnswer:
                    highestScoreAnswer = tempScoreAnswer
                    guessAnswer = ind
                print("CurrentAnswer score",tempScoreAnswer)
                print("HighestScoreAnswer score",highestScoreAnswer)
                ind += 1
            print("GuessAnswer: ", guessAnswer)
            guessAnsList.append(guessAnswer)
            print("Gru took: %.2fs" % (time.time()-sTime))
        return guessAnsList

    def WordLevelAttention(self, storyVector, attentionValueVector):

        storyVector = np.ravel(array(storyVector))
        storyVector = storyVector.tolist()
        wordLevelStoryVector = np.array([(storyVector[i] + storyVector[i+1]) * attentionValueVector[i] for i in range(len(attentionValueVector))])
        
        return wordLevelStoryVector

    def AttentionValue(self, storyVector, questionVector):
        # calculate AttentionValue, using cosine similarity between storyVector and questionVector^2
        # transpose question vector length to match up storyVector for calculate cosine similarity

        #qwe = np.pad(questionVector, pad_width = (0, len(storyVector)-len(questionVector)), mode = 'constant')
       
        attentionValue = []
        for index in range(len(storyVector)):
            attentionValue.append(cosine(storyVector[index], np.square(questionVector)))
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
        temp_all_hidden_state, temp_final_hidden_state = GRU(self.gru_units, return_sequences=True, return_state=True, activation='softmax')(inputs)
        model = Model(inputs=inputs, outputs=[temp_all_hidden_state, temp_final_hidden_state])
        # define input data
        data = inputV.reshape(1,seqlen,1)
        # train model using encoder method
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
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
        history = model.fit(train_x, train_x, self.model_fit_epochs)
        print()
        
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
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