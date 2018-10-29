import os
import sys
import logging
import numpy as np
from gensim.models import Word2Vec
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class AttentionWithGRU():
    def __init__(self, CQADataSet, w2vmodel, tqn):
        self.CQADataSet = CQADataSet
        self.w2vmodel = w2vmodel
        self.tqn = tqn

    def AttentionWithGRUMain(self):
        
        guessAnsList = []
        questionWordList = self.CQADataSet[0].getQuestion()
        
        # QuestionBidirectionalGRU
        QueBidirGRU = QuestionBidirectionalGRU(questionWordList)

        # StoryBidirectionalGRU 
        #SotryBidirectionalGRU()
        
        # AnswersBidirectionalGRU 
        #AnswersBidirectionalGRU()
        
        # Attention 
        #Attention()

        return guessAnsList

    def QuestionBidirectionalGRU(self, questionWordList):
        
        # forward
        for q in questionWordList:
            
        # backward
    
    
