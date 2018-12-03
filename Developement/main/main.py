import sys
sys.path.append('../')
import logging
import time
from keras import backend as K 
from os import listdir
from DevelopmentModeInitial import DevelopmentModeInitial
from SanityCheck.SanityCheck import SanityCheck
from AttentionWithGRU.AttentionWithGRU import AttentionWithGRU

def AttentionMethod(CQADataSet, tqn):

    correct_answer = [4,1,3,2,2,2,1,3,3,1,1,3,1,4,2,2,3,1,3,4,4,2,4,1,3,2,3,3,2,4,1,4,3,4,1,2,4,2,1,4,4,1,3,3,1,4,3,1,3,2,1,3,4,3,3,3,2,1,1,3,2,3,1,3,1,3,2,4,4,1,2,2,4,2,1,3,3,4,3,4,2,1,1,3,3,4,2,3,3,4,1,4,2,3,1,4,1,2,2,3,3,4,1,4,4,1,2,1,2,4,2,4,1,1,2,1,4,2,3,4,1,2,2,3,1,4,3,2,1,3,4,2,1,2,1,3,1,2,4,1,3,1,4,3,2,4,1,4,1,4,4,4,3,2,4,2,1,4,2,3,4,1,3,4,2,1,4,3,1,4,3,4,1,1,4,1,4,4,3,3,2,3,2,1,2,3,3,1,2,3,4,4,1,4,1,4,2,4,4,2,1,3,4,4,4,1,1,4,3,1,1,1,4,4,2,3,3,1,2,3,2,3,4,3,1,2,3,3,4,3,4,1,4,2,3,1,4,2,1,2,1,2,3,3,3,4,1,2,1,1,4,1,4,1,4,3,1,1,3,1,3,1,4,1,3,1,1,3,1,4,1,1,3,3,4,4,1,1,1,4,3,1,1,3,2,4,2,4,1,4,2,3,1,4,1,2,1,1,1,2,4,2,3,2,3,1,4,2,2,3,2,2,3,4,4,1,2,2,3,4,1,4,3,4,1,3,4,1,3,3,3,1,2,3,3,2,2,4,4,3,1,4,4,1,2,3,2,4,3,2,3,2,4,1,4,1,3,3,1,2,1,2,2,1,3,2,4,3,4,4,4,1,2,1,3,3,2,1,3,1,4,3,4,2,1,2,2,1,1,3,3,1,4,1,4,1,1,2,1,1,3,2,1,3,1,1,4,2,1,2,1,1,2,3,1,4,3,4,2,3,2,1,2,2,2,1,2,2,1,2,3,1,1,4,2,1,1,1,2,1,4,2,3,2,2,1,3,4,1,3,4,1,2,1,3,4,2,2,1,2,4,1,2,3,2,1,4,2,2,1,1,1,1,2,1,4,1,1,4,2,1,1,1,4,1,3,2,3,1,1,1,3,3,1,1,1,3,1,4,4,1,4,3,4,4,1,4,2,1,4,2,3,4,1,3,4,3,2,3,1,2,1,1,4,4,3,1,2,4,3,2,2,3,2,2,1,2,2,3,1,2,4,2,1,4,1,1,3,3,4,3,4,1,2,4,2,3,1,1,4,3,4,2,4,1,2,3,2,3,4,1,2,4,2,3,1,2,1,3,1,1,4,2,1,3,4,1,2,4,4,1,1,1,1,1,1,1,1,1,2,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,2,4,4,3,1,2,4,2,1,1,3,3,1,2,4,3,2,2,4,3,3,2,3,2,4,2,1,2,2,1,2,3,4,1,2,3,4,1,2,1,2,3,4,1,2,4,3,1,2,3,4,1,1,1,4,3,4,2,3,3,4,2,2,4,2,2,2,3,4,1,4,4,3,3,3,4,2,2,2,3,2,3,1,2,3,4,3,2,2,1,1,2,3,4,3,4,4,4,2,3,1,4,1,4,2,3,2,4,1,4,2,4,2,2,3,1,2,3,1,3,1,2,4,2,2,4,4,3,1,3,4,1,4,2,1,4,3,3,4,2,3,3,3,1,4,4,2,2,1,3,2,1,2,3,4,2,1,1,3,2,3,4,3,1,2,3,4,1,2,1,3,1,4,2,2,2,1,3,4,2,3,4,3,1,2,2,3,3,1,1,2,4,3,1,2,2,1,4,2,3,3,4,1,2,4,3,2,3,4,2,3,1,2,2,3,4,3,4,1,2,3,2,1,4,2,3,4,3,1,3,3,2,4,2,4,1,2,3,1,3,4,2,2,3,1,2,4,2,1,3,2,1,4,1,4,1,3,1,2,3,1,1,3,1,3,3,3,3,4,1,3,2,4,1,2,1,3,1,1,1,4,2,1,3,4,1,3,2,1,3,1,1,1,4,3,1,3,2,1,3,3,2,1,1,2,3,4,4,1,3,4,1,1,1,3,2,3,3,2,1,4,4,3,4,3,3,2,4,4,1,1,4,1,2,2,4,1,1,4,2,4,3,3,3,2,1,3,2,3,4,1,2,3,4,2,1,2,3,3,2,4,1,4,1,3,3,2,3,3,2,4,3,1,2,4,3,4,3,1,4,1,2,4,1,2,4,1,3,1,2,3,2,4,1,4,4,1,1,3,2,2,3,3,3,1,4,4,4,1,3,1,4,4,4,3,4,4,1,3,2,3,4,1,3,2,3,2,3,1,3,2,1,3,2,2,3,4,2,1,4,2,3,2,1,4,1,3,4,1,3,2,4,2,4,1,2,1,2,1,4,1,3,1,3,4,2,4,3,2,1,4,2,1,3,2,1,1,3,4,2,3,4,4,2,2,3,1,4,2,3,1,2,3,2,1,1,4,2,1,1,2,3,1,1,1,1,1,1,1,3,1,1,2,3,4,4,3,1,2,2,4,2,2,2,2,2,3,3,4,1,3,2,2,1,2,3,2,2,3,4,2,1,3,2,3,2,4,2,4,3,2,1,3,3,1,3,2,4,2,2,3,1,3,3,3,2,1,2,2,1,3,1,4,3,2,3,4,2,4,2,3,2,3,4,1,1,3,1,3,1,2,2,1,3,2,3,1,3,4,2,1,3,2,2,3,1,4,2,1,2,1,3,4,3,3,2,1,4,2,4,1,3,1,1,1,3,3,2,2,1,1,1,1,2,1,3,1,4,3,4,2,4,2,3,1,2,4,4,3,4,2,1,4,2,1,1,3,2,1,1,4,2,1,3,1,2,2,1,3,3,1,4,3,4,4,1]
    guess_correct = 0
    guessAnsList = []
    # final answer list
    sTime = time.time()
    for i in range(tqn):
        print("Processing number: ", i)
        # corpus content initial
        questionWordList = CQADataSet[i].getQuestion()
        storyWordList = CQADataSet[i].getCorpus()
        answerList = CQADataSet[i].getAnswer()
        guessAns = AttentionWithGRU(questionWordList, storyWordList, answerList).AttentionWithGRUMain()
        guessAnsList.append(guessAns)
        K.clear_session()
    
    print("Total took: %.2fs" % (time.time()-sTime))
    for i in range(tqn):
        if guessAnsList[i] == correct_answer[i]:
            guess_correct +=1
    print("Final guess correctness: ", guess_correct)
        
def SanityCheckMethod(CQADataSet, model, tqn):
    
    x = 0.1
    bestX = 0
    highestCorrectCount  = 0
    
    # initital idfTable
    idfTable = SanityCheck(CQADataSet, tqn, 0).calIDF()
    print(len(idfTable))
    while x < 1:
        tempCorrectCount = 0
        for cqn in range(tqn):
            ans = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x, idfTable)
            if ans == CQADataSet[cqn].correct_answer:
                tempCorrectCount += 1
        
        if highestCorrectCount < tempCorrectCount:
            highestCorrectCount = tempCorrectCount
            bestX = x
        
        x += 0.1

    return bestX, highestCorrectCount/tqn


def main(argv=None):

    # initial setting
    data = 'MOST'
    # development mode
    print("\nStart development mode.\n\nInput parameters [1]qasp [2]mp.\n")
    qasp = sys.argv[1]
    mp = sys.argv[2]
    tqn = len(listdir(qasp))
    # get all instance and load word2vec model
    print("Start create CQA instance and load model.\n")
    
    # start SanityCheckMethod iteration
    #x, accuracy, bestX, bestAccuracy = 0, 0, 0, 0
    
    modelFiles = listdir(mp)
    bestModel = ""
    for m in modelFiles:
        if m[len(m)-6:len(m)] == ".model":
            modelPath = mp + m
            print("W2V Model: %s" %m)
            CQADataSet, w2vmodel = DevelopmentModeInitial(qasp, modelPath, tqn, data).getCQADataSetAndModel()
            AttentionMethod(CQADataSet, tqn)
            #SanityCheckMethod(CQADataSet, w2vmodel, tqn)
            
    
if __name__ == "__main__":
    main()
