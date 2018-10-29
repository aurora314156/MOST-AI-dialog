import sys
sys.path.append('../')
import logging
from os import listdir
from DevelopmentModeInitial import DevelopmentModeInitial
from SanityCheck.SanityCheck import SanityCheck
from AttentionWithGRU.AttentionWithGRU import AttentionWithGRU

def AttentionMethod(CQADataSet, w2vmodel, tqn):
    for cqn in range(1):
        guessAnsList = AttentionWithGRU(CQADataSet, w2vmodel, tqn).AttentionWithGRUMain()
        print(guessAnsList)
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
            print("Process model: %s" %m)
            CQADataSet, w2vmodel = DevelopmentModeInitial(qasp, modelPath, tqn, data).getCQADataSetAndModel()
            AttentionMethod(CQADataSet, w2vmodel, tqn)
            #SanityCheckMethod(CQADataSet, w2vmodel, tqn)
            
    
if __name__ == "__main__":
    main()
