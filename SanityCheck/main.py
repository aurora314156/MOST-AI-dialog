
import sys
import random
import logging
from os import listdir
from ContestModeInitial import ContestModeInitial
from DevelopmentModeInitial import DevelopmentModeInitial
from SanityCheck import SanityCheck

"""
    Parameters
    --------------------------------------
    QuestionAnswerSetPath: qasp
    ModelPath: mp
    AnswerPath: ap
    TotalQuestionDataSetAmount: tqn
    CurrentQuestionNumber: cqn
    mode [different mode]: cwt, MOST ..... 
    CWTInitial: 華語文能力檢定
    ChineseQuestionDataInitial: CQDInitial
    EnglishQuestionDataInitial: EQDInitial
    
    --------------------------------------
"""
# def SanityCheckMethodTest(CQADataSet, model, tqn):

#     x = 0.1
#     l = ['1','2','3','4']
#     with open('test', 'w') as ww:
#         for cqn in range(tqn):
#             if len(CQADataSet[cqn].answer)<4:
#                 ans = str(random.randint(1,4))
#             else:
#                 ans = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x)
#                 if ans == 'A':
#                     ans = l[0]
#                 elif ans == 'B':
#                     ans = l[1]
#                 elif ans == 'C':
#                     ans = l[2]
#                 elif ans == 'D':
#                     ans = l[3]

#             ww.write(ans+"\n")

def SanityCheckMethod(CQADataSet, model, tqn):

    x = 0.1
    bestX = 0
    highestCorrectCount  = 0
    while x < 1:
        tempCorrectCount = 0
        for cqn in range(tqn):
            print(cqn)
            ans = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x)
            if ans == CQADataSet[cqn].correct_answer:
                tempCorrectCount += 1
        
        if highestCorrectCount < tempCorrectCount:
            highestCorrectCount = tempCorrectCount
            bestX = x
        print("bestX: %.2f" %(bestX))
        print("Accuracy: %.4f" %(highestCorrectCount/tqn))
        x += 0.1

    print("bestX: %.2f" %(bestX))
    print("Accuracy: %.4f" %(highestCorrectCount/tqn))
    return bestX, highestCorrectCount/tqn

def main():

    # initial setting
    data = 'MOST'
    
    # # contest mode
    # if m == 1:
    #     print("Start contest mode.\nInput parameters [1]qasp [2]mp [3]ap.")
    #     qasp = sys.argv[1]
    #     mp = sys.argv[2]
    #     tqn = len(listdir(qasp))
    #     print("Start create CQA instance and load model.")
    #     CQADataSet, model = ContestModeInitial(qasp, mp, tqn, data).getCQADataSetAndModel()
    #     SanityCheckMethod(CQADataSet, model, tqn)
    
    # development mode
    print("Start development mode.\nInput parameters [1]qasp [2]mp [3]ap.\n")
    qasp = sys.argv[1]
    mp = sys.argv[2]
    ap = sys.argv[3]
    tqn = len(listdir(qasp))
    # get all instance and load word2vec model
    print("Start create CQA instance and load model.\n")
    
    
    # start SanityCheckMethod iteration
    x, accuracy, bestX, bestAccuracy = 0, 0, 0, 0
    modelFiles = listdir(mp)
    bestModel = ""
    for m in modelFiles:
        if m[len(m)-6:len(m)] == ".model":
            modelPath = mp + m
            print("Process model: %s" %m)
            CQADataSet, model = DevelopmentModeInitial(qasp, modelPath, tqn, data, ap).getCQADataSetAndModel()
            x, accuracy = SanityCheckMethod(CQADataSet, model, tqn)
        if accuracy > bestAccuracy:
            bestX = x
            bestAccuracy = accuracy
            bestModel = m
    print(bestAccuracy)
    print(bestModel)
    
if __name__ == "__main__":
    main()
