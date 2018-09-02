
import sys
import random
from os import listdir
from Initial import Initial
from SanityCheck import SanityCheck

"""
    Parameters
    --------------------------------------
    QuestionAnswerSetPath: qasp
    ModelPath: mp
    TotalQuestionDataSetAmount: tqn
    CurrentQuestionNumber: cqn
    language: lang
    CWTInitial : 華語文能力檢定
    ChineseQuestionDataInitial: CQDInitial
    EnglishQuestionDataInitial: EQDInitial

    --------------------------------------
"""
def SanityCheckMethodTest(CQADataSet, model, tqn):

    x = 0.1
    l = ['1','2','3','4']
    with open('test', 'w') as ww:
        for cqn in range(tqn):
            if len(CQADataSet[cqn].answer)<4:
                ans = str(random.randint(1,4))
            else:
                ans = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x)
                if ans == 'A':
                    ans = l[0]
                elif ans == 'B':
                    ans = l[1]
                elif ans == 'C':
                    ans = l[2]
                elif ans == 'D':
                    ans = l[3]

            ww.write(ans+"\n")

def SanityCheckMethod(CQADataSet, model, tqn):

    x = 0.1
    bestX = 0
    highestCorrectCount  = 0
    while x < 1:
        tempCorrectCount = 0
        for cqn in range(tqn):
            ans = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x)
            if ans == CQADataSet[cqn].correct_answer[0]:
                tempCorrectCount += 1
        
        if highestCorrectCount < tempCorrectCount:
            highestCorrectCount = tempCorrectCount
            bestX = x
        print("bestX: %.2f" %(x))
        print("Accuracy: %.4f" %(highestCorrectCount/tqn))
        x += 0.1
        

def main():

    qasp = sys.argv[1]
    mp = sys.argv[2]
    tqn = len(listdir(qasp))
    lang = 'MOST'
    # get all instance and load word2vec model
    print("Start create CQA instance and load model.")
    CQADataSet, model = Initial(qasp, mp, tqn, lang).getCQADataSetAndModel()
    # start SanityCheckMethod
    SanityCheckMethodTest(CQADataSet, model, tqn)

# initial setting

# initial setting

if __name__ == "__main__":
    main()
