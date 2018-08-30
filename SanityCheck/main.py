
from os import listdir
import sys
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
    ChineseQuestionDataInitial: CQDInitial
    EnglishQuestionDataInitial: EQDInitial
    --------------------------------------
"""

def SanityCheckMethod(CQADataSet, model, tqn):

    x = 0.1
    for cqn in range(1):
        ans = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x)
        print(ans)

def main():

    qasp = sys.argv[1]
    mp = sys.argv[2]
    tqn = len(listdir(qasp))
    lang = 'ch'
    # get all instance and load word2vec model
    print("Start create CQA instance and load model.")
    CQADataSet, model = Initial(qasp, mp, tqn, lang).getCQADataSetAndModel()
    # start SanityCheckMethod
    SanityCheckMethod(CQADataSet, model, tqn)

# initial setting


# initial setting

if __name__ == "__main__":
    main()
