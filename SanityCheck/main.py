
import sys
from Initial import Initial
from SanityCheck import SanityCheck

"""
    QuestionAnswerSetPath : qasp
    ModelPath : mp
    TotalQuestionNumber : tqn
    CurrentQuestionNumber : cqn
    language : lang
    ChineseQuestionDataInitial : CQDInitial
    EnglishQuestionDataInitial : EQDInitial
"""

def SanityCheckMethod(CQADataSet, model, tqn):

    x = 0.1
    for cqn in range(tqn):
        ans = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x)
        print(ans)

def main():

    qasp = sys.argv[1]
    mp = sys.argv[2]
    tqn = 100
    lang = 'ch'
    # get all instance
    CQADataSet, model = Initial(qasp, mp, tqn, lang).getCQADataSetAndModel()
    # start SanityCheckMethod
    SanityCheckMethod(CQADataSet, model, tqn)

if __name__ == "__main__":
    main()
