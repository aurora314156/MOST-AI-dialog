

import logging
import sys
from Initial import Initial
from QuestionHandler import QuestionHandler
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

def sanityCheck(CQADataSet, tqn):

    for cqn in range(tqn):
        SanityCheck = SanityCheck(CQADataSet, tqn, cqn)


def main():

    qasp = sys.argv[1]
    mp = sys.argv[2]
    tqn = 1
    lang = 'ch'
    # get all instance
    CQADataSet = Initial(qasp, tqn, lang).getCQADataSet() 
    
    sanityCheck(CQADataSet, qn)

if __name__ == "__main__":
    main()
