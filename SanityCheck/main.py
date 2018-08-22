

import logging
import sys
from Initial import Initial

"""
    QuestionAnswerSetPath : qasp
    ModelPath : mp
    QuestionNumber : qn
    language : lang
    ChineseQuestionDataInitial : CQDInitial
    EnglishQuestionDataInitial : EQDInitial
"""

def main():

    qasp = sys.argv[1]
    mp = sys.argv[2]
    qn = 1
    lang = 'eng'
    CQADataSet = Initial(qasp, mp, qn, lang).readCQAData()
    print(CQADataSet[0].getDataSetAttributes())

if __name__ == "__main__":
    main()
