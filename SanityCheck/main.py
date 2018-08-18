

import logging
import sys
from Initial import Initial


"""
    QuestionAnswerSetPath : qasp
    ModelPath : mp
    QuestionNumber : qn
"""

def main():

    qasp = sys.argv[1]
    mp = sys.argv[2]
    qn = 1500
    SanityCheck = Initial(qasp, mp, qn)
    print(SanityCheck)

if __name__ == "__main__":
    main()
