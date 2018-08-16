

import logging
import sys
#from Initial import Initial
from SanityCheck import SanityCheck

"""
    QuestionAnswerSetPath : qasp
    QuestionNumber : qn
    ModelPath : mp
"""

def main():

    qasp = sys.argv[1]
    mp = sys.argv[2]
    qn = 1500
    s = SanityCheck(qasp,mp,qn)
    s.SanityCheckMain()
    

if __name__ == "__main__":
    main()
