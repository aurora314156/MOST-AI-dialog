

import math
from CQDInitial import CQDInitial


class SanityCheck():
    def __init__(self, CQADataset, tqn, cqn):
        self.CQADataset = CQADataset
        self.tqn = tqn
        self.cqn = cqn

    def SanityCheckMain(self):
        questionList = self.CQADataset[self.cqn].getQuestion()
        print(questionList)
        #IDFTable = calIDF()
        

    # def calIDF(self):
    #     idf = []
    #     N = self.tqn
    #     result = N - self.CQADataset[self.qn] + 0.5 

    # def docFreq(self):
    #     for q in range(self.qn):
    #         cqa = ReadCQA.readCQA(q)
    #         for questionWords in cqa['question']:
    #             if q not in self.docFreq:
    #                 self.docFreq[q] = 1
    #             else:
    #                 self.docFreq[q] +=1
        
    #     return self.docFreq




