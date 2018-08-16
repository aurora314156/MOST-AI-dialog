

import json
from ReadCQA import ReadCQA
from CalDocFreq import CalDocFreq

class SanityCheck():
    def __init__(self, qasp, mp, qn=1):
        self.qasp = qasp
        self.mp = mp
        self.qn = qn
        
    def SanityCheckMain(self):
        docFreq = CalDocFreq(self.qn).CalDocFreqMain()
        print(docFreq)
    

    # def calIdf():
    #     return calIdf
    # def calAlign():
    #     return align
    # def getAnser():
    #     score=calIdf() * calAlign()
        
    # return score




# class CalIdf(SanityCheckMain):
#     def __init__(self, path, num, docFreq):
#         self.path = path
#         self.questionNum = num
#         self.docFreq = docFreq
    
#     def getTermFromQuestion(self):
#         fileName = self.path + str(self.questionNum) + '.json'
#         with open(fileName , 'r') as reader:
#             jf = json.loads(reader.read())
#         return jf

#     def calIdf(self):
#         jf = self.getTermFromQuestion()
#         for q in jf['question']:
#             if q in self.docFreq:
#                 print(q,sequestionWordslf.docFreq[q])
        


