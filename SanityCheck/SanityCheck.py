

import json
from CalDocFreq import CalDocFreq
from Initial import Initial


class SanityCheck(Initial):
    def __init__(self, qasp="none", mp="none", qn="none"):
        super().__init__(qasp, mp, qn)
        
    def SanityCheckMain(self):
        docFreq = CalDocFreq(self.qn).CalDocFreqMain()
        return docFreq
    

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
        


