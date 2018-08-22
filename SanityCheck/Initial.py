import json

from CQDInitial import CQDInitial
from EQDInitial import EQDInitial

# read cqa data from json
class Initial:
    def __init__(self, qasp, mp, qn, lang):
        self.qasp = qasp
        self.mp = mp
        self.qn = qn
        self.lang = lang

    def readCQAData(self):
        fileName = self.qasp + '/' + str(self.qn) + '.json'
        CQA = []
        with open(fileName , 'r') as reader:
            jf = json.loads(reader.read())
            if self.lang == 'eng':
                CQA.append(CQDInitial(jf))
            
        return CQA
