import json

from CQDInitial import CQDInitial
from EQDInitial import EQDInitial

# read cqa data from json
class Initial:
    def __init__(self, qasp, tqn, lang):
        self.qasp = qasp
        self.tqn = tqn
        self.lang = lang

    # create instance and append all instance to list
    def getCQADataSet(self):

        CQAInstanceList = []
        for q in range(self.tqn):
            CQAInstanceList.append(readCQAData(q))

        return CQAInstanceList
    
    # get CQA Data from json
    def readCQAData(self, q):
        fileName = self.qasp + '/' + str(q) + '.json'
        with open(fileName , 'r') as reader:
            jf = json.loads(reader.read())
            if self.lang == 'ch':
                return CQDInitial(jf)
                
        