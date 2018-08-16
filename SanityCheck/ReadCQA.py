

import json

# read cqa data from json
class ReadCQA():
    def __init__(self, qasp, qn):
        self.qn = qn
    def readCQA(self):
        fileName = self.qasp + str(self.qn) + '.json'
        with open(fileName , 'r') as reader:
            jf = json.loads(reader.read())
        return jf