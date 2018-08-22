
from ReadCQA import ReadCQA

# calculate all question term frequency
class CalDocFreq:
    def __init__(self, qn):
        self.docFreq = {}
        self.qn = qn

    def CalDocFreqMain(self):
        for q in range(self.qn):
            cqa = ReadCQA.readCQA(q)
            for questionWords in cqa['question']:
                if q not in self.docFreq:
                    self.docFreq[q] = 1
                else:
                    self.docFreq[q] +=1
        
        return self.docFreq
    