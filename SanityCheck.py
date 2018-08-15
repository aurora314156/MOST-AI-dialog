
import math
import json
import sys


class SanityCheckMain:
    def __init__(self, C , Q , A):
        self.C = C
        self.Q = Q
        self.A = A


class CalIdf():
    def __init__(self, path, num, docFreq):
        self.path = path
        self.questionNum = num
        self.docFreq = docFreq
    
    def getTermFromQuestion(self):
        fileName = self.path + str(self.questionNum) + '.json'
        with open(fileName , 'r') as reader:
            jf = json.loads(reader.read())
        return jf

    def calIdf(self):
        jf = self.getTermFromQuestion()
        for q in jf['question']:
            if q in self.docFreq:
                print(q,self.docFreq[q])
        


# for cal term frequency from question
class CalDocFreq():
    def __init__(self, path):
        self.docFreq = {}
        self.path = path

    def calTermFreq(self):
        for i in range(1500):
            fileName = self.path + str(i) + '.json'
            with open(fileName , 'r') as reader:
                jf = json.loads(reader.read())
            for q in jf['question']:
                if q not in self.docFreq:
                    self.docFreq[q] = 1
                else:
                    self.docFreq[q] +=1

        return self.docFreq



def main():

    QuestionAnswerSetPath = sys.argv[1]
    temp = CalDocFreq(QuestionAnswerSetPath).calTermFreq()
    CalIdf(QuestionAnswerSetPath, 10, temp).calIdf()
    

if __name__ == "__main__":
    main()
