
import math
import time
from gensim.models import word2vec
from CQDInitial import CQDInitial
from random import randint


class SanityCheck():

    def __init__(self, CQADataset, tqn, cqn):
        self.CQADataset = CQADataset
        self.tqn = tqn
        self.cqn = cqn

    def SanityCheckMain(self, model, x, idfTable):
        
        questionList = self.CQADataset[self.cqn].getQuestion()
        answerList = self.CQADataset[self.cqn].getAnswer()
        
        finalAns, flag = 0, 0
        highestScore = 0
        ans = [1,2,3,4]

        if len(answerList) == 4:
            for A in answerList:
                currentAnsScore = 0
                for q in questionList:
                    align = self.align(model, x, q, A)
                    currentAnsScore += math.log(((self.tqn - idfTable[q]+ 0.5) / ( idfTable[q]+0.5 ))) * align
                if highestScore <= currentAnsScore:
                    highestScore = currentAnsScore
                    finalAns = flag
                flag += 1
        else:
            finalAns = randint(0, 3)
            
        return ans[finalAns]

    # calculate IDF table
    def calIDF(self):

        idfTable = {}
        for t in range(self.tqn):
            questionData = self.CQADataset[t].getQuestion()
            for q in questionData:
                if q not in idfTable:
                    idfTable[q] = 1
                else:
                    idfTable[q] += 1
        
        return idfTable

    # calculate align
    def align(self, model, x, q, A):
        
        termScoreTable = self.similarTermScoreTable(model, q, A)
        # if similar terms bigger than one, add neg to align
        if len(termScoreTable) > 1:
            align = termScoreTable[0] + x * termScoreTable[1]
        else:
            align = termScoreTable[0]
        
        return align
    
    # calculate consine similarity table
    def similarTermScoreTable(self, model, q, A):

        similarTermScore = []
        for cutWord in A:
            # if word not in model append 0
            try:
                model.similarity(q, cutWord)
            except KeyError:
                similarTermScore.append(0)
                continue
            else:
                similarTermScore.append(model.similarity(q, cutWord))

        similarTermScore.sort(reverse=True)
        return similarTermScore