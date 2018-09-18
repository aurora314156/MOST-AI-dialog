
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

    def SanityCheckMain(self, model, x):
        
        questionList = self.CQADataset[self.cqn].getQuestion()
        answerList = self.CQADataset[self.cqn].getAnswer()
        
        finalAns, flag = 0, 0
        highestScore = 0
        ans = [1,2,3,4]
        if len(answerList) == 4:
            
            for A in answerList:
                currentAnsScore = 0
                for q in questionList:
                    cal = self.calIDF(q)
                    align = self.align(model, x, q, A)
                    currentAnsScore += cal * align
                if highestScore <= currentAnsScore:
                    highestScore = currentAnsScore
                    finalAns = flag
                flag += 1
        else:
            finalAns = randint(0, 3)
        return ans[finalAns]

    # calculate IDF
    def calIDF(self, q):

        docFreq = 0
        for t in range(self.tqn):
            otherQuestionList = self.CQADataset[t].getQuestion()
            if q in otherQuestionList:
                docFreq +=1

        idf_qi = math.log( (self.tqn - docFreq + 0.5 ) / docFreq + 0.5 )
        return idf_qi

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
        #similarTermScore = sorted(similarTermScore, reverse=True)
        return similarTermScore