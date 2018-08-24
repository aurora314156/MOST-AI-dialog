
import math

from gensim.models import word2vec
from CQDInitial import CQDInitial

class SanityCheck():
    def __init__(self, CQADataset, tqn, cqn):
        self.CQADataset = CQADataset
        self.tqn = tqn
        self.cqn = cqn

    def SanityCheckMain(self, model, x):
        
        questionList = self.CQADataset[self.cqn].getQuestion()
        answerList = self.CQADataset[self.cqn].getAnswer()
        
        finalAns, flag = 0, 0
        ans = ['A','B','C','D']
        for A in answerList:
            currentAnsScore = 0
            currentAnsHighScore = 0
            for q in questionList:
                currentAnsScore += self.calIDF(q) * self.align(model, x, q, A)
            if currentAnsHighScore < currentAnsScore:
                currentAnsHighScore = currentAnsScore
                finalAns = flag
            flag += 1

        return ans[finalAns]

    def calIDF(self, q):

        docFreq = 0
        for t in range(self.tqn):
            otherQuestionList = self.CQADataset[t].getQuestion()
            if q in otherQuestionList:
                docFreq +=1

        idf_qi = math.log( (self.tqn - docFreq + 0.5 ) / docFreq + 0.5 )
        return idf_qi
    
    def align(self, model, x, q, A):
        
        termScoreTable = self.similarTermScoreTable(model, q, A)
        # if similar terms bigger than one, add neg to align
        if len(termScoreTable) > 1:
            align = termScoreTable[0] + x * termScoreTable[1]
        else:
            align = termScoreTable[0]        
        return align
        
    def similarTermScoreTable(self, model, q, A):

        similarTermScore = []
        for cutWord in A:
            # if word not in model append 0
            try:
                c = model[cutWord]
                v = model[q]
            except KeyError as e:
                similarTermScore.append(0)
                continue
            
            similarTermScore.append(model.similarity(q, cutWord))
        similarTermScore = sorted(similarTermScore, reverse=True)
        return similarTermScore