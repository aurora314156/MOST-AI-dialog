
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
            HighestScore = 0
            for q in questionList:
                currentAnsScore += self.calIDF(q) * self.align(model, x, q, A)
            if HighestScore < currentAnsScore:
                HighestScore = currentAnsScore
                finalAns = flag
            print("currentAnsScore {score}".format(score=currentAnsScore))
            print("HighestScore {score}".format(score=HighestScore))
            flag += 1
            print()
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
            except KeyError as e:
                similarTermScore.append(0)
                continue
            else:
                similarTermScore.append(model.similarity(q, cutWord))
        
        similarTermScore = sorted(similarTermScore, reverse=True)
        return similarTermScore