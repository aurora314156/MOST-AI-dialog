from QuestionDataSetInitial import QuestionDataSetInitial


class CWTInitial(QuestionDataSetInitial):
    def __init__(self, jf):
        self.corpus = jf['corpus']
        self.question = jf['question']
        self.answer = jf['answer']
    #    self.correct_answer = jf['correct_answer']
        
    # get instance attributes
    def getCorpus(self):
        return self.corpus
    def getQuestion(self):
        return self.question
    def getAnswer(self):
        return self.answer
    #def getCorrectAnswer(self):
    #    return self.correct_answer


