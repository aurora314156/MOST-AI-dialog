from QuestionDataSetInitial import QuestionDataSetInitial


class EQDInitial(QuestionDataSetInitial):
    def __init__(self, jf):
        self.corpus = jf['corpus']
        self.question = jf['question']
        self.answer = jf['answer']
    # get instance attributes
    def getCorpus(self):
        return self.corpus
    def getQuestion(self):
        return self.question
    def getAnswer(self):
        return self.answer


