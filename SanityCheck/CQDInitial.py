from QuestionDataSetInitial import QuestionDataSetInitial



class CQDInitial(QuestionDataSetInitial):
    def __init__(self, jf):
        self.corpus = jf['corpus']
        self.question = jf['question']
        self.answer = jf['answer']
    def getDataSetAttributes(self):
        print(self.corpus)
        print(self.question)
        print(self.answer)




