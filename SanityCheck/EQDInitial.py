from QuestionDataSetInitial import QuestionDataSetInitial


class EQDInitial(QuestionDataSetInitial):
    def __init__(self, jf):
        self.corpus = jf['corpus']
        self.question = jf['question']
        self.answer = jf['answer']
    def getDataSetAttributes():
        print(self.corpus)
        print(self.question)
        print(self.answer)




