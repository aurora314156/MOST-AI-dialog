

class QuestionHandler():
    def __init__(self, questionData):
        self.qasp = questionData.qasp
        self.mp = questionData.mp
        self.qn = questionData.qn
        
    def CreateSanityCheck(self):
        raise Exception("I dont't know how to run sanity check.")

        
        