from CreateMethod import CreateMethod

class Initial:
    def __init__(self, qasp, mp, qn = 1):
        self.qasp = qasp
        self.mp = mp
        self.qn = qn
    def CreateSanityCheck(self):
        CreateSanityCheckFromMethod = CreateMethod().CreateSanityCheck()
        return CreateSanityCheckFromMethod
