class Initial:
    def __init__(self, qasp, mp, qn = 1):
        self.qasp = qasp
        self.mp = mp
        self.qn = qn
    def CallCreateSanityCheck(self):
        CreateMethod().CreateSanityCheck()
        
class SanityCheck(Initial):
    def __init__(self, qasp, mp, qn):
        super().__init__(qasp, mp, qn)


class CreateMethod():
    def CreateSanityCheck(self):
        s = SanityCheck("none", "none", "none")
        print(s.qasp)

def main():

    qasp = "123"
    mp = "456"
    qn = 1500
    Initial(qasp, mp, qn).CallCreateSanityCheck()
    

if __name__ == "__main__":
    main()
