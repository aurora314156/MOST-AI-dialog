

class Initial:
    def __init__(self, qasp, mp, qn = 1):
        super.__init__(qasp,mp,qn)


class SanityCheck(Initial):
    def __init__(self, qasp, mp, qn):
        super().__init__(qasp, mp, qn)
    

    def main():
        docFreq = CalDocFreq(self.qn).main()
        print(docFreq)


def main():

    qasp = sys.argv[1]
    mp = sys.argv[2]
    qn = 1500
    Initial(qasp, mp, 1)
    b = SanityCheck(None,None,None)
    b.main()
    


if __name__ == "__main__":
    main()
