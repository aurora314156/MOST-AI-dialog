import json
import logging
from Initial import Initial
from gensim import models
from CWTInitial import CWTInitial
from CQDInitial import CQDInitial


# read cqa data from json
class DevelopmentModeInitial(Initial):

    def __init__(self, qasp, mp, tqn, data):
        self.qasp = qasp
        self.mp = mp
        self.tqn = tqn
        self.data = data
        
    
    # create instance and append all instance to list
    def getCQADataSetAndModel(self):

        CQAInstanceList = []
        correctAnsList = [4,1,3,4,3,2,3,2,1,1,3,2,3,3,3,4,2,2,3,2,4,1,2,4,2,3,1,3,2,3,4,3,4,2,1,4,2,2,3,2,3,1,4,3,4,3,4,1,2,3,2,1,4,1,2,3,3,1,2,1,1,3,2,3,2,3,1,3,4,4,1,3,3,2,4,2,1,4,1,2,3,2,1,3,2,3,1,3,1,2,4,3,4,4,2,2,1,4,2,1,1,3,4,1,1,1,2,1,3,2,4,1,3,2,2,2,2,3,3,2,4,1,4,2,2,3,1,1,1,3,1,1,4,1,3,2,3,1,1,4,2,1,2,1,3,1,3,1,1,1,2,2,2,3,1,2,3,1,3,4,3,3,2,1,4,1,2,4,2,2,1,4,1,2,3,1,4,2,2,3,3,2,1,2,4,3,4,1,1,1,4,4,1,4,3,3,1,4,4,3,1,1,1,4,4,4,3,4,1,4,1,3,2,3,2,4,2,2,2,2,1,1,1,2,3,1,2,2,3,1,3,3,1,3,1,4,2,2,1,2,3,2,1,3,1,4,2,3,2,4,1,1,3,4,3,2,3,1,2,1,4,2,3,1,3,2,2,4,2,3,1,4,4,2,3,3,3,2,1,4,1,1,2,2,3,3,3,4,4,4,1,3,3,2,4,4,4,3,1,3,1,4,2,3,4,2,4,1,3,1,2,4,3,1,3,2,4,3,4,3,2,1,2,3,2,2,3,3,1,3,3,1,4,4,1,3,4,1,4,3,2,4,3,3,1,2,1,3,2,1,3,1,2,3,2,1,3,4,3,2,2,1,4,2,3,3,3,2,4,2,2,3,1,1,4,3,1,2,3,1,3,1,4,3,3,1,1,2,1,3,2,3,2,3,2,1,2,3,4,1,2,2,1,2,3,2,1,1,1,3,3,4,3,2,3,1,3,4,1,4,3,4,2,3,3,3,3,4,2,4,3,4,4,1,3,4,1,3,1,2,2,1,2,1,1,4,1,3,2,3,4,1,2,3,4,4,3,2,2,2,1,4,4,3,3,2,2,1,4,3,2,1,4,4,3,1,4,4,1,4,4,4,3,1,4,2,2,3,4,3,2,2,4,2,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,2,2,2,3,3,3,3,2,1,2,4,1,3,4,4,3,3,4,1,3,2,3,2,4,3,2,1,1,4,2,3,2,2,4,1,3,2,3,2,1,4,3,1,3,3,4,4,2,1,2,3,4,3,3,3,1,3,4,2,1,1,3,2,4,2,3,1,2,4,1,1,3,2,2,4,1,2,1,3,2,4,1,2,2,2,3,2,2,3,1,3,2,3,2,2,3,1,1,2,3,2,3,1,2,3,2,1,3,1,4,3,2,3,4,2,1,4,1,4,3,1,4,2,1,2,4,1,4,2,3,1,2,1,1,2,4,4,1,3,3,4,2,4,2,2,1,3,3,1,4,3,2,4,2,2,2,3,2,4,3,4,1,3,4,2,2,3,4,1,3,2,1,4,1,1,2,4,2,1,3,1,2,1,1,2,1,3,1,3,1,1,2,2,1,1,4,4,2,4,2,1,3,4,3,4,4,2,4,1,4,3,4,4,4,2,4,4,3,1,1,2,4,4,3,4,3,4,3,2,2,3,2,4,2,4,3,1,3,4,3,4,4,2,3,3,1,3,2,1,2,2,3,4,2,1,4,3,2,4,2,3,2,2,3,1,2,3,2,3,4,3,2,3,3,2,4,2,3,4,2,3,2,1,3,4,3,3,2,1,2,4,1,3,2,2,1,3,2,4,3,1,4,2,4,2,4,2,1,1,4,2,1,3,2,3,1,4,2,2,1,2,3,1,3,1,3,1,3,2,1,4,4,1,2,3,1,3,1,2,3,1,2,4,3,1,2,1,3,1,3,3,1,3,3,1,1,3,4,1,3,2,4,4,2,4,3,4,1,3,2,1,4,3,2,2,1,2,2,2,1,1,2,1,1,2,3,2,1,1,2,2,2,2,1,3,1,3,2,3,1,1,2,1,3,1,1,1,1,1,3,4,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,3,1,1,1,2,1,1,1,1,1,1,1,1,1,2,1,1,4,2,1,2,1,1,1,1,3,1,1,2,4,2,1,1,2,1,1,3,2,3,1,1,1,3,1,2,2,4,1,1,1,1,1,2,4,3,1,3,2,3,2,1,2,3,4,3,2,3,3,2,1,2,3,1,1,2,1,3,1,2,2,4,3,1,1,1,1,2,1,2,1,2,1,3,1,4,1,3,1,4,3,2,1,3,1,1,3,2,2,1,4,3,2,2,3,1,3,4,3,2,2,3,2,3,3,2,1,3,4,4,2,2,2,2,2,2,2,3,4,2,1,2,3,2,4,3,3,3,3,4,2,2,2,3,3,3,2,2,3,3,2,1,4,4,3,2,3,3,2,4,2,4,3,2,3,3,3,4,4,2,3,1,1,3,2,2,2,4,3,4,2,3,2,1,2,3,4,2,2,4,4,2,2,3,2,2,2,2,3,4,3,3,1,2,3,2,2,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,3,4,3,4,4,4,3,4,4,4,3,4,4,4,1,1,4,4,1,3,3,4,4,4,1,3,4,4,1,4,3,1,3,4,4,1,1,4,1,4,3,4,3,1,4,4,3,4,4,3,1,2,1,4,1,4,2,3,2,1,3,4,2,3,2,2,3,2,4,4,1,2,2,3,1,2,3,1,4,3,1,4,3,3,2,2,3,4,2,4,1,1,2,2,3,2,4,3,3,4,1,2,3,2,3,1,4,3,3,1,2,4,1,3,2,2,4,3,2,4,4,3,2,1,3,2,4,2,1,4,2,1,4,3,4,3,2,1,4,2,3,3,2,1,3,4,3,2,3,1,4,4,3,2,3,4,3,3,2,1,4,3,1,2,1,3,4,1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,2,2,1,1,3,1,2,2,2,1,2,3,4,2,2,1,1,2,2,3,1,1,1,4,1,3,3,2,2,1,2,4,3,4,3,2,4,1,1,3,1,4,2,4,3,1,3,1,1,4,2,3,2,1,3,2,3,4,3,1,4,4,2,2,4,3,1,2,1,1,2,4,1,3,1,4,2,2,1,2,2,3,3,2,4,2,2,1]      

        for q in range(self.tqn):
            CQAInstanceList.append(self.readCQAData(q, correctAnsList[q]))

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = models.Word2Vec.load(self.mp)
        
        return CQAInstanceList, model
    
    # get CQA Data from json
    def readCQAData(self, q, ans):
        fileName = self.qasp + '/' + str(q) + '.json'
        with open(fileName , 'r') as reader:
            jf = json.loads(reader.read())
            if self.data == 'cwt':
                return CWTInitial(jf)
            elif self.data == 'MOST':
                return CQDInitial(jf, ans)
