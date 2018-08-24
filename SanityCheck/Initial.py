import json
import logging
from CQDInitial import CQDInitial
from EQDInitial import EQDInitial
from gensim import models

# read cqa data from json
class Initial:
    def __init__(self, qasp, mp, tqn, lang):
        self.qasp = qasp
        self.mp = mp
        self.tqn = tqn
        self.lang = lang
    # create instance and append all instance to list
    def getCQADataSetAndModel(self):

        CQAInstanceList = []
        for q in range(self.tqn):
            CQAInstanceList.append(self.readCQAData(q))

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = models.Word2Vec.load(self.mp)
        
        return CQAInstanceList, model
    
    # get CQA Data from json
    def readCQAData(self, q):
        fileName = self.qasp + '/' + str(q) + '.json'
        with open(fileName , 'r') as reader:
            jf = json.loads(reader.read())
            if self.lang == 'ch':
                return CQDInitial(jf)

        