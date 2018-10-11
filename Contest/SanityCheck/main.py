
import sys
import csv
import random
import logging
from os import listdir
from ContestModeInitial import ContestModeInitial
from SanityCheck import SanityCheck

"""
    Parameters
    --------------------------------------
    QuestionAnswerSetPath: qasp
    ModelPath: mp
    TotalQuestionDataSetAmount: tqn
    CurrentQuestionNumber: cqn
    mode [different mode]: cwt, MOST ..... 
    CWTInitial: 華語文能力檢定
    ChineseQuestionDataInitial: CQDInitial
    EnglishQuestionDataInitial: EQDInitial
    
    --------------------------------------
"""

def writeAnsToCsv(ansList):
    outputList = []
    outputMerge = []
     #read question number from csv file
    with open('readNumber.csv', newline= '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        outputList = list(spamreader)

        for i in range(0, len(ansList), 1):
            data = str(outputList[i+1][0])+ str(ansList[i])
            outputMerge.append(data)

    with open('SanityCheckMethod.csv', 'w', newline='') as csvfile:
        csvfile.write('ID,Answer')
        csvfile.write('\n')
        for i in outputMerge:                     
            csvfile.write(i)    
            csvfile.write('\n')


def SanityCheckMethod(CQADataSet, model, tqn):

    # weight for negative information
    x = 0.1
    ansList = []
    # initital idfTable
    print("Start calculation idf Table.")
    idfTable = SanityCheck(CQADataSet, tqn, 0).calIDF()
    print(len(idfTable))
    print("Start execution Sanity check method.")
    for cqn in range(tqn):
        #print("process quesiton: {0}.".format(cqn))
        ans = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x, idfTable)
        ansList.append(ans)

    writeAnsToCsv(ansList)
    print("Mission complete.")
    
def main():

    # initial setting
    data = 'MOST'
    print("Start contest mode.\nInput parameters [1]qasp [2]mp.\n")
    qasp = sys.argv[1]
    mp = sys.argv[2]
    tqn = len(listdir(qasp))

    # get all instance and load word2vec model
    print("Start create CQA instance and load model.\n")
    # start SanityCheckMethod iteration
    modelFiles = listdir(mp)
    for m in modelFiles:
        if m[len(m)-6:len(m)] == ".model":
            modelPath = mp + m
            CQADataSet, model = ContestModeInitial(qasp, modelPath, tqn, data).getCQADataSetAndModel()
    
    SanityCheckMethod(CQADataSet, model, tqn)

if __name__ == "__main__":
    main()
