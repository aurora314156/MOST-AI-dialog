import sys, time, logging, csv
sys.path.append('../')
from os import listdir
from keras import backend as K
from DevelopmentModeInitial import DevelopmentModeInitial
from SanityCheck.SanityCheck import SanityCheck
from AttentionWithGRU.AttentionWithGRU import AttentionWithGRU


def writeResToCSV(methodName, guessAnsList):
    outputList = []
    outputMerge = []
    #read question number from csv file
    with open('readNumber.csv', newline= '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        outputList = list(spamreader)

        for i in range(0, len(guessAnsList), 1):
            data = str(outputList[i+1][0])+ str(guessAnsList[i])
            outputMerge.append(data)

    with open(methodName+'.csv', 'w', newline='') as csvfile:
        csvfile.write('ID,Answer')
        csvfile.write('\n')
        for i in outputMerge:                     
            csvfile.write(i)
            csvfile.write('\n')

def AttentionMethod(CQADataSet, tqn):
    
    print("Processing AttentionMethod.")
    # model parameters
    g, m, h = 10, 2, 2
    sTime = time.time()
    # final answer list
    guessAnsList = []
    for i in range(tqn):    
        # clear tensorflow draw graph
        K.clear_session()
        print("Processing number: %s\n"%(i))
        print("******Parameters******\nGru_units: %s, model_fit_epochs: %s, hops: %s\n"%(g, m, h))
        # corpus content initial
        questionWordList = CQADataSet[i].getQuestion()
        storyWordList = CQADataSet[i].getCorpus()
        answerList = CQADataSet[i].getAnswer()
        guessAnswer = AttentionWithGRU(questionWordList, storyWordList, answerList, g, m, h).AttentionWithGRUMain()
        guessAnsList.append(guessAnswer)
    
    print("Total tooks: %.2fs" % (time.time()-sTime))
    writeResToCSV('AttentionMethod', guessAnsList)

def SanityCheckMethod(CQADataSet, model, tqn):

    print("Processing sanitycheckmethod.")
    sTime = time.time()
    x, guessAnsList, errorCount = 0.1, [], 0
    # initital idfTable
    idfTable = SanityCheck(CQADataSet, tqn, 0).calIDF()

    for cqn in range(tqn):
        ans,errCheck = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x, idfTable)
        guessAnsList.append(ans)
        if errCheck == True:
            errorCount+=1
    
    print("Total error format corpus: ", errorCount)
    print("Total tooks: %.2fs\n" % (time.time()-sTime))
    writeResToCSV('SanityCheck', guessAnsList)

def main(argv=None):
    # initial setting
    data = 'MOST'
    # development mode
    print("\nStart development mode.\n\nInput parameters [1]qasp [2]mp.\n")
    qasp = sys.argv[1]
    mp = sys.argv[2]
    tqn = len(listdir(qasp))
    # get all instance and load word2vec model
    print("Start create CQA instance and load model.\n")
    
    modelFiles = listdir(mp)
    for m in modelFiles:
        if m[len(m)-6:len(m)] == ".model":
            modelPath = mp + m
            print("W2V Model: %s" %m)
            CQADataSet, w2vmodel = DevelopmentModeInitial(qasp, modelPath, tqn, data).getCQADataSetAndModel()
            SanityCheckMethod(CQADataSet, w2vmodel, tqn)
            AttentionMethod(CQADataSet, tqn)
    
if __name__ == "__main__":
    main()
