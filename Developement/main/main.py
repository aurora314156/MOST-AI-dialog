import sys, time, logging
sys.path.append('../')
from os import listdir
from DevelopmentModeInitial import DevelopmentModeInitial
from SanityCheck.SanityCheck import SanityCheck
from AttentionWithGRU.AttentionWithGRU import AttentionWithGRU

def AttentionMethod(CQADataSet, tqn):
    tqn = 1
    # model parameters
    gru_units, model_fit_epochs, hops = [50,100,200], [2,4], [2,4,6]

    for g in gru_units:
        for m in model_fit_epochs:
            for h in hops:
                correct_answer = [1,1,2,2,2,3,1,4,2,1,1,3,3,2,3,2,3,1,3,2,1,4,2,4,2,2,2,4,2,2,4,1,3,2,2,3,1,2,3,2,1,3,1,1,3,4,1,4,2,2,3,3,3,2,2,3,4,2,4,3,3,3,1,4,1,3,3,3,1,4,2,2,4,3,3,1,3,1,4,4,2,3,4,3,4,1,4,1,1,2,1,3,4,1,3,1,1,1,2,2,2,1,4,3,2,1,3,1,3,4,4,1,3,1,3,2,4,4,1,2,3,2,3,3,2,4,2,3,1,2,1,3,1,3,1,4,2,2,1,1,2,4,1,3,1,3,1,2,2,1,3,4,2,1,2,4,2,2,2,4,1,2,3,3,1,3,4,1,3,4,3,1,3,4,4,2,3,4,3,1,4,2,2,4,1,1,2,2,2,4,4,2,3,3,1,4,3,2,4,2,2,3,1,3,4,2,1,4,2,4,1,3,2,4,2,2,4,3,2,4,1,4,2,3,1,2,4,3,2,1,3,4,3,4,2,4,2,2,4,2,3,1,1,4,4,2,3,3,2,4,1,2,2,1,3,3,3,4,1,2,2,3,2,3,3,2,4,2,3,1,3,3,4,1,2,1,4,4,2,4,1,3,2,3,4,1,4,3,1,2,2,1,4,3,4,3,4,1,4,3,4,1,2,3,4,3,1,3,2,2,2,4,4,1,2,3,1,2,4,2,3,3,2,4,4,1,4,1,3,3,4,2,4,1,3,2,1,2,2,2,1,1,3,2,2,4,3,2,4,4,3,1,2,4,4,1,1,3,2,4,2,3,3,1,3,2,3,2,1,2,2,4,4,1,3,3,2,3,3,4,1,3,1,2,1,3,4,3,3,2,4,4,4,1,3,4,4,2,1,3,2,4,2,3,1,4,1,2,3,3,2,4,3,1,2,3,1,2,1,2,1,2,4,4,2,2,3,2,4,1,2,1,2,3,4,1,1,1,3,4,4,4,1,3,1,2,2,4,1,2,2,4,2,4,2,4,1,1,3,2,3,1,4,2,1,1,4,4,3,2,1,4,4,3,3,4,3,1,4,1,2,4,2,2,2,2,3,4,1,4,1,3,4,3,4,3,4,2,1,3,4,3,4,1,1,4,4,1,4,2,2,4,4,2,3,4,3,3,3,4,3,2,1,1,3,1,4,1,2,4,1,2,3,3,1,2,2,3,1,4,3,3,4,4,3,2,4,4,2,1,4,3,4,1,2,3,3,1,2,4,3,4,2,3,1,3,2,4,4,3,3,3,4,2,3,1,2,4,2,1,3,4,2,3,2,2,1,2,4,2,4,2,1,3,4,4,2,2,4,4,3,2,1,3,4,3,3,3,1,4,1,2,2,2,2,3,4,2,1,3,2,4,3,1,1,1,2,4,4,2,1,4,4,2,1,1,2,3,2,3,2,4,4,4,1,4,3,3,1,3,3,3,3,1,4,2,3,2,4,2,2,4,1,4,2,1,4,2,1,3,1,1,3,1,2,4,2,2,2,3,2,3,4,1,2,1,3,2,4,4,1,4,3,1,3,1,4,2,3,3,2,4,2,3,2,3,3,3,1,4,3,4,4,3,4,1,4,2,4,1,3,1,2,3,2,3,2,1,4,2,4,1,3,4,4,1,1,4,3,1,2,1,4,1,3,2,3,3,3,2,1,4,3,4,2,1,1,2,2,4,4,1,3,3,1,1,1,4,4,3,2,2,3,4,1,3,4,1,1,1,3,3,2,3,2,3,1,3,2,1,1,1,2,4,1,4,2,3,3,3,4,4,1,1,4,2,1,4,3,1,2,3,4,4,2,2,1,2,4,2,1,3,4,2,3,4,4,4,1,2,1,2,4,4,4,1,3,2,1,4,4,4,3,1,3,2,2,2,2,2,4,4,3,1,2,2,2,4,3,3,2,4,2,1,4,1,3,4,1,1,1,3,2,3,4,4,1,2,1,3,4,2,1,3,1,3,3,3,1,1,2,2,2,3,1,3,4,3,3,3,4,4,2,3,1,4,4,4,1,2,4,3,3,3,1,4,4,1,2,1,4,1,3,2,2,1,1,3,3,2,1,4,4,3,4,2,4,1,1,3,1,4,2,4,2,1,1,4,1,3,4,4,2,2,1,1,2,4,1,2,2,2,3,1,3,2,3,3,2,2,1,1,4,1,1,3,3,3,2,1,1,1,1,2,3,1,4,4,3,4,1,2,2,4,2,4,4,1,3,2,4,3,1,3,1,4,2,3,4,2,3,1,4,4,2,3,1,4,3,1,2,4,2,1,3,2,2,1,4,3,3,2,3,4,2,1,2,3,2,4,2,1,3,1,2,4,3,4,2,3,1,2,4,1,3,2,1,4,2,4,2,3,1,2,1,4,3,3,2,4,3,1,2,4,3,1,3,4,2,3,1,4,3,3,1,4,2,4,3,4,2,1,4,3,2,1,3,2,4,1,2,3,4,2,4,3,4,2,3,4,2,1,4,3,2,1,3,3,1,4,1,2,1,1,3,2,1,3,2,2,4,4,2,4,3,2,3,3,4,2,1,4,3,2,4,2,4,3,1,4,2,3,1,2,3,1,4,1,3,4,1,2,3,4,1,3,2,2,4,4,1,3,2,4,4,2,3,4,2,1,4,1,3,4,3,2,1,4,2,1,3,1,3,4,4,3,1,4,4,4,3,3,2,1,4,1,1,2,3,2,4,2,3,1,4,1,2,3,1,2,3,4,2,3,4,2,1,3,4,2,1,3,1,4,3,2,2,3,4,1,3,1,3,2,3,4,3,3,1,4,2,1,4,2,3,2,1,3,4,1,2,2,1,3,4,2,1,2,1,2,3,4,1,2,3,4,2,1,3,4,2,3,2,3,4,1,2,1,1,4,2,1,2,4,1,3,2,4,1,2,3,4,1,2,3,1,2,4,2,1,4,4,3,1,2,4,2,1,2,4,1,1,4,3,1,4,3,2,1,3,4,3,1,4,2,4,1,2,3,4,2,3,1,2,4,3,2,3,1,1,3,4,2,1,3,1,4,1,3,2,3,4,4,2,3,4,1,3,1,2,3,1,2,4,4,2,1,3,2,1,3,2,1,4,1,2,4,1,3,1,3,1,4,1,2,4,3,4,1,4,4,2,4,3,4,2,4,1,3,2,1,2,3,4,1,2,1,2,3,1,3,3,1,4,2,2,1,3,1,2,3,4,2,1,3,4,1,2,3,4,1,2,3,4,1,2,1,3,2,1,3,4,1,2,3,1,2,3,4,1,2,3,1,2,4,1,3,1,3,4,1,2,3,4,3,1,2,1,3,4,2,4,1,3,4,2,1,3,4,4,1,4,2,3,4,2,1,3,4,4]
                sTime = time.time()
                guessAnsList = AttentionWithGRU(CQADataSet, tqn, g, m, h).AttentionWithGRUMain()
                print("Total took: %.2fs" % (time.time()-sTime))

                guess_correct = 0
                for i in range(tqn):
                    if guessAnsList[i] == correct_answer[i]:
                        guess_correct += 1
                print("Final guess correctness: ", guess_correct)

                with open('log.txt', 'a') as log:
                    log.write("\n ******This epoch experiment result******\n")
                    log.write("Total cost time: %.2fs.\n" % (time.time()-sTime))
                    log.write("Correctness: %.2f", %(guess_correct/len(correct_answer)))
                    log.write("Gru_units: %s, model_fit_epochs: %s, hops: %s\n" % (g, m, h))

def SanityCheckMethod(CQADataSet, model, tqn):
    
    x = 0.1
    bestX = 0
    highestCorrectCount  = 0
    
    # initital idfTable
    idfTable = SanityCheck(CQADataSet, tqn, 0).calIDF()
    print(len(idfTable))
    while x < 1:
        tempCorrectCount = 0
        for cqn in range(tqn):
            ans = SanityCheck(CQADataSet, tqn, cqn).SanityCheckMain(model, x, idfTable)
            if ans == CQADataSet[cqn].correct_answer:
                tempCorrectCount += 1
        
        if highestCorrectCount < tempCorrectCount:
            highestCorrectCount = tempCorrectCount
            bestX = x
        
        x += 0.1

    return bestX, highestCorrectCount/tqn


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
    
    # start SanityCheckMethod iteration
    #x, accuracy, bestX, bestAccuracy = 0, 0, 0, 0
    
    modelFiles = listdir(mp)
    bestModel = ""
    for m in modelFiles:
        if m[len(m)-6:len(m)] == ".model":
            modelPath = mp + m
            print("W2V Model: %s" %m)
            CQADataSet, w2vmodel = DevelopmentModeInitial(qasp, modelPath, tqn, data).getCQADataSetAndModel()
            AttentionMethod(CQADataSet, tqn)
            #SanityCheckMethod(CQADataSet, w2vmodel, tqn)
            
    
if __name__ == "__main__":
    main()
