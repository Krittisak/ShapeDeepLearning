import sys
import csv

def getPredictionAccuracies(data):
        val = -1.0
        numcorrect = -1
        total = -1
        if len(data) != 0:
            vecLen = len(data[0])
            dataLen = vecLen/2
            confidences = []
            actual = []
            for i in data:
                c = []
                a = []
                for j in range(0, dataLen):
                    c.append(i[j])
                    a.append(i[j+dataLen])
                confidences.append(c)
                actual.append(a)

            numCorrect = 0
            for i in range(0,len(confidences)):
                if confidences[i].index(max(confidences[i])) == actual[i].index(max(actual[i])):
                    numCorrect += 1

        val = (numCorrect*1.0)/(len(data))
        return numCorrect, len(data), val

with open(sys.argv[1], 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)
    data.pop(0)
    n, t, a = getPredictionAccuracies(data)
    print("Total predicted: " + str(t) + "\nNum Correct: " + str(n) + "\nAccuracy: " + str(a))



