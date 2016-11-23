import numpy


def parseRawDataUtil(idxValStr):
    idxValPair = idxValStr[2:-1].split(':')
    return [int(idxValPair[0]), float(idxValPair[1])]

def parseRawData(rawData, d):
    labelVec = numpy.zeros(d + 1)
    labelVec[0] = float(rawData[0][2:-1])
    vecStr = rawData[1:]
    vecIdxVal = list(map(parseRawDataUtil, vecStr))
    for idxValPair in vecIdxVal:
        labelVec[idxValPair[0]] = idxValPair[1]
    return labelVec

def processLibSVMData(inputFileName, d):
    outputFileName = inputFileName + '.npy'
    
    listArrayStr = numpy.loadtxt(inputFileName, dtype='str')
    n = len(listArrayStr)
    mat = numpy.zeros((n, d+1))
    for i in range(n):
        rawData = listArrayStr[i]
        mat[i, :] = parseRawData(rawData, d)
    
    numpy.save(outputFileName, mat)
    return mat

if __name__ == '__main__':
    inputFileName = 'YearPredictionMSD.t'
    d = 90 # number of features
    processLibSVMData(inputFileName, d)