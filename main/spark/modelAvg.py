from __future__ import print_function

import sys
from pyspark.sql import SparkSession

import numpy
import time
import os

# ================ Read raw data from text file ================ #

def idxval2vec(idx, val, d):
    """
    Convert d-dimensional sparse vector to a dense vector.
    The sparse vector is indexed by a list of indices and a list of values.
    Input
        idx: list of int
        val: list of float
        d: int
    Output
        vec: NumPy Array
    """
    vec = numpy.zeros(d)
    vec[idx] = val
    return vec



def parseRawDataSpark(filename, d, numPartitions=0):
    """
    Read data from a file and parse the raw data into an RDD of (label vector)-pairs
    Input:
        filename: String;
            each line in the file is a (label, vector)-pair organized as follows:
                2001  1:49.943569 4:8.748610 5:-17.406281 6:-13.099050 17:408.984863 36:-0.369940
        d: Int, the dimension of the vector
    Output:
        rddLabelVector: RDD of (Float, d-dim NumPy Array)
    """
    # file to strings
    if numPartitions == 0:
        rddRawData = sc.textFile(filename)
    #elif numPartitions < 20:
    #    rddRawData = sc.textFile(filename).repartition(numPartitions)
    else:
        rddRawData = sc.textFile(filename, minPartitions=numPartitions)
        
    rddSplitedData = rddRawData.map(lambda line: line.split())
    # parse the strings to RDDs of (label, vector)
    rddLabelVector = (rddSplitedData.map(lambda arr: (float(arr[0]), [a.split(':') for a in arr[1:]]))
                 .map(lambda arr: (arr[0], [int(a[0])-1 for a in arr[1]], [float(a[1]) for a in arr[1]]))
                 .map(lambda arr: (arr[0], idxval2vec(arr[1], arr[2], d)))
                 )
    return rddLabelVector

# =================== Process the raw data =================== #

def randPartition(n, k):
    '''
    Uniformly partition the set {0, 1, ..., n-1} into k disjoint sets.
    Input
        n: Int
        k: Int
    Output
        idxPartition: List, whose elements are Lists of Int
    '''
    randperm = list(numpy.random.permutation(n))
    nBegin = 0
    nStep = int(numpy.ceil(n / k))
    idxPartition = list()
    for i in range(k):
        nEnd = min(nBegin + nStep, n)
        idx = randperm[nBegin: nEnd]
        idxPartition.append(idx)
        nBegin = nEnd
    return idxPartition


def list2mat(listLabelVec, k):
    '''
    Input
        listLabelVec: List of (y_i, x_i),
            where y_i is float and x_i is d-dimensional NumPy Array
        k: integer
    Output
        vecY: n-by-1 NumPy Array, the stack of y_0, y_1, ..., y_{n-1}
        matX: n-by-d NumPy Array, the stack of y_0, y_1, ..., y_{n-1}
        idxPartition: List, whose elements are Lists of Int;
            it is the random partition of {0, 1, ..., n-1} into k disjoint sets
    '''
    n = len(listLabelVec)
    d = len(listLabelVec[0][1])
    # concatinate the (x_i, y_i) pairs to form X and y
    vecY = numpy.zeros((n, 1))
    matX = numpy.zeros((n, d))
    for i in range(n):
        labelVec = listLabelVec[i]
        vecY[i, :] = labelVec[0]
        matX[i, :] = labelVec[1]
    # random partition of {0, 1, ..., n-1} into k disjoint sets
    idxPartition = randPartition(n, k)
    
    return vecY, matX, idxPartition


def meanYmaxXSpark(rddLabelVector):
    """
    Input:
        rddLabelVector: RDD of (Float, d-dim NumPy Array), which is (label, vector)-pair
    Output:
        meanY: the average of the labels
        maxX: the entry-wise max of abs(X)
        n: the total number of the (label, vector)-pairs
    """
    n = rddLabelVector.count()
    # preprocess the data
    sumY, maxX = (rddLabelVector.map(lambda a: (a[0], numpy.abs(a[1])))
                 .reduce(lambda a, b: (a[0]+b[0], numpy.maximum(a[1], b[1])))
                  )
    meanY = sumY / rddLabelVector.count()
    return meanY, maxX, n



def preprocessSpark(rddLabelVector, foldOfCrossValid, meanY, maxX):
    """
    Normalize and concatinate the data.
    Input
        rddLabelVector: RDD of (Float, d-dim NumPy Array)
        foldOfCrossValid: Int
        meanY: Float
        maxX: 1-by-d Array
    Output
        rddProcessed: RDD of (labels, data matrix, disjoint sets)
            labels: n-by-1 NumPy Array
            data matrix: n-by-d NumPy Array
            disjoint sets: List whose entries are Lists of Int
    """
    # normalize X and Y
    broadcastMeanY = sc.broadcast(meanY)
    broadcastMaxX = sc.broadcast(maxX.reshape(len(maxX)))
    rddProcessed = rddLabelVector.map(lambda a: (a[0]-broadcastMeanY.value, a[1] / broadcastMaxX.value))
    # locally concatinate the (x_i, y_i) pairs to form X and y
    rddProcessed = rddProcessed.glom().map(lambda l: list2mat(l, foldOfCrossValid))
    
    return rddProcessed
    

# ================ The training of ridge regression ================ #
def trainRR(matX, matY, vecGamma):
    '''
    Solve: argmin_W  1/n * || X W - Y ||_F^2 + gamma * || W ||_F^2
    Input
        matX: n-by-d NumPy Array
        matY: n-by-1 NumPy Array
        vecGamma: List, containing different values of gamma
    Output
        models: d-by-l NumPy Array,
            where l is the lenth of vecGamma;
            each column of models is a solution W to the optimization problem
    '''
    lenGamma = len(vecGamma)
    n, d = matX.shape
    models = numpy.zeros((d, lenGamma))
    matU, vecS, matV = numpy.linalg.svd(matX, full_matrices=False)
    matUY = numpy.dot(matU.T, matY)
    for i in range(lenGamma):
        gamma = vecGamma[i]
        vec1 = vecS + n * gamma / vecS
        matW = matUY / vec1.reshape(d, 1)
        models[:, i] = numpy.dot(matV.T, matW).reshape(d)
    return models



def kFoldTrain(matX, matY, idxPartition, vecGamma):
    '''
    The training stage of the k-fold cross-validation.
    Partition (X, Y) into k parts according to the List idxPartition.
    Each time train the models on k-1 parts (leave one for validation).
    
    Input
        matX: n-by-d NumPy Array
        matY: n-by-1 NumPy Array
        idxPartition: List, whose elements are Lists of Int;
            it is the random partition of {0, 1, ..., n-1} into k disjoint sets
        vecGamma: List, containing different values of gamma
    Output
        listModels: List (length k) of models;
            models is d-by-l NumPy Array, where l is the lenth of vecGamma.
    '''
    k = len(idxPartition)
    listModels = list()
    
    for i in range(k):
        idxTrain = []
        for j in range(k):
            if j != i:
                idxTrain = idxTrain + idxPartition[j]
        matXtrain = matX[idxTrain, :]
        matYtrain = matY[idxTrain, :]
        models = trainRR(matXtrain, matYtrain, vecGamma)
        listModels.append(models)
    return listModels




def kFoldValid(matX, matY, idxPartition, listModels):
    '''
    The validation stage of the k-fold cross-validation.
    
    Input
        matX: n-by-d NumPy Array
        matY: n-by-1 NumPy Array
        idxPartition: List, whose elements are Lists of Int;
            it is the random partition of {0, 1, ..., n-1} into k disjoint sets
        listModels: List (length k) of models;
            models is d-by-l NumPy Array, where l is the lenth of vecGamma.
    Output
        squaredError: l-dimensional NumPy Array of the squared validation errors;
            each element of squaredError corresponds to one gamma. 
    '''
    k = len(idxPartition)
    lenGamma = listModels[0].shape[1]
    squaredError = numpy.zeros(lenGamma)
    for i in range(k):
        idxTest = idxPartition[i]
        matResidual = numpy.dot(matX[idxTest, :], listModels[i]) - matY[idxTest, :]
        err = numpy.sum(matResidual ** 2, axis=0)
        squaredError += err
    return squaredError
        


def kFoldCVSpark(rddTrain, vecGamma):
    '''
    The whole procedure of the k-fold cross-validation.
    
    Input
        rddTrain: RDD of (labels, data matrix, disjoint sets)
            labels: n-by-1 NumPy Array
            data matrix: n-by-d NumPy Array
            disjoint sets: List whose entries are Lists of Int
        vecGamma: List, containing different values of gamma
    Output
        listModels: List (length k) of models;
            models is d-by-l NumPy Array, where l is the lenth of vecGamma
        squaredError: l-dimensional NumPy Array of the squared validation errors;
            each element of squaredError corresponds to one gamma. 
    '''
    listModels = (rddTrain.map(lambda a: kFoldTrain(a[1], a[0], a[2], vecGamma))
              .reduce(lambda a, b: [a[i] + b[i] for i in range(len(a))])
             )
    g = rddTrain.count()
    listModels = [matW/g for matW in listModels]
    broadcastListModels = sc.broadcast(listModels)
    squaredError = (rddTrain.map(lambda a: kFoldValid(a[1], a[0], a[2], broadcastListModels.value))
                    .reduce(lambda a, b: a+b)
                   )
    return listModels, squaredError
    



def trainSpark(rddTrain, gamma):
    '''
    The training stage of the ridge regression.
    
    Input
        rddTrain: RDD of (labels, data matrix, disjoint sets)
            labels: n-by-1 NumPy Array
            data matrix: n-by-d NumPy Array
            disjoint sets: List whose entries are Lists of Int
        gamma: Int
    Output
        model: d-by-1 NumPy Array
    '''
    model = (rddTrain.map(lambda a: trainRR(a[1], a[0], [gamma]))
             .reduce(lambda a, b: a+b)
            )
    g = rddTrain.count()
    return model / g




def predictSpark(rddTest, model):
    '''
    The test stage of the ridge regression.
    
    Input
        rddTest: RDD of (labels, data matrix, disjoint sets)
            labels: n-by-1 NumPy Array
            data matrix: n-by-d NumPy Array
            disjoint sets: List whose entries are Lists of Int
        model: d-by-1 NumPy Array
    Output
        squaredError: Float
    '''
    broadcastModel = sc.broadcast(model)
    squaredError = (rddTest.map(lambda a: numpy.linalg.norm(a[0] - numpy.dot(a[1], broadcastModel.value)) ** 2)
                   .reduce(lambda a, b: a+b))
    return squaredError
    



def runModelAvgSpark(rddLabelVectorTrain, rddLabelVectorTest, param):
    '''
    Solve the ridge regression by model averaging.
    The pipeline:
        1. read, parse, and process the data
        2. use cross-validation to determine the regularization parameter gamma
        3. train the model using the optimal gamma
        4. predict the labels of the test data
    Input
        rddLabelVectorTrain, rddLabelVectorTest: RDDs of (Float, d-dim NumPy Array)
        param: Dictionary of parameters
    Output
        result: a dictionary of the mean squared errors and the elapsed time
    '''
    # normalization and other pre-process of the training data
    print('Processing the data...')
    meanY, maxX, nTrain = meanYmaxXSpark(rddLabelVectorTrain)
    rddTrain = preprocessSpark(rddLabelVectorTrain, param['foldOfCrossValid'], meanY, maxX).cache()
    rddTrain.count() # avoid lazy evaluation
    
    # run k-fold cross-validation to find the best gamma
    print(str(param['foldOfCrossValid']) + '-fold cross-validation...')
    t0 = time.time()
    listModels, squaredError = kFoldCVSpark(rddTrain, param['vecGamma'])
    t1 = time.time()
    mseCV = squaredError / nTrain
    print('The validation errors are ')
    print(mseCV)
    idxOpt = numpy.argmin(squaredError)
    gammaOpt = param['vecGamma'][idxOpt]
    
    # train the model using the optimal gamma
    print("Training the model using the best gamma...")
    t2 = time.time()
    modelOpt = trainSpark(rddTrain, gammaOpt)
    t3 = time.time()
    mseTrain = predictSpark(rddTrain, modelOpt) / nTrain
    print('The training error is ' + str(mseTrain))
    
    # normalization and other pre-process of the test data
    print("Testing...")
    nTest = rddLabelVectorTest.count()
    rddTest = preprocessSpark(rddLabelVectorTest, param['foldOfCrossValid'], meanY, maxX).cache()
    rddTest.count() # avoid lazy evaluation
    
    # predict the labels of the test data
    t4 = time.time()
    mseTest = predictSpark(rddTest, modelOpt) / nTest
    t5 = time.time()
    print('The test error is ' + str(mseTest))
    
    result = {'mseCV': mseCV,
              'mseTrain': mseTrain,
              'mseTest': mseTest,
              'timeCV': t1-t0,
              'timeTrain': t3-t2,
              'timeTest': t5-t4
             }
    return result


def parseConfig(inputDir, configFile):
    listStr = numpy.loadtxt(inputDir + configFile, dtype='str', comments='#')
    dirConfig = {"numExecutors": os.environ["NUM_EXECUTORS"],
                 "executorCores": int(listStr[0][2:-1]),
                 "driverMemory": listStr[1][2:-1],
                 "executorMemory": listStr[2][2:-1],
                 "trainFile": listStr[3][2:-1],
                 "testFile": listStr[4][2:-1]
                }
    return dirConfig


if __name__ == "__main__":
    inputDir = os.environ['INPUT_DIR']
    configFile = "/main/spark/config.txt"
    dirConfig = parseConfig(inputDir, configFile)
    
    # set Spark context
    from pyspark import SparkConf, SparkContext
    
    conf = (SparkConf()
             .setAppName("PySparkModelAvg")
             .set("spark.driver.memory", dirConfig["driverMemory"])
             .set("spark.executor.memory", dirConfig["executorMemory"])
             .set("spark.executor.cores", dirConfig["executorCores"])
             #.set("spark.executor.instances", dirConfig["numExecutors"])
           )
    sc = SparkContext(conf = conf)
    sc.setLogLevel("WARN")
    
    # set the parameter
    inputDir = os.environ['INPUT_DIR']
    param = {'d': 90,
             'foldOfCrossValid': 5,
             'vecGamma': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
             'trainFileName': inputDir + dirConfig["trainFile"],
             'testFileName': inputDir + dirConfig["testFile"],
             'numPartitions': int(dirConfig["numExecutors"])
            }
    #print(sc._conf.getAll())
    
    
    print('# ================= Reading Data from File ================= #')
    # read and parse the training data
    rddLabelVectorTrain = parseRawDataSpark(param['trainFileName'], param['d'], param['numPartitions']).cache()
    # read and parse the test data
    rddLabelVectorTest = parseRawDataSpark(param['testFileName'], param['d']).cache()
    
    numPartitions = rddLabelVectorTrain.getNumPartitions()
    print('The number of partitions is ' + str(numPartitions))
    rddLabelVectorTrain.count()
    rddLabelVectorTest.count()

    print('# ===================== Train and Test ===================== #')
    # run the whole procedure
    result = runModelAvgSpark(rddLabelVectorTrain, rddLabelVectorTest, param)
    
    print('# ======================== Results ======================== #')
    print('Time of cross-validation: ' + str(result['timeCV']))
    print('Time of training: ' + str(result['timeTrain']))
    print('Time of test: ' + str(result['timeTest']))
    print('Test MSE: ' + str(result['mseTest']))
    
    
    sc.stop()