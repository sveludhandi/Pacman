from Testing import *
from DataInterface import*
from DecisionTree import*
def testNursery(setFunc = setEntropy, infoFunc = infoGain):
    """Correct classification averate rate is about 0.95"""
    examples,attrValues,labelName,labelValues = getExtraCreditDataset()
    print 'Testing Nursery dataset. Number of examples %d.'%len(examples)
    tree = makeTree(examples, attrValues, labelName, setFunc, infoFunc)
    f = open('nursery.out','w')
    f.write(str(tree))
    f.close()
    print 'Tree size: %d.\n'%tree.count()
    print 'Entire tree written out to nursery.out in local directory\n'
    dataset = getExtraCreditDataset()
    evaluation = getAverageClassificaionRate((examples,attrValues,labelName,labelValues))
    print 'Results for training set:\n%s\n'%str(evaluation)
    printDemarcation()
    return (tree,evaluation)
testNursery()
