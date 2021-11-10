from math import log
import sys
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

class Node:
  """
  A simple node class to build our tree with. It has the following:

  children (dictionary<str,Node>): A mapping from attribute value to a child node
  attr (str): The name of the attribute this node classifies by.
  islead (boolean): whether this is a leaf. False.
  """

  def __init__(self,attr):
    self.children = {}
    self.attr = attr
    self.isleaf = False

class LeafNode(Node):
    """
    A basic extension of the Node class with just a value.

    value (str): Since this is a leaf node, a final value for the label.
    islead (boolean): whether this is a leaf. True.
    """
    def __init__(self,value):
        self.value = value
        self.isleaf = True

class Tree:
  """
  A generic tree implementation with which to implement decision tree learning.
  Stores the root Node and nothing more. A nice printing method is provided, and
  the function to classify values is left to fill in.
  """
  def __init__(self, root=None):
    self.root = root

  def prettyPrint(self):
    print str(self)

  def preorder(self,depth,node):
    if node is None:
      return '|---'*depth+str(None)+'\n'
    if node.isleaf:
      return '|---'*depth+str(node.value)+'\n'
    string = ''
    for val in node.children.keys():
      childStr = '|---'*depth
      childStr += '%s = %s'%(str(node.attr),str(val))
      string+=str(childStr)+"\n"+self.preorder(depth+1, node.children[val])
    return string

  def count(self,node=None):
    if node is None:
      node = self.root
    if node.isleaf:
      return 1
    count = 1
    for child in node.children.values():
      if child is not None:
        count+= self.count(child)
    return count

  def __str__(self):
    return self.preorder(0, self.root)

  def classify(self, classificationData):
    """
    Uses the classification tree with the passed in classificationData.`

    Args:
        classificationData (dictionary<string,string>): dictionary of attribute values
    Returns:
        str
        The classification made with this tree.
    """
    node = self.root
    while not node.isleaf:
        node = node.children[classificationData[node.attr]]
    return node.value


def getPertinentExamples(examples,attrName,attrValue):
    """
    Helper function to get a subset of a set of examples for a particular assignment
    of a single attribute. That is, this gets the list of examples that have the value
    attrValue for the attribute with the name attrName.

    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValue (str): a value of the attribute
    Returns:
        list<dictionary<str,str>>
        The new list of examples.
    """
    newExamples = []
    for example in examples:
        if (example[attrName] == attrValue):
            newExamples.append(example)
    return newExamples

def getClassCounts(examples,className):
    """
    Helper function to get a dictionary of counts of different class values
    in a set of examples. That is, this returns a dictionary where each key
    in the list corresponds to a possible value of the class and the value
    at that key corresponds to how many times that value of the class
    occurs.

    Args:
        examples (list<dictionary<str,str>>): list of examples
        className (str): the name of the class
    Returns:
        dictionary<string,int>
        This is a dictionary that for each value of the class has the count
        of that class value in the examples. That is, it maps the class value
        to its count.
    """
    classCounts = {}
    for example in examples:
        if (className in example):
            if (example[className] not in classCounts):
                classCounts[example[className]] = 1
            else:
                classCounts[example[className]] += 1
    return classCounts

def getMostCommonClass(examples,className):
    """
    A freebie function useful later in makeSubtrees. Gets the most common class
    in the examples. See parameters in getClassCounts.
    """
    counts = getClassCounts(examples,className)
    return max(counts, key=counts.get) if len(examples)>0 else None

def getAttributeCounts(examples,attrName,attrValues,className):
    """
    Helper function to get a dictionary of counts of different class values
    corresponding to every possible assignment of the passed in attribute.
      That is, this returns a dictionary of dictionaries, where each key
      corresponds to a possible value of the attribute named attrName and holds
      the counts of different class values for the subset of the examples
      that have that assignment of that attribute.

    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<str>): list of possible values for the attribute
        className (str): the name of the class
    Returns:
        dictionary<str,dictionary<str,int>>
        This is a dictionary that for each value of the attribute has a
        dictionary from class values to class counts, as in getClassCounts
    """

    attributeCounts = {}
    for attrValue in attrValues:
        attributeCounts[attrValue] = getClassCounts(getPertinentExamples(examples, attrName, attrValue), className)
    return attributeCounts


def setEntropy(classCounts):
    """
    Calculates the set entropy value for the given list of class counts.
    This is called H in the book. Note that our labels are not binary,
    so the equations in the book need to be modified accordingly. Note
    that H is written in terms of B, and B is written with the assumption
    of a binary value. B can easily be modified for a non binary class
    by writing it as a summation over a list of ratios, which is what
    you need to implement.

    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The set entropy score of this list of class value counts.
    """
    # H(D) = -sum(p(Yi)logP(Yi))
    # P(Yi) = #examples labeled Yi/ #examples
    entropy = 0.0
    for count in classCounts:
        prob = float(count)/sum(classCounts)
        entropy -= (prob) * log(prob, 2)
    return entropy


def remainder(examples,attrName,attrValues,className):
    """
    Calculates the remainder value for given attribute and set of examples.
    See the book for the meaning of the remainder in the context of info
    gain.

    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The remainder score of this value assignment of the attribute.
    """
    # Remainder(A) = sum((|Dai|/|D|)H(Dai))
    # Remainder(A) = sum((Pk + Nk)/(P + N)B(Pk/ Pk + Nk))
    remain = 0
    ncounts = list(getClassCounts(examples, className).values())
    for attrValue in attrValues:
        pertinentExamples = getPertinentExamples(examples, attrName, attrValue)
        kcounts = list(getClassCounts(pertinentExamples, className).values())
        remain += float(sum(kcounts))/sum(ncounts) * setEntropy(kcounts)
    return remain

def infoGain(examples,attrName,attrValues,className):
    """
    Calculates the info gain value for given attribute and set of examples.
    See the book for the equation - it's a combination of setEntropy and
    remainder (setEntropy replaces B as it is used in the book).

    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The gain score of this value assignment of the attribute.
    """
    return setEntropy(list(getClassCounts(examples, className).values())) - remainder(examples, attrName, attrValues, className)

def giniIndex(classCounts):
    """
    Calculates the gini value for the given list of class counts.
    See equation in instructions.

    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The gini score of this list of class value counts.
    """
    # 1 sum(P2j)
    pj2 = 0
    for count in classCounts:
        pj2 += (float(count)/sum(classCounts)) ** 2
    return (1 - pj2)

def giniGain(examples,attrName,attrValues,className):
    """
    Return the inverse of the giniD function described in the instructions.
    The inverse is returned so as to have the highest value correspond
    to the highest information gain as in entropyGain. If the sum is 0,
    return sys.maxint.

    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The summed gini index score of this list of class value counts.
    """
    # gini(S, cs) = n1/n(gini(S1)) + n2/n(gini(S2))
    # S -> attrValues
    # n -> len(examples)

    gain = 0
    for attrValue in attrValues:
        gain += float(len(getPertinentExamples(examples, attrName, attrValue)))/len(examples) * giniIndex(getClassCounts(getPertinentExamples(examples, attrName, attrValue), className).values())
    return sys.maxint if gain == 0 else 1 / gain

def makeTree(examples, attrValues,className,setScoreFunc,gainFunc):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.

    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Tree
        The classification tree for this set of examples
    """
    remainingAttributes=attrValues.keys()
    return Tree(makeSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc))

def makeSubtrees(remainingAttributes,examples,attributeValues,className,defaultLabel,setScoreFunc,gainFunc):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """
    # gainFunc takes in examples, attrName(str), attrValues(list<str>), className(str)
    # returns summed giniScore
    # scoreFunc takes in classCounts(list<int>)
    # returns giniScore
    if len(examples) == 0:
        return LeafNode(defaultLabel)
    cCounts = getClassCounts(examples, className)
    for count in cCounts:
        if cCounts[count] == len(examples):
            return LeafNode(examples[0][className])
    if len(remainingAttributes) == 0:
        return LeafNode(getMostCommonClass(examples, className))

    gain = float("-inf")
    attr = None
    for attribute in remainingAttributes:
        if gainFunc(examples, attribute, attributeValues[attribute], className) > gain:
            gain = gainFunc(examples, attribute, attributeValues[attribute], className)
            attr = attribute
    subtree_default = getMostCommonClass(examples, className)
    subtree_attributes = list(remainingAttributes)
    subtree_attributes.remove(attr)
    node = Node(attr)
    for attrVal in attributeValues[attr]:
        subset = getPertinentExamples(examples, attr, attrVal)
        subtree = makeSubtrees(subtree_attributes, subset, attributeValues, className, subtree_default, setScoreFunc, gainFunc)
        node.children[attrVal] = subtree
    return node

def makePrunedTree(examples, attrValues,className,setScoreFunc,gainFunc,q):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.

    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Tree
        The classification tree for this set of examples
    """
    remainingAttributes=attrValues.keys()
    return Tree(makePrunedSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc,q))

def makePrunedSubtrees(remainingAttributes,examples,attributeValues,className,defaultLabel,setScoreFunc,gainFunc,q):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie classEntropy or gini)
        gainFunc (func): the function to score gain of attributes (ie entropyGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """
    if len(examples) == 0:
        return LeafNode(defaultLabel)
    cCounts = getClassCounts(examples, className)
    for count in cCounts:
        if cCounts[count] == len(examples):
            return LeafNode(examples[0][className])
    if len(remainingAttributes) == 0:
        return LeafNode(getMostCommonClass(examples, className))

    gain = float("-inf")
    attr = None
    for attribute in remainingAttributes:
        if gainFunc(examples, attribute, attributeValues[attribute], className) > gain:
            gain = gainFunc(examples, attribute, attributeValues[attribute], className)
            attr = attribute


    chiDict = {}
    index = 0
    for key in getAttributeCounts(examples, attr, attributeValues[attr], className).keys():
        chiDict[key] = sum(getAttributeCounts(examples, attr, attributeValues[attr], className).values()[index].values())
        index += 1

    dev = 0
    for key in getAttributeCounts(examples, attr, attributeValues[attr], className).keys():
        for item in getAttributeCounts(examples, attr, attributeValues[attr], className)[key].keys():
            px = float(getAttributeCounts(examples, attr, attributeValues[attr], className)[key][item])
            pxhat = (getClassCounts(examples, className)[item] / float((len(examples)))) * chiDict[key]
            dev += + (px - pxhat) ** 2 / pxhat
    v = len(attributeValues[attr]) - 1

    if stats.chi2.sf(dev, v) > q:
        return LeafNode(getMostCommonClass(examples, className))

    subtree_default = getMostCommonClass(examples, className)
    subtree_attributes = list(remainingAttributes)
    subtree_attributes.remove(attr)
    node = Node(attr)
    for attrVal in attributeValues[attr]:
        subset = getPertinentExamples(examples, attr, attrVal)
        subtree = makePrunedSubtrees(subtree_attributes, subset, attributeValues, className, subtree_default, setScoreFunc, gainFunc, q)
        node.children[attrVal] = subtree
    return node

