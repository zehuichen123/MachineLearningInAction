#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from math import log
import matplotlib.pyplot as plt

decisionNode=dict(boxstyle='sawtooth',fc='0.8')
leafNode=dict(boxstyle='round4',fc='0.8')
arrow_args=dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def calcShannonEnt(dataSet):
	numEntries=len(dataSet)
	labelCounts={}
	for featVec in dataSet:
		currentLabel=featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	shannonEnt=0.0
	for key in labelCounts:
		prob=float(labelCounts[key])/numEntries
		shannonEnt-=prob*log(prob,2)
	return shannonEnt

def createDataSet():
	dataSet=[[1,1,'yes'],
			[1,1,'yes'],
			[1,0,'no'],
			[0,1,'no'],
			[0,1,'no']]
	labels=['no surfacing','flippers']
	return dataSet,labels

def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if featVec[axis]==value:
			reducedFeatVec=featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1
	baseEntrory=calcShannonEnt(dataSet)
	bestInfoGain=0.0
	bestFeature=-1
	for i in range(numFeatures):
		featureList=[example[i] for example in dataSet]
		uniqueVals=set(featureList)
		newEntrory=0.0
		for value in uniqueVals:
			subDataSet=splitDataSet(dataSet,i,value)
			prob=len(subDataSet)/float(len(dataSet))
			newEntrory+=(prob*calcShannonEnt(subDataSet))
		infoGain=baseEntrory-newEntrory
		if infoGain>bestInfoGain:
			bestInfoGain=infoGain
			bestFeature=i
	return bestFeature

def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classList:
			classCount[vote]=0
		classCount[vote]+=1
	sortedClassCount=sorted(classCount.items(),
		key=itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def createTree(dataSet,labels):
	classList=[example[-1] for example in dataSet]
	if classList.count(classList[0])==len(classList):
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)
	bestFeature=chooseBestFeatureToSplit(dataSet)
	bestFeatureLabel=labels[bestFeature]
	myTree={bestFeatureLabel:{}}
	del(labels[bestFeature])
	featValues=[example[bestFeature] for example in dataSet]
	uniqueVals=set(featValues)
	for value in uniqueVals:
		subLabels=labels[:]
		myTree[bestFeatureLabel][value]=createTree(
			splitDataSet(dataSet,bestFeature,value),subLabels)
	return myTree

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeaf(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalw, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalw
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeaf(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def getNumLeaf(myTree):
	numLeafs=0
	firstStr=list(myTree.keys())[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			numLeafs+=getNumLeaf(secondDict[key])
		else:
			numLeafs+=1
	return numLeafs

def getTreeDepth(myTree):
	treeDepth=0
	firstStr=list(myTree.keys())[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth=1+getTreeDepth(secondDict[key])
		else:
			thisDepth=1
		if thisDepth>treeDepth:
			treeDepth=thisDepth
	return treeDepth

def retrieveTree(i):
	listOfTrees=[{'no surfacing':{0:'no',1:{'flippers':
	{0:'no',1:'yes'}}}},
	{'no surfacing':{0:'no',1:{'flippers':
	{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
	return listOfTrees[i]

def classify(inputTree,featLabels,testVec):
	firstStr=list(inputTree.keys())[0]
	secondDict=inputTree[firstStr]
	featIndex=featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex]==key:
			if type(secondDict[key]).__name__=='dict':
				classLabel=classify(secondDict[key],featLabels,testVec)
			else:
				classLabel=secondDict[key]
	return classLabel
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=createTree(lenses,lensesLabels)
print(lensesTree)
createPlot(lensesTree)