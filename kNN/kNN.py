#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
from os import listdir
def createDataSet():
	group=array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
	labels=['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	dataSetSize=dataSet.shape[0]
	diffMat=tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat=diffMat**2
	sqDistance=sqDiffMat.sum(axis=1)
	distance=sqDistance**0.5
	sortedDistIndices=distance.argsort()
	classCount={}
	for i in range(k):
		voteIlabel=labels[sortedDistIndices[i]]
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
	sortClassCount=sorted(classCount.items(),
		key=operator.itemgetter(1),reverse=True)
	return sortClassCount[0][0]

def autoNorm(dataSet):
	m=dataSet.shape[0]
	minVals=dataSet.min(0)
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	NormDataSet=zeros(shape(dataSet))
	NormDataSet=dataSet-tile(minVals,(m,1))
	NormDataSet=NormDataSet/tile(ranges,(m,1))
	return NormDataSet,ranges,minVals

def file2matrix(filename):
	fr=open(filename)
	numberOfLines=len(fr.readlines())
	returnMat=zeros((numberOfLines,3))
	classLableVector=[]
	fr=open(filename)
	index=0
	for line in fr.readlines():
		line=line.strip()
		listFromLine=line.split('\t')
		returnMat[index,:]=listFromLine[0:3]
		classLableVector.append(int(listFromLine[-1]))
		index+=1
	return returnMat,classLableVector

def datingClassTest():
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normDataSet,ranges,minVals=autoNorm(datingDataMat)
	hoRatio=0.10
	errorRate=0.0
	m=normDataSet.shape[0]
	NumTestVecs=int(hoRatio*m)
	for i in range(NumTestVecs):
		classifierResult=classify0(normDataSet[i,:],normDataSet[NumTestVecs:m,:],
									datingLabels[NumTestVecs:m],3)
		print('the classifierresult is %d and the real answer is %d'
									%(classifierResult,datingLabels[i]))
		if classifierResult!=datingLabels[i]:
			errorRate+=1.0
	print('total error rate is %f'%(errorRate/float(NumTestVecs)))

def img2vector(filename):
	returnVector=zeros((1,1024))
	fr=open(filename)
	for i in range(32):
		lineStr=fr.readline()
		for j in range(32):
			returnVector[0,i*32+j]=int(lineStr[j])
	return returnVector

def handWritingClassTest():
	hwLabels=[]
	trainingFileList=listdir('trainingDigits')
	m=len(trainingFileList)
	trainingMat=zeros((m,1024))
	for i in range(m):
		fileNameStr=trainingFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNum=int(fileStr.split('_')[0])
		hwLabels.append(classNum)
		trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)
	testFileList=listdir('testDigits')
	mTest=len(testFileList)
	errorRate=0.0
	for i in range(mTest):
		fileNameStr=testFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNum=int(fileStr.split('_')[0])
		vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)
		classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
		if classifierResult!=classNum:
			errorRate+=1.0
	print('total error rate is %f'%(errorRate/float(mTest)))

handWritingClassTest()
#datingClassTest()