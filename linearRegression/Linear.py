#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt 
def loadDataSet(fileName):
	numFeat=len(open(fileName).readline().split('\t'))-1
	dataMat=[];labelMat=[]
	fr=open(fileName)
	for line in fr.readlines():
		lineArr=[]
		currLine=line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(currLine[i]))
		labelMat.append(float(currLine[numFeat]))
		dataMat.append(lineArr)
	return dataMat,labelMat

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def standRegres(xArr,yArr):
	xMat=mat(xArr);yMat=mat(yArr).T
	xTx=xMat.T*xMat
	if linalg.det(xTx)==0.0:
		print('can not convert into inverse')
		return 
	ws=xTx.I*(xMat.T*yMat)
	return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
	xMat=mat(xArr);yMat=mat(yArr).T
	m=shape(xMat)[0]
	weights=mat(eye(m))
	for j in range(m):
		diffMat=testPoint-xMat[j,:]
		weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
	xTx=xMat.T*(weights*xMat)
	if linalg.det(xTx)==0:
		print('opps')
		return 
	ws=xTx.I*(xMat.T*(weights*yMat))
	return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
	m=shape(testArr)[0]
	yHat=zeros(m)
	for i in range(m):
		yHat[i]=lwlr(testArr[i],xArr,yArr,k)
	return yHat

def plotLine(xMat,yMat):
	import matplotlib.pyplot as plt
	yHat=lwlrTest(xArr,xArr,yArr,0.01)
	xMat=mat(xMat)
	srtInd=xMat[:,1].argsort(0)
	xSort=xMat[srtInd][:,0,:]
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(xSort[:,1],yHat[srtInd])
	ax.scatter(xMat[:,1].flatten().A[0],mat(yMat).T[:,0].flatten().A[0],s=2,c='red')
	#xCopy=xMat.copy()
	#xCopy.sort(0)
	#yHat=xCopy*ws
	#ax.plot(xCopy[:,1],yHat)
	plt.show()

def ridgeRegres(xMat,yMat,lam=0.2):
	xTx=xMat.T*xMat
	denom=xTx+eye(shape(xMat)[1])*lam
	if linalg.det(denom)==0.0:
		print('opps')
		return
	ws=denom.I*(xMat.T*yMat)
	return ws

def ridgeTest(xArr,yArr):
	xMat=mat(xArr);yMat=mat(yArr).T
	yMean=mean(yMat,0)
	yMat=yMat-yMean
	xMeans=mean(xMat,0)
	xVar=var(xMat,0)
	xMat=(xMat-xMeans)/xVar
	numTestPts=30
	wMat=zeros((numTestPts,shape(xMat)[1]))
	for i in range(numTestPts):
		ws=ridgeRegres(xMat,yMat,exp(i-10))
		wMat[i,:]=ws.T
	return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
	xMat=mat(xArr);yMat=mat(yArr).T
	yMean=mean(yMat,0)
	yMat=yMat-yMean
	xMat=regularize(xMat)
	m,n=shape(xMat)
	ws=zeros((n,1));returnMat=xMat.copy();wsMax=ws.copy()
	for i in range(numIt):
		print(ws.T)
		lowestError=float("inf")
		for j in range(n):
			for sign in [1,-1]:
				wsTest=ws.copy()
				wsTest[j]+=eps*sign
				yTest=xMat*wsTest
				rssE=rssError(yMat.A,yTest.A)
				if rssE<lowestError:
					lowestError=rssE
					wsMax=wsTest
		ws=wsMax.copy()
		returnMat[i,:]=ws.T
	return returnMat

xArr,yArr=loadDataSet('abalone.txt')
#returnMat=ridgeTest(xArr,yArr)
#print(returnMat)
returnMat=stageWise(xArr,yArr,0.005,500)
fig=plt.figure()
ay=fig.add_subplot(111)
ay.plot(returnMat)
plt.show()
