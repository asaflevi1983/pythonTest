# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import shelve
import operator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from decimal import getcontext, Decimal

# Set the precision.

from sklearn import decomposition
from sklearn import datasets
import numpy as np
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
motorsNames = [
  'General Subsystem',
  'Main Drive',
  'GDU',
  'PIP Break',
  'Impression Break',
  'Bridge IFR',
  'Press IFR',
  'X Alignment',
  'X Line Sensor',
  'Fingers',
  'BRIDGE IFR ENGAGE',
  'BRIDGE VACCUM SHUTTER',
  'Front Impression Engage',
  'Rear Impression Engage',
  'EVR',
  'Pick Up Conveyor',
  'Scanning Conveyor',
  'EVR VALVE 1',
  'EVR VALVE 2',
  'EVR VALVE 3',
  'EVR VAC PRESSURE',
  'SPM X',
  'SPM Z',
  'SPM FAN',
  'ILS FAN',
  'Duplex Linear Conveyor',
  'Duplex Vacuum Conveyor',
  'DLC ENGAGE',
  'DLC VALVE 1',
  'DLC VALVE 2',
  'DLC VALVE 3',
  'DLC VAC PRESSURE',
  'DVC FRONT FAN',
  'DVC REAR FAN',
  'DLC FLIP VALVE',
  'Front MPCU Engage',
  'Rear MPCU Engage',
  'MPCU Roller',
  'BID Drive',
  'Cleaning Station Drive',
  'Cleaning Station Engage Valve',
  'Cleaning Station Wetting Valve',
  'PIP Holder',
  'PIPH ENGAGE VALVE',
  'EBM Open Mechanism',
  'EBM ENGAGE VALVE',
  'BARS Shuttle motion',
  'BARS Blanket rotate',
  'BARS Blanket Linear',
  'BARS Paper collect',
  'Feed Input',
  'Feed Roller',
  'Impression Engage Front',
  'Impression Engage Rear',
  'Impression Roller',
  'Feed Process',
  'Input IBS',
  'Exit IBS',
  'Feed Exit',
  'FI NIP',
  'FR NIP',
  'FP NIP',
  'FE NIP']

errorsToAnalyze = ['INK_SAS_PUMP_FLOW_FAILURE', 'MAIN_MOTOR_PLC_NOT_AT_SPEED', 'PH_EVENT_STACKER_UNEVEN_STACK_SURFACE_B5',
            'PLC_MOTOR_GENERAL_DRIVE_ERROR_PH_Bridge IFR', 'PLC_MOTOR_GENERAL_DRIVE_ERROR_PH_Fingers']
motorsErrorsRequireParameters=['PLC_MOTOR_GENERAL_DRIVE_ERROR_ENGINE', 'PLC_MOTOR_POSITION_ERROR_ENGINE', 'PLC_MOTOR_HOME_ERROR_ENGINE', 'PLC_MOTOR_GENERAL_DRIVE_ERROR_PH', 'PLC_MOTOR_POSITION_ERROR_PH', 'PLC_MOTOR_HOME_ERROR_PH']


def saveToFileFilteredByElementRawData():
  rawData = pd.read_excel('Failures.xlsx', sheet_name='Selected 5 failures events')
  fileToSave = shelve.open("mydata.dat")
  for index,row in rawData.iterrows():
    for specialTag in motorsErrorsRequireParameters:
        if specialTag in row['event_name']:
          rawData.set_value(index, 'event_name', row['event_name']+'_'+row['event_p1'])

  for tag in errorsToAnalyze:
    fileToSave[tag]=rawData.loc[rawData['new_failure_name'] == tag]
  fileToSave.close()

def saveToFileFailuresNames():
  rawfailures = pd.read_excel('index.xlsx', sheet_name='Failures')
  rawjams = pd.read_excel('index.xlsx', sheet_name='Jams')
  listFailures =rawfailures['Event_Name'].tolist()
  listJams =rawjams['Event_Name'].tolist()
  listJams.append(listFailures)
  for motor in motorsNames:
    for motorError in motorsErrorsRequireParameters:
      listJams.append(motorError+'_'+motor)
  fileToSave = shelve.open("mydata.dat")
  fileToSave['failures'] =listJams;
  fileToSave.close()



def createEventDistributionByFailureType(failureToAnalyze, i):
  rawFailures = shelve.open("mydata.dat")[failureToAnalyze]
  failuresList=shelve.open("mydata.dat")['failures']
  rawFailures = rawFailures[['new_failure_name', 'failure_datetime', 'event_name']]
  listOfEvents = rawFailures['event_name'].unique()
  listOfEventsDictionary = {}
  for event in listOfEvents:
    if any(event in s for s in failuresList):
      listOfEventsDictionary[event] = 0;

  grouped = rawFailures.groupby('failure_datetime')
  grouped.get_group
  for name, group in grouped:
    group = group.drop_duplicates()
    for event in group['event_name']:
      if event in listOfEventsDictionary.keys():
        listOfEventsDictionary[event] = listOfEventsDictionary[event] + 1;
  sortedList = sorted(listOfEventsDictionary.items(), key=operator.itemgetter(1), reverse=True)
  size = 70;
  printSortedDictionary(sortedList,size,failureToAnalyze + '_' + str(int(grouped.ngroups)))


def createBagOfWordsMatrix(failureToAnalyze, i):
  rawdf = shelve.open("mydata.dat")[failureToAnalyze]
  failuresList = shelve.open("mydata.dat")['failures']
  rawdf = rawdf[['new_failure_name', 'failure_datetime', 'event_name']]
  listOfEvents = rawdf['event_name'].unique()
  listOfEventsDictionary ={}
  indexForDictionary=0
  for event in listOfEvents:
    if any(event in s for s in failuresList):
      listOfEventsDictionary[event]=indexForDictionary
      indexForDictionary=indexForDictionary+1
  grouped = rawdf.groupby('failure_datetime')
  matrixScenarios=np.zeros((grouped.ngroups,(len(listOfEventsDictionary))))
  grouped.get_group
  indexGroupCounter=0
  for name, group in grouped:
    group = group.drop_duplicates()
    for event in group['event_name']:
      if event in listOfEventsDictionary:
        matrixScenarios[indexGroupCounter,listOfEventsDictionary[event]]=1
    indexGroupCounter=indexGroupCounter+1
  return {'matrixScenarios':matrixScenarios, 'listOfEventsDictionary':listOfEventsDictionary  }


def createAndPlotBagOfWordsMatrix(failureToAnalyze, i):
  print("Computing "+failureToAnalyze)

  result =createBagOfWordsMatrix(failureToAnalyze, i)
  matrix=result['matrixScenarios']
  dictOfdesc=result['listOfEventsDictionary']
  pcaResult,pcaMachine=pca(matrix)
  pcaDict=createDictionaryfromMatrix(pcaResult)
  inversedmat = pcaMachine.inverse_transform(pcaResult)
  for idx, val in enumerate(pcaResult):
    #print ("case:"+ str(idx))
    inversed=inversedmat[idx]
    for idxerr,error in enumerate(matrix[idx]):
      if error==1:
        for key, valdi in dictOfdesc.items():
          if valdi==idxerr:
            x=1
        #    print (key)

   # print(matrix[idx])
   # print (pcaResult[idx])


  #

 # for key, val in pcaDict.items():
 #   print("{} = {}".format(key, val))
  print(len(pcaDict))
  plotScatterDiagram(pcaDict, failureToAnalyze,100, "PCA")
  plotScatterDiagram(createDictionaryfromMatrix(tnse(matrix)[0]), failureToAnalyze, 100, "tsne")

def pca(matrixScenarios):
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2,whiten=False)
  principalComponents = pca.fit_transform(matrixScenarios)
  return principalComponents,pca


def tnse(matrixScenarios):
  tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
  X_tsne = tsne.fit_transform(matrixScenarios)
  return X_tsne,tnse;


def createDictionaryfromMatrix(principalComponents):

  principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
  dictionaryForPoints = {}
  for index, point in principalDf.iterrows():
    point['principal component 1']=round(point['principal component 1'], 4)
    point['principal component 2']=round(point['principal component 2'], 4)

    if (point['principal component 1'], point['principal component 2']) not in dictionaryForPoints.keys():
      dictionaryForPoints[(point['principal component 1'], point['principal component 2'])] = 0;
    dictionaryForPoints[(point['principal component 1'], point['principal component 2'])] = dictionaryForPoints[(
    point['principal component 1'], point['principal component 2'])] + 1
  return dictionaryForPoints;

def plotScatterDiagram(dictionaryForPoints, tag, groupsize, method_name):
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_xlabel('Principal Component 1', fontsize=15)
  ax.set_ylabel('Principal Component 2', fontsize=15)
  ax.set_title(tag + '  ' + str(groupsize), fontsize=20)
  xValues =[]
  yValues = []
  sizes=[]
  for key in dictionaryForPoints.keys():
    xValues.append(key[0])
    yValues.append(key[1])
    sizes.append(dictionaryForPoints[key]*50)
  colors = np.random.rand(len(dictionaryForPoints))
  ax.scatter(xValues
               ,yValues
               , c=colors
               , s=sizes, alpha=0.3)
  ax.grid()
  fig.savefig(tag + '_scatter_'+method_name+'.png')

def printSortedDictionary (sortedList,size,titleAndFileName):
  keysForHist = [sortedList[i][0] for i in range(len(sortedList))]
  valuesForHist = [sortedList[i][1] for i in range(len(sortedList))]
  keysForHist = keysForHist[:size]
  valuesForHist = valuesForHist[:size]
  fig, ax = plt.subplots(1, 1)
  fig.suptitle(titleAndFileName)
  fig.set_size_inches( 16.53,11.69, forward=True)
  ax.barh(range(len(keysForHist)), valuesForHist, align='center', alpha=0.4, height=0.9, color="blue")
  plt.yticks(range(len(keysForHist)), keysForHist)
  plt.tick_params(axis='both', which='major', labelsize=10)
  plt.tick_params(axis='both', which='minor', labelsize=4)
  fig.subplots_adjust(left=0.3)
  fig.savefig(titleAndFileName + '.png')

def main():
 # createAndPlotBagOfWordsMatrix('INK_SAS_PUMP_FLOW_FAILURE', 0)
 i=1
 for tag_ in errorsToAnalyze:
   createAndPlotBagOfWordsMatrix(tag_, i)
   createEventDistributionByFailureType(tag_, i)
   i=i+1


if __name__== "__main__":
  #kmeans(tagsList[0],2)
  getcontext().prec = 3

  main()
  #writeToShelf()
  #writeToShelfIndex()




