'''
Created on Dec 13, 2019

@author: aiswa
'''

import numpy as np
import torch as tr
import csv

class Net(tr.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tr.nn.Linear(in_features = 16, out_features = 15)
        self.l2 = tr.nn.Linear(in_features = 15, out_features = 1)
        
    def forward(self, x):
        h = tr.relu(self.l1(x.flatten()))
        y = tr.sigmoid(self.l2(h))
        return y
    
    def convertState(self,state):
        convertedState = np.zeros((4,4))
        np.place(convertedState,state == 'X',-1)
        np.place(convertedState,state == 'O',-1)
        np.place(convertedState,state == '+',1)
        np.place(convertedState,state == 'B',0)
        convertedState = tr.tensor(convertedState).float()
        return convertedState
    
    def csvReader(self,state):
        target = 0
        with open('tic-tac-toe-4.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count = line_count + 1
                else:
                    currentState = np.asarray(row[:16]).reshape(4,4)
                    if(state == currentState).all():
                        status = row[16]
                        if(status == 'TRUE'):
                            target = 1
                        elif(status == 'FALSE'):
                            target = -1
                        else:
                            target = 0
                        
                        break 
                    line_count = line_count + 1
                    
        return target
                    
    def getValueForStates(self,states):
        maxVal = 0.
        for state in states:
            np.place(state,state == None, "B")
            
            board = self.convertState(state)
        
            y = self(board)
            
            if(float(y.data[0]) > maxVal):
                maxVal = float(y.data[0])
                    
        return  maxVal,[],1
    
    def feedDataToModel(self,state,target):
        np.place(state,state == None, "B")
        board = self.convertState(state)
        optimizer = tr.optim.SGD(params=self.parameters(), lr=.1)
        for i in range(100):
            optimizer.zero_grad()
            y = self(board)
            loss = (y - target)**2
            
            loss.backward()
            optimizer.step()

    
    def traindata(self):
        with open('tic-tac-toe-4.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count = line_count + 1
                else:
                    target = 0
                    currentState = np.asarray(row[:16]).reshape(4,4)
                    status = row[16]
                    if(status == 'TRUE'):
                        target = 1
                    elif(status == 'FALSE'):
                        target = -1
                    else:
                        target = 0
                    
                    line_count = line_count + 1
                    board = self.convertState(currentState)
                    optimizer = tr.optim.SGD(params=self.parameters(), lr=.1)
                    for i in range(100):
                        optimizer.zero_grad()
                        y = self(board)
                        print(i)
                        print(y)
                        print(y.data[0])
                        loss = (y - target)**2
                        loss.backward()
                        optimizer.step()
                        
                    tr.save(self,"model")
                        
if __name__ == "__main__":
    net = Net()

    net.traindata()