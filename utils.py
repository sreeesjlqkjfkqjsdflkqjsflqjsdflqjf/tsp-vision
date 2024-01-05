'''
Created on May 1, 2022

@author: deckyal
'''

def checkDirMake(directory):
    #print(directory)
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)