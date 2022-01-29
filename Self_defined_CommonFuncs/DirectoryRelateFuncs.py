import os
import re
def DirectoryReporter(LayerNum=0):
    """""
    Report the directory
    The LayerNum means the upper layer number of directory.
    1. To get current directory, implement
    DirectoryReporter()
    Output: 'C:/Users/user/Desktop/3DOrgs/EXCEL'
    
    2. To show the directory without the last two layers
    DirectoryReporter(2)
    Output: 'C:/Users/user/Desktop'
    """""
    c=os.getcwd()#Get current work directory
    c=c.replace('\\','/')#Replace'\\'by'/'
    Slashidx=[idx for idx in range(0,len(c)) if c[idx]=='/']#Find index of '/'
    if LayerNum!=0:
        c=c[:Slashidx[-LayerNum]]
    return c


def FileList(InPdirectory,ext=None):
    
    '''''
    Input:Input file directory and input file format.
    Output:lists of specific format
    '''''
    if isinstance(ext,str):
        fnL = os.listdir(InPdirectory)
        FileList=[fnL[i] for i in range(len(fnL)) if fnL[i].split('.')[-1]==ext]
    else:
        FileList=os.listdir(InPdirectory)
    
    return FileList

def ExtractNumFromListEltName(EXCEList,AssignedType=''):
    """""
    Extract the number from the list element name.
    Produce the new list contains these number.
    AssignedType=''(as default), the number in output list will be int.
    AssignedType='str', output elts will be string
    AssignedType='float', output elts will be float
    """""
    NumList=[]
    for j in range(len(EXCEList)):
        IntList = [i for i in EXCEList[j] if i.isdigit()]
        StrS=''.join(IntList)
        if StrS=='':
            StrS=0
        else:
            pass
        if AssignedType=='str':
            StrS=StrS
        elif AssignedType=='float':
            StrS=float(StrS)
        else:
            StrS=int(StrS)
        NumList.append(StrS)
    return NumList


def SortListByAnotherList(ListbeSort,ListMakeSort,Reverse=False):
    """""
    Sort the list by another list
    The two list must have same length.
    Reverse=True/False to control the sorted order
    """""
    return [x for _, x in sorted(zip(ListMakeSort, ListbeSort),reverse=Reverse)]

def file_extension_remover(file_name):
    """""
    Remove file extension
    return the string
    """""
    split_name=os.path.splitext(file_name)
    file=split_name[0]
    return file

def float_extractor_from_string(string):
    """""
    Enter string
    Returns the float list.
    To extract specific float:
    Output[i], where i is the index from the list.
    """""
    return re.findall(r'[\d\.\d]+', string)