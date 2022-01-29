import cv2 as cv
import numpy as np
from PIL import Image
import sys,os
from skimage import measure
import math as m
import pandas as pd
import warnings
import scipy.io
warnings.filterwarnings("ignore")
#import time
#import matplotlib.pyplot as plt

def ImageLists(InPdirectory,ext='png'):
    '''''
    No0.
    Input:Input file directory and input file format.
    Output:lists of figure
    '''''
    fnL = os.listdir(InPdirectory)
    figList=[s for s in fnL if s[-3:]==ext]
    PosiList=[float(i[4:-5]) for i in figList]
    PlaneList=[s[0:3] for s in figList]
    return figList,PosiList,PlaneList

def TrimImage(pix,b=0,w=255):
    '''''
    No1.Trim image function
    Preserve the black and white pixel only, get rid of others.
    Input:pixel np array (only one slice)
    Output:trim pixel array,PIL.Image
    '''''
    Rnum=pix.shape[0]
    Cnum=pix.shape[1]
    i,j=[0,0]
    while i < Rnum:
        if ((b or w) not in pix[i,:]):
            pix=np.delete(pix,i,axis=0)
            Rnum=pix.shape[0]
        else:
            i+=1
        

    while j < Cnum:
        if ((b or w) not in pix[:,j]):
            pix=np.delete(pix,j,axis=1)
            Cnum=pix.shape[1]
        else:
            j+=1
    imtr=Image.fromarray(pix)
    return pix,imtr


def BWInverse(p):
    """""
    No2.Convert/Inverse image to BW image
    """""
    p_=255-p
    return p_

def PropertiesMeasure(p_):
    """""
    No3.Properties measuring
    """""
    l_b=measure.label(p_,connectivity=2)
    properties = measure.regionprops(l_b)
    return properties

def PrincipalDirecCentroid(p_,properties):
    """""
    No4.Perform PCA to each object in single image
    Returns principal direction, centroid and Standardized points of each object as lists.
    """""
    PrincipalDirList=[]
    Centroid=[]
    STDCoor=[]
    for i in range(0,len(properties)):
        s=p_.shape
        pixCoor=properties[i].coords
        Cen=properties[i].centroid
        Cen=(Cen[1],s[0]-Cen[0])
        #np.unique(l_b)
        c=pixCoor[:,1].reshape(-1,1)
        r=pixCoor[:,0].reshape(-1,1)
        y=s[0]-r
        PlaneCoord=np.concatenate((c,y),axis=1)
        NormPlaneCoord=(PlaneCoord-np.mean(PlaneCoord,axis=0))
        COV=np.cov(NormPlaneCoord[:,0],NormPlaneCoord[:,1])
        U,S,V=np.linalg.svd(COV)
        #U=U[:,0]#.reshape(2,1)
        PrincipalDirList.append(U)
        Centroid.append(Cen)
        STDCoor.append(NormPlaneCoord)
    return PrincipalDirList,Centroid,STDCoor

def DirectionFabric(PrincipalDirList):
    """""
    No5.Calculate the 2X2 direction fabric from principal direction list.
    The principal direction list contains 2D major principal direction vector of each object.
    """""
    F=np.zeros((2,2))
    sizeF=F.shape
    for i in range(0,len(PrincipalDirList)):
        n=PrincipalDirList[i][:,0]
        for j in range(0,sizeF[0]):
            for k in range(0,sizeF[1]):
                F[j,k]=F[j,k]+n[j]*n[k]
    num_object=len(PrincipalDirList)
    F=F/len(PrincipalDirList)
    
    return F, num_object


def AreaFraction(p_,properties):
    """""
    No6.Calculate the area fraction by ratio of object pixel number and total pixel number
    Input 'properties'
    Output : area fraction of single figure.
    """""
    AList=[]
    s=p_.shape
    Total_A=np.prod(s)
    for i in range(0,len(properties)):
        a=properties[i].area
        AList.append(a)
    AreaFraction=np.sum(AList)/Total_A
    return AreaFraction

def OBJCheck(properties,PrincipalDirList,Centroid,STDCoor):
    """""
    No7.In the single slice, delete the object if the product of principal directions components is 'NaN',
    aviod error in further calculation.
    """""
    i=0
    while i<len(PrincipalDirList):
        if m.isnan(np.prod(PrincipalDirList[i])):
            del PrincipalDirList[i]
            del properties[i]
            del Centroid[i]
            del STDCoor[i]
            i=0
        else:
            i+=1
    return properties,PrincipalDirList,Centroid,STDCoor

def FabricTwo_D(figList,Indirec):
    """""
    No8.Merge No1~No7 functions
    Calculate the directional fabric and area fraction of images in a folder
    Return both of them as lists
    0609:add the number of object.
    """""
    DirFabric=[]
    AreaFrac=[]
    OBJ_num=[]
    for i in figList:
        Direc=Indirec+'/'+i
        imcv=cv.imread(Direc)
        p=imcv[:,:,0]
        p,im=TrimImage(p)
        p_=BWInverse(p)
        properties=PropertiesMeasure(p_)
        PrincipalDirList,Centroid,STDCoor=PrincipalDirecCentroid(p_,properties)
        Af=AreaFraction(p_,properties)
        properties,PrincipalDirList,Centroid,STDCoor=OBJCheck(properties,PrincipalDirList,Centroid,STDCoor)
        F,num_object=DirectionFabric(PrincipalDirList)
        DirFabric.append(F)
        AreaFrac.append(Af)
        OBJ_num.append(num_object)
    return DirFabric,AreaFrac,OBJ_num

def ExPand3D(figList,DirFabric):
    """""
    No.9 Expand 2D matrix to 3D
    Based on the plane name, expand 2X2 matrix to 3X3 matrix by adding one row and column of zeros.
    """""
    ThreeDFabric=[]
    for i in range(0,len(figList)):
        A=np.zeros((3,3))
        TwoDF=DirFabric[i]
        if figList[i][:3]=='x-y':
            A[0,0]=TwoDF[0,0]
            A[0,1]=TwoDF[0,1]
            A[1,0]=TwoDF[1,0]
            A[1,1]=TwoDF[1,1]
        elif figList[i][:3]=='x-z':
            A[0,0]=TwoDF[0,0]
            A[0,2]=TwoDF[0,1]
            A[2,0]=TwoDF[1,0]
            A[2,2]=TwoDF[1,1] 
        elif figList[i][:3]=='y-z':
            A[1,1]=TwoDF[0,0]
            A[1,2]=TwoDF[0,1]
            A[2,1]=TwoDF[1,0]
            A[2,2]=TwoDF[1,1]
        ThreeDFabric.append(A)
    return ThreeDFabric

def FabricExtraction(ThreeDFabric):
    """""
    No.10
    Decompose 3X3 matrix to six lists
    """""
    F11=[F[0,0] for F in ThreeDFabric]
    F22=[F[1,1] for F in ThreeDFabric]
    F33=[F[2,2] for F in ThreeDFabric]
    F23=[F[1,2] for F in ThreeDFabric]
    F13=[F[0,2] for F in ThreeDFabric]
    F12=[F[0,1] for F in ThreeDFabric]
    return F11,F22,F33,F23,F13,F12


def RepPlaneFrame(DFXY):
    '''''
    No11.
    Use only ONE row to represent each plane
    '''''
    dfxy=DFXY[['A11','A22','A33','A23','A13','A12','Area Fraction']].mean()
    dfxy=pd.DataFrame(dfxy).T
    dfxy.insert(0,'Plane',DFXY['Plane'][0])
    return dfxy

def AverageRepDataFrame(InDir,Sep):
    """""
    No.12:Merge
    Inputs:
    1. InDir: Directory of specimens(each specimen as a folder)
    2. Sep: Folder name, i.e.'CE-1'
    
    Outputs:
    1. DF: Dataframe contains the fabric data of each slice of each specimen, row number=slice number.
    2. RDataFrame: Dataframe contains the average fabric data, row number = 1.
    """""
    Indirec= InDir+'/'+Sep
    figList,PosiList,PlaneList=ImageLists(Indirec)
    DirFabric,AreaFrac,OBJ_num=FabricTwo_D(figList,Indirec)
    ThreeDFabric=ExPand3D(figList,DirFabric)
    A11,A22,A33,A23,A13,A12=FabricExtraction(ThreeDFabric)
    Data={'FileName':figList,'Plane':PlaneList,'Position':PosiList,\
         #'2D Dirtional Fabric':DirFabric,'3D Directional Fabric':ThreeDFabric,\
          'Object number':OBJ_num,'A11':A11,'A22':A22,'A33':A33,'A23':A23,'A13':A13,'A12':A12,\
         'Area Fraction':AreaFrac}
    DF=pd.DataFrame(Data)
    DF.insert(0,'Specimen',Sep)
    DFXY=DF[DF['Plane']=='x-y'].sort_values(by=['Position']).reset_index().drop(columns=['index'])
    DFXZ=DF[DF['Plane']=='x-z'].sort_values(by=['Position']).reset_index().drop(columns=['index'])
    DFYZ=DF[DF['Plane']=='y-z'].sort_values(by=['Position']).reset_index().drop(columns=['index'])
    #Adjust dataframe
    dfxy=RepPlaneFrame(DFXY)
    dfxz=RepPlaneFrame(DFXZ)
    dfyz=RepPlaneFrame(DFYZ)
    #DFC contains three rows, which represent three plane
    DFC=pd.concat([dfxy,dfxz,dfyz],ignore_index=True)

    #Combine 2D fabric from three planes by taking average of non zero values.
    Diagonal=0.5*DFC[['A11','A22','A33']].sum()
    OFFDiagonal=DFC[['A23','A13','A12']].sum()
    MeanAf=DFC[['Area Fraction']].mean()
    RDF=pd.concat([Diagonal,OFFDiagonal,MeanAf])
    RDataFrame=pd.DataFrame(RDF).T
    RDataFrame.insert(0,'Specimen',Sep)
    return DF,RDataFrame


def SortDataFrame(GloDF):
    """""
    No13.After combine the slicing data by average values in terms of positions,
    and then by mean values of non zero components in terms of planes.
    Each specimen will simply be represented by one row of data frame.
    This function will extract the specimen ID from specimen name (file name),
    then sort them based on IDs.
    """""
    SpecimenList=list(GloDF[['Specimen'][0]])
    IDList=[]
    for SPE in SpecimenList:
        IDList.append(int(''.join(filter(str.isdigit, SPE))))

    GloDF.insert(1,'ID',IDList)
    GloDF=GloDF.sort_values('ID',ignore_index=True)
    return GloDF

def sortSpecimen(GloEachSliceDF):
    """""
    No14.Return a specimen list with correct order.
    """""
    Specimen=list(np.unique(GloEachSliceDF[['Specimen']]))
    IDList=[]
    for SPE in Specimen:
        IDList.append(int(''.join(filter(str.isdigit, SPE))))
    #With IDList the Specimen list could be sorted by IDList!
    zip_list=zip(IDList,Specimen)
    zip_list=sorted(zip_list)
    Specimen=[element for _, element in zip_list]
    return Specimen

#Adjust GloEachSliceDF, sort specimen order. And also sort by position in each specimen.
def SortTheGlobalFrameOfEachSlice(GloEachSliceDF):
    """""
    No15.Decompose the global data frame contains each slice > Specimen phase > Plane phase
    Then sort the slice information by position, finally re-construct the sorted global dataframe again as the output.
    """""
    SortedGLEachSliceFD=pd.DataFrame()
    Specimen=sortSpecimen(GloEachSliceDF)
    Plane=list(np.unique(GloEachSliceDF[['Plane']]))
    for S in Specimen:
        SpecimenSelect=GloEachSliceDF[GloEachSliceDF['Specimen']==S]
        for P in Plane:
            PlaneSelect=SpecimenSelect[SpecimenSelect['Plane']==P].sort_values('Position',ignore_index=True)
            SortedGLEachSliceFD=SortedGLEachSliceFD.append(PlaneSelect,ignore_index=True)
    return SortedGLEachSliceFD

def SpecimenPlaneSelection(SFD,Specimen,Plane):
    """""
    No16.It allows me to select the slices data by specify
    1. Specimen: i.e. 'CE-2'
    2. Plane: i.e. 'x-z'
    """""
    SpecimenSelect=SFD[SFD['Specimen']==Specimen]
    PlaneSelect=SpecimenSelect[SpecimenSelect['Plane']==Plane]
    return PlaneSelect
##0627 Normalization related works
def CalNormOfRow(GloDF,a):
    """""
    No17.Input: Dataframe includes the orientation fabric components in columns range (start from a)
    Can use GloDF.columns to specify the index a.
    Output: the List of matrix norm for each row
    """""
    NormList=[]
    for i in range(len(GloDF)):
        Row=GloDF.loc[i]
        A=np.array([ [Row[a],Row[a+5],Row[a+4]],\
                    [Row[a+5],Row[a+1],Row[a+3]],\
                    [Row[a+4],Row[a+3],Row[a+2]] ])
        Norm=np.linalg.norm(A)
        NormList.append(Norm)
    return NormList

def NormalizeFabricArray(GloDF,idx,NormList):
    """""
    No18.Input:
    1. Dataframe
    2. idx:column index of F11/A11
    3. NormList: list of norm of orientation fabric
    Output:
    Array of normalized direciton fabric
    """""
    Dir_FabricArray=np.array(GloDF[GloDF.columns[idx:idx+6]])#Establish the direction fabric from dataframe
    shp=Dir_FabricArray.shape
    #normalized_Dir=Dir_FabricArray[0,:]/NormList[0]
    Spec_num=shp[0]
    for i in range(Spec_num):
        if i==0:
        #Reshape is importantm use it before concatenate
            Norm_DirArray=(Dir_FabricArray[i,:]/NormList[i]).reshape(1,-1)
        else:
            B=(Dir_FabricArray[i,:]/NormList[i]).reshape(1,-1)
            Norm_DirArray=np.concatenate((Norm_DirArray,B),axis=0)
    return Norm_DirArray

def ReplaceFabricByNormalize(GloDF,Norm_DirArray,idx):
    """""
    N019.
    """""
    #Replace GloDF by normalized column, copy a dataframe for modification.
    GloDF_N=GloDF.copy()
    #Convert numpy array to dataframe
    DFNorm_DirArray=pd.DataFrame(Norm_DirArray, columns = [i for i in range(Norm_DirArray.shape[1])])
    #Replace the matrix range by normalized values
    GloDF_N[GloDF_N.columns[idx:idx+6]]=DFNorm_DirArray[DFNorm_DirArray.columns]
    return GloDF_N

def ExcelWrite(DfList,SheetList,path,xlsxName):
    """""
    No20.Write several DataFrames to excel with different sheet.
    DfList:Include all data frames
    SheetList: Specify the sheet name for each dara frame.
    path:directory of writing excel
    xlsxName:name of excel file.
    """""
# create excel writer
    writer = pd.ExcelWriter(path+xlsxName)
# write dataframe to excel sheet named 'marks'
    for i in range(len(DfList)):
        DfList[i].to_excel(writer,SheetList[i])
# save the excel file
    writer.save()
    print('DataFrames are written successfully to Excel Sheet.')

def MergeAllfuncs(InDir):
    """""
    No21.
    Calculate direction fabric & area fractions for each (specimen) folder in the pics-num folder.
    Input directiry include n-pics
    """""
    GloDF=pd.DataFrame()
    GloEachSliceDF=pd.DataFrame()
    fnL = os.listdir(InDir)
    for Sep in fnL:
        DF,RDataFrame=AverageRepDataFrame(InDir,Sep)
        GloDF=GloDF.append(RDataFrame,ignore_index=True)
        GloEachSliceDF=GloEachSliceDF.append(DF,ignore_index=True)

    GloDF=SortDataFrame(GloDF)    
    SortedGLEachSliceFD=SortTheGlobalFrameOfEachSlice(GloEachSliceDF)
    #Normalize the direction fabric
    return GloDF,SortedGLEachSliceFD

def OrientationFabricNorm(GloDF,idx=2):
    """""
    No22.
    """""
    NormList=CalNormOfRow(GloDF,idx)
    Norm_DirArray=NormalizeFabricArray(GloDF,idx,NormList)
    
    GloDF_N=ReplaceFabricByNormalize(GloDF,Norm_DirArray,idx)
    GloDF.insert(len(GloDF.columns)-1,"Norm of DirFabric",NormList)
    return GloDF,GloDF_N

def convertDFfromMat(matDir):
    """""
    No23.
    Convert .mat to dataframe
    The format was restricted that it could only be used for trial project.
    """""
    mat = scipy.io.loadmat(matDir)
    Specimen_num=len(mat['stl_info'])
    ID=[int(mat['stl_info'][i][0][1][0][0]) for i in range(0,Specimen_num)]
    F11=[round(mat['stl_info'][i][0][3][0][0],4) for i in range(0,Specimen_num)]
    F22=[round(mat['stl_info'][i][0][4][0][0],4) for i in range(0,Specimen_num)]
    F33=[round(mat['stl_info'][i][0][5][0][0],4) for i in range(0,Specimen_num)]
    F23=[round(mat['stl_info'][i][0][6][0][0],4) for i in range(0,Specimen_num)]
    F13=[round(mat['stl_info'][i][0][7][0][0],4) for i in range(0,Specimen_num)]
    F12=[round(mat['stl_info'][i][0][8][0][0],4) for i in range(0,Specimen_num)]
    Vf=[round(mat['stl_info'][i][0][9][0][0],4) for i in range(0,Specimen_num)]
    Three_DF={'ID':ID,'F11':F11,'F22':F22,'F33':F33,'F23':F23,'F13':F13,'F12':F12,'VolFrac':Vf}
    TDFFromMat=pd.DataFrame(Three_DF)
    return TDFFromMat