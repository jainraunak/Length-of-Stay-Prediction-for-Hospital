import numpy as np
import pandas as pd
import math
import sys
from scipy.special import softmax

def remove(string):
    return string.replace(" ", "")

def loglikeihood(w,X,Y):
    r = X.shape[0]
    X = np.matmul(X,w)
    X = softmax(X,axis=1)
    ga = math.pow(10,-15)
    X = np.log(np.clip(X,ga,1-ga))
    Y = np.multiply(Y,X)
    ans = -np.sum(Y)/r
    return ans

trainfile = sys.argv[1]
testfile = sys.argv[2]
df = pd.read_csv(trainfile)
dftest = pd.read_csv(testfile)
#df = df.sample(frac=1)
dftrain = df.iloc[:,1:-1]
x = dftrain.values
r = dftrain.shape[0]
r = int(r/10)
names = dftrain.columns
n = len(names)
i = 0
names1 = []
while(i < n):
    s = remove(names[i])
    if(s == "ZipCode-3digits"):
        s = "ZipCodedigits"
    names1.append(s)
    i = i+1


dftrain.columns = names1
ytrain = df.iloc[:,-1]
col = np.ones(dftrain.shape[0])
dftrain['bparam'] = col
Ytest = ytrain[9*r:].values
ytrain = pd.get_dummies(ytrain)

r = dftrain.shape[0]
r = int(r/10)

r = dftrain.shape[0]
index = 0
names = []

#dftrain['HealthServiceArea'] = (dftrain['HealthServiceArea']-dftrain['HealthServiceArea'].mean())/np.square((dftrain['HealthServiceArea'].std()))
#dftrain['APRDRGCode'] = (dftrain['APRDRGCode']-dftrain['APRDRGCode'].mean())/np.square((dftrain['APRDRGCode'].std()))
dftrain['BirthWeight'] = (dftrain['BirthWeight']-dftrain['BirthWeight'].mean())/np.square((dftrain['BirthWeight'].std()))
dftrain['PatientDisposition'] = (dftrain['PatientDisposition']-dftrain['PatientDisposition'].mean())/np.square((dftrain['PatientDisposition'].std()))

s = 0
arr1 = [x for x in dftrain.Ethnicity.value_counts().index]
s = s+len(arr1)
arr2 = [x for x in dftrain.TypeofAdmission.value_counts().index]
s = s+len(arr2)
arr3 = [x for x in dftrain.AgeGroup.value_counts().index]
s = s+len(arr3)
arr4 = [x for x in dftrain.OperatingCertificateNumber.value_counts().sort_values(ascending=False).index]
if(len(arr4) > 150):
    arr4 = arr4[0:150]
s = s+len(arr4)
arr5 = [x for x in dftrain.PaymentTypology1.value_counts().index]
s = s+len(arr5)
arr8 = [x for x in dftrain.APRSeverityofIllnessCode.value_counts().index]
s = s+len(arr8)
arr9 = [x for x in dftrain.APRRiskofMortality.value_counts().index]
s = s+len(arr9)
arr10 = [x for x in dftrain.APRMedicalSurgicalDescription.value_counts().index]
s = s+len(arr10)
arr13 = [x for x in dftrain.APRMDCCode.value_counts().index]
s = s+len(arr13)
arr14 = [x for x in dftrain.CCSProcedureCode.value_counts().sort_values(ascending=False).index]
if(len(arr14) > 20):
    arr14 = arr14[0:20]
s = s+len(arr14)
arr15 = [x for x in dftrain.CCSDiagnosisCode.value_counts().sort_values(ascending=False).index]
if(len(arr15) > 20):
    arr15 = arr15[0:20]
s = s+len(arr15)
arr16 = [x for x in dftrain.APRDRGCode.value_counts().sort_values(ascending=False).index]
if(len(arr16) > 20):
    arr16 = arr16[0:20]
s = s+len(arr16)
arr17 = [x for x in dftrain.ZipCodedigits.value_counts().sort_values(ascending=False).index]
if(len(arr17) > 10):
    arr17 = arr17[0:10]
s = s+len(arr17)
arr18 = [x for x in dftrain.FacilityName.value_counts().sort_values(ascending=False).index]
if(len(arr18) > 10):
    arr18 = arr18[0:10]
s = s+len(arr18)
arr19 = [x for x in dftrain.HospitalCounty.value_counts().sort_values(ascending=False).index]
if(len(arr19) > 10):
    arr19 = arr19[0:10]
s = s+len(arr19)
arr20 = [x for x in dftrain.HealthServiceArea.value_counts().sort_values(ascending=False).index]
s = s+len(arr20)
arr21 = [x for x in dftrain.EmergencyDepartmentIndicator.value_counts().sort_values(ascending=False).index]
s = s+len(arr21)

npa = np.zeros((r,s))
ga = math.pow(10,-15)
q = np.log(np.clip(dftrain['TotalCosts'].values,ga,1-ga))

for i in arr1:
    npa[:, index] = np.where(dftrain['Ethnicity'] == i, q, 0)
    names.append('Ethnicity' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('Ethnicity',inplace=True,axis=1)

for i in arr2:
    npa[:, index] = np.where(dftrain['TypeofAdmission'] == i, q, 0)
    names.append('TypeofAdmission' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('TypeofAdmission',inplace=True,axis=1)

for i in arr3:
    npa[:, index] = np.where(dftrain['AgeGroup'] == i,q, 0)
    names.append('AgeGroup' + '_ls_' + str(i))
    index = index + 1

for i in arr4:
    npa[:, index] = np.where(dftrain['OperatingCertificateNumber'] == i,q, 0)
    names.append('OperatingCertificateNumber' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('OperatingCertificateNumber',inplace=True,axis=1)

for i in arr5:
    npa[:, index] = np.where(dftrain['PaymentTypology1'] == i,q, 0)
    names.append('PaymentTypology1' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('PaymentTypology1',inplace=True,axis=1)

for i in arr8:
    npa[:, index] = np.where(dftrain['APRSeverityofIllnessCode'] == i,q, 0)
    names.append('APRSeverityofIllnessCode' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('APRSeverityofIllnessCode',inplace=True,axis=1)

for i in arr9:
    npa[:, index] = np.where(dftrain['APRRiskofMortality'] == i,q, 0)
    names.append('APRRiskofMortality' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('APRRiskofMortality',inplace=True,axis=1)

for i in arr10:
    npa[:, index] = np.where(dftrain['APRMedicalSurgicalDescription'] == i,q, 0)
    names.append('APRMedicalSurgicalDescription' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('APRMedicalSurgicalDescription',inplace=True,axis=1)

for i in arr13:
    npa[:, index] = np.where(dftrain['APRMDCCode'] == i,q, 0)
    names.append('APRMDCCode' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('APRMDCCode',inplace=True,axis=1)

for i in arr15:
    npa[:, index] = np.where(dftrain['CCSDiagnosisCode'] == i,q, 0)
    names.append('CCSDiagnosisCode' + '_ls_' + str(i))
    index = index + 1
#dftrain.drop('CCSDiagnosisDescription',inplace=True,axis=1)

for i in arr14:
    npa[:, index] = np.where(dftrain['CCSProcedureCode'] == i,q, 0)
    names.append('CCSProcedureCode' + '_ls_' + str(i))
    index = index + 1
#dftrain.drop('CCSProcedureDescription',inplace=True,axis=1)

for i in arr16:
    npa[:, index] = np.where(dftrain['APRDRGCode'] == i,q, 0)
    names.append('APRDRGCode' + '_ls_' + str(i))
    index = index + 1
#dftrain.drop('CCSProcedureDescription',inplace=True,axis=1)

for i in arr17:
    npa[:, index] = np.where(dftrain['ZipCodedigits'] == i,q, 0)
    names.append('ZipCodedigits' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('ZipCodedigits',inplace=True,axis=1)

for i in arr18:
    npa[:, index] = np.where(dftrain['FacilityName'] == i,q, 0)
    names.append('FacilityName' + '_ls_' + str(i))
    index = index + 1
dftrain.drop('FacilityName',inplace=True,axis=1)

for i in arr19:
    npa[:, index] = np.where(dftrain['HospitalCounty'] == i,q, 0)
    names.append('HospitalCounty' + '_ls_' + str(i))
    index = index + 1

for i in arr20:
    npa[:, index] = np.where(dftrain['HealthServiceArea'] == i,q, 0)
    names.append('HealthServiceArea' + '_ls_' + str(i))
    index = index + 1

for i in arr21:
    npa[:, index] = np.where(dftrain['EmergencyDepartmentIndicator'] == i,q, 0)
    names.append('EmergencyDepartmentIndicator' + '_ls_' + str(i))
    index = index + 1

dt = pd.DataFrame(npa)
dt.columns = names

arr1 = [x for x in dftrain.HealthServiceArea.value_counts().index]
arr2 = [x for x in dftrain.EmergencyDepartmentIndicator.value_counts().index]
nptr = np.zeros((r,len(arr1)*len(arr2)))
nam = []
i = 0
for var1 in arr1:
    for var2 in arr2:
        nptr[:,i] = np.where(((dftrain['HealthServiceArea'] == var1) & (dftrain['EmergencyDepartmentIndicator'] == var2)),q,0)
        i = i+1
        nam.append(str(var1)+'_emer'+str(var2))
d = pd.DataFrame(nptr)
d.columns = nam

arr1 = [x for x in dftrain.AgeGroup.value_counts().index]
arr2 = [x for x in dftrain.EmergencyDepartmentIndicator.value_counts().index]
nptr = np.zeros((r,len(arr1)*len(arr2)))
nam = []
i = 0
for var1 in arr1:
    for var2 in arr2:
        nptr[:,i] = np.where(((dftrain['AgeGroup'] == var1) & (dftrain['EmergencyDepartmentIndicator'] == var2)),q,0)
        i = i+1
        nam.append(str(var1)+'AgeGroup_emer'+str(var2))
d2 = pd.DataFrame(nptr)
d2.columns = nam

dftrain = pd.concat([dftrain,dt,d,d2],axis=1)

dftrain.drop('Race',inplace=True,axis=1)

dftrain.drop('Gender',inplace=True,axis=1)

dftrain.drop('AgeGroup',inplace=True,axis=1)

dftrain.drop('FacilityId',inplace=True,axis=1)

dftrain.drop('CCSDiagnosisCode',inplace=True,axis=1)

dftrain.drop('CCSProcedureCode',inplace=True,axis=1)

dftrain.drop('PaymentTypology2',inplace=True,axis=1)

dftrain.drop('PaymentTypology3',inplace=True,axis=1)

dftrain.drop('APRDRGCode',inplace=True,axis=1)

dftrain.drop('HospitalCounty',inplace=True,axis=1)

dftrain.drop('HealthServiceArea',inplace=True,axis=1)

dftrain.drop('EmergencyDepartmentIndicator',inplace=True,axis=1)

print(dftrain.shape)
names = dftrain.columns


r = dftrain.shape[0]
r = int(r/10)
Xtrain = dftrain[0:9*r].values
Ytrain = ytrain[0:9*r].values
Xtest = dftrain[9*r:].values
Xt = Xtrain.T
alpha = 0.45
beta = 0.75
k = 500
i = 1
num = 80
r,c = Xtrain.shape
w = np.zeros((c,8))
n = 2.5
while (i <= num):
    X = np.matmul(Xtrain, w)
    yhat = softmax(X,axis=1)
    err = yhat-Ytrain
    g = np.matmul(Xt,err)
    g = g/r
    fn = np.linalg.norm(g)
    fn = np.square(fn)
    fn = alpha*fn
    lx = loglikeihood(w,Xtrain,Ytrain)
    while(loglikeihood(w-n*g,Xtrain,Ytrain) > (lx-n*fn)):
        n = n*beta
    print(i,n)
    j = 0
    while((j+1)*k < r):
        X = np.matmul(Xtrain[j*k:(j+1)*k,:],w)
        yhat = np.asarray(softmax(X,axis=1))
        err = yhat-Ytrain[j*k:(j+1)*k]
        g = np.matmul(Xtrain[j*k:(j+1)*k,:].T,err)
        g = g/k
        w = w-n*g
        j = j+1
    i = i+1
X = np.matmul(Xtest,w)
X = np.asarray(softmax(X, axis=1))
ans = np.argmax(X, axis=1)
ans = ans+1
w = np.asarray(w)
w = w.flatten()
correct = 0
wrong = 0;
for x in range(Ytest.shape[0]):
    if(ans[x] == Ytest[x]):
        correct = correct + 1

pred_error = correct/Ytest.shape[0]
print(pred_error)