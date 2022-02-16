import numpy as np
import pandas as pd
import math
import sys
from scipy.special import softmax
from datetime import datetime, timedelta

def remove(string):
    return string.replace(" ", "")

def loglikeihood(w,X,Y):
    ga = math.pow(10,-15)
    r = X.shape[0]
    X = np.matmul(X,w)
    X = softmax(X,axis=1)
    X = np.log(np.clip(X,ga,1-ga))
    Y = np.multiply(Y,X)
    ans = -np.sum(Y)/r
    return ans

part = sys.argv[1]

if(part == 'a'):
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    param = sys.argv[4]
    outputfile = sys.argv[5]
    weightfile = sys.argv[6]

    dftrain = pd.read_csv(trainfile,index_col = 0)
    dftest = pd.read_csv(testfile,index_col = 0)
    ytrain = dftrain['Length of Stay']
    dftrain = dftrain.drop(columns = ['Length of Stay'])
    data = pd.concat([dftrain, dftest], ignore_index=True)
    cols = dftrain.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    r = data.shape[0]
    Xtrain = np.asarray(data[:dftrain.shape[0],:])
    Xtest = np.asarray(data[dftrain.shape[0]:, :])
    Xtrain = np.c_[np.ones(Xtrain.shape[0]),Xtrain]
    Xtest = np.c_[np.ones(Xtest.shape[0]),Xtest]
    r,c = Xtrain.shape
    dtype = pd.read_csv(param,delimiter='\t',header=None)
    w = np.zeros((c, 8))
    Xt = Xtrain.T
    ytrain = np.asarray(pd.get_dummies(ytrain))
    type = np.asarray(dtype.values)
    if(type[0][0] == 1):
        n = type[1]/r
        num = int(type[2])
        i = 1
        while(i <= num):
            X = np.matmul(Xtrain,w)
            yhat = np.asarray(softmax(X,axis=1))
            err = yhat-ytrain
            g = np.matmul(Xt,err)
            w = w-n*g
            i = i+1
    elif(type[0][0] == 2):
        n0 = type[1]/r
        num = int(type[2])
        i = 1
        f = 1.0
        while (i <= num):
            X = np.matmul(Xtrain, w)
            yhat = softmax(X,axis=1)
            err = yhat-ytrain
            g = np.matmul(Xt, err)
            n = n0/math.sqrt(f)
            w = w-n*g
            i = i+1
            f = f+1.0
    elif(type[0][0] == '3'):
        s = str(type[1])
        s = s[2:-2]
        x = s.split(",")
        n0 = float(x[0])
        alpha = float(x[1])
        beta = float(x[2])
        num = int(type[2][0])
        i = 1
        while(i <= num):
            n = n0
            X = np.matmul(Xtrain,w)
            yhat = softmax(X,axis=1)
            err = yhat-ytrain
            g = np.matmul(Xt, err)
            g = g/r
            fn = np.square(np.linalg.norm(g))
            fn = alpha*fn
            lx = loglikeihood(w,Xtrain,ytrain)
            while(loglikeihood(w-n*g,Xtrain,ytrain) > lx-n*fn):
                n = n*beta
            w = w-n*g
            i = i+1
    X = np.matmul(Xtest,w)
    X = np.asarray(softmax(X,axis=1))
    ans = np.argmax(X,axis=1)
    ans = ans+1
    w = np.asarray(w)
    w = w.flatten()
    np.savetxt(outputfile,ans)
    np.savetxt(weightfile,w)

elif(part == 'b'):
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    param = sys.argv[4]
    outputfile = sys.argv[5]
    weightfile = sys.argv[6]

    dftrain = pd.read_csv(trainfile, index_col=0)
    dftest = pd.read_csv(testfile, index_col=0)
    ytrain = dftrain['Length of Stay']
    dftrain = dftrain.drop(columns=['Length of Stay'])
    data = pd.concat([dftrain, dftest], ignore_index=True)
    cols = dftrain.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    r = data.shape[0]
    Xtrain = np.asarray(data[:dftrain.shape[0], :])
    Xtest = np.asarray(data[dftrain.shape[0]:, :])
    Xtrain = np.c_[np.ones(Xtrain.shape[0]), Xtrain]
    Xtest = np.c_[np.ones(Xtest.shape[0]), Xtest]
    r, c = Xtrain.shape
    dtype = pd.read_csv(param, delimiter='\t', header=None)
    w = np.zeros((c, 8))
    ytrain = np.asarray(pd.get_dummies(ytrain))
    type = np.asarray(dtype.values)

    if(type[0][0] == 1):
        n = type[1][0]
        num = int(type[2][0])
        k = int(type[3][0])
        i = 1
        while (i <= num):
            j = 0
            while((j+1)*k <= r):
                X = np.matmul(Xtrain[j*k:(j+1)*k,:], w)
                yhat = np.asarray(softmax(X, axis=1))
                err = yhat-ytrain[j*k:(j+1)*k]
                g = np.matmul(Xtrain[j*k:(j+1)*k,:].T, err)
                g = g/k
                w = w-n*g
                j = j+1
            i = i+1
    elif(type[0][0] == 2):
        n0 = type[1][0]
        num = int(type[2][0])
        k = int(type[3][0])
        i = 1
        f = 1.0
        while (i <= num):
            j = 0
            while((j+1)*k <= r):
                X = np.matmul(Xtrain[j*k:(j+1)*k,:], w)
                yhat = softmax(X, axis=1)
                err = yhat-ytrain[j*k:(j+1)*k]
                g = np.matmul(Xtrain[j*k:(j+1)*k,:].T, err)
                g = g/k
                n = n0/math.sqrt(f)
                w = w-n*g
                j = j+1
            i = i+1
            f = f+1.0
    elif(type[0][0] == '3'):
        s = str(type[1])
        s = s[2:-2]
        x = s.split(",")
        n0 = float(x[0])
        alpha = float(x[1])
        beta = float(x[2])
        num = int(type[2])
        k = int(type[3])
        Xt = Xtrain.T
        i = 1
        while(i <= num):
            X = np.matmul(Xtrain, w)
            yhat = softmax(X, axis=1)
            err = yhat-ytrain
            g = np.matmul(Xt,err)
            g = g/r
            fn = np.square(np.linalg.norm(g))
            fn = alpha*fn
            lx = loglikeihood(w,Xtrain,ytrain)
            n = n0
            while(loglikeihood(w-n*g,Xtrain,ytrain) > (lx-n*fn)):
                n = n*beta
            j = 0
            while((j+1)*k < r):
                X = np.matmul(Xtrain[j*k:(j+1)*k,:],w)
                yhat = np.asarray(softmax(X,axis=1))
                err = yhat-ytrain[j*k:(j+1)*k]
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
    np.savetxt(outputfile, ans)
    np.savetxt(weightfile,w)

elif(part == 'c'):
    curr = datetime.now()
    c1 = datetime.now()
    om = timedelta(minutes=9)
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    outputfile = sys.argv[4]
    weightfile = sys.argv[5]
    dftrain = pd.read_csv(trainfile, index_col=0)
    dftest = pd.read_csv(testfile, index_col=0)
    ytrain = dftrain['Length of Stay']
    dftrain = dftrain.drop(columns=['Length of Stay'])
    data = pd.concat([dftrain, dftest], ignore_index=True)
    cols = dftrain.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    r = data.shape[0]
    Xtrain = np.asarray(data[:dftrain.shape[0], :])
    Xtest = np.asarray(data[dftrain.shape[0]:, :])
    Xtrain = np.c_[np.ones(Xtrain.shape[0]), Xtrain]
    Xtest = np.c_[np.ones(Xtest.shape[0]), Xtest]
    r,c = Xtrain.shape
    w = np.zeros((c,8))
    Xt = Xtrain.T
    ytrain = np.asarray(pd.get_dummies(ytrain))
    alpha = 0.45
    beta = 0.75
    k = 500
    i = 1
    num = 1000
    n = 2.5
    while(i <= num and c1+om > datetime.now()):
        X = np.matmul(Xtrain,w)
        yhat = softmax(X, axis=1)
        err = yhat - ytrain
        g = np.matmul(Xt, err)
        g = g/r
        fn = np.square(np.linalg.norm(g))
        fn = alpha*fn
        lx = loglikeihood(w,Xtrain,ytrain)
        while (loglikeihood(w-n*g,Xtrain,ytrain) > (lx-n*fn)):
            n = n*beta
        j = 0
        while((j+1)*k < r):
            X = np.matmul(Xtrain[j*k:(j+1)*k,:],w)
            yhat = np.asarray(softmax(X,axis=1))
            err = yhat - ytrain[j*k:(j+1)*k]
            g = np.matmul(Xtrain[j*k:(j+1)*k,:].T,err)
            g = g/k
            w = w-n*g
            j = j+1
        curr1 = datetime.now()
        onem = timedelta(minutes=1)
        final = curr+onem
        if(final <= curr1 and curr1 < c1+om):
            curr = curr1
            X = np.matmul(Xtest,w)
            X = np.asarray(softmax(X,axis=1))
            ans = np.argmax(X,axis=1)
            ans = ans+1
            w1 = np.asarray(w)
            w1 = w1.flatten()
            np.savetxt(outputfile,ans)
            np.savetxt(weightfile,w1)
        i = i+1
    curr1 = datetime.now()
    if(curr1 < c1+om):
        X = np.matmul(Xtest,w)
        X = np.asarray(softmax(X,axis=1))
        ans = np.argmax(X,axis=1)
        ans = ans+1
        w = np.asarray(w)
        w = w.flatten()
        np.savetxt(outputfile,ans)
        np.savetxt(weightfile,w)

elif(part == 'd'):
    curr = datetime.now()
    c1 = datetime.now()
    om = timedelta(minutes=14)
    trainfile = sys.argv[2]
    testfile = sys.argv[3]
    outputfile = sys.argv[4]
    weightfile = sys.argv[5]
    df = pd.read_csv(trainfile)
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
        if(s=="ZipCode-3digits"):
            s = "ZipCodedigits"
        names1.append(s)
        i = i+1

    dftrain.columns = names1
    ytrain = df.iloc[:,-1]
    col = np.ones(dftrain.shape[0])
    dftrain['bparam'] = col
    ytrain = pd.get_dummies(ytrain)



    df = pd.read_csv(testfile)
    dftest = df.iloc[:,1:]
    x = dftest.values
    r = dftest.shape[0]
    r = int(r/10)
    names = dftest.columns
    n = len(names)
    i = 0
    names1 = []
    while(i < n):
        s = remove(names[i])
        if (s == "ZipCode-3digits"):
            s = "ZipCodedigits"
        names1.append(s)
        i = i+1

    dftest.columns = names1
    col = np.ones(dftest.shape[0])
    dftest['bparam'] = col



    r = dftrain.shape[0]
    r1 = dftest.shape[0]
    index = 0
    names = []

    dftrain['BirthWeight'] = (dftrain['BirthWeight']-dftrain['BirthWeight'].mean())/np.square((dftrain['BirthWeight'].std()))
    dftrain['PatientDisposition'] = (dftrain['PatientDisposition']-dftrain['PatientDisposition'].mean())/np.square((dftrain['PatientDisposition'].std()))
    dftest['BirthWeight'] = (dftest['BirthWeight']-dftest['BirthWeight'].mean())/np.square((dftest['BirthWeight'].std()))
    dftest['PatientDisposition'] = (dftest['PatientDisposition']-dftest['PatientDisposition'].mean())/np.square((dftest['PatientDisposition'].std()))

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
    if (len(arr16) > 20):
        arr16 = arr16[0:20]
    s = s+len(arr16)
    arr17 = [x for x in dftrain.ZipCodedigits.value_counts().sort_values(ascending=False).index]
    if (len(arr17) > 10):
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
    q1 = np.log(np.clip(dftest['TotalCosts'].values,ga,1-ga))
    npa1 = np.zeros((r1,s))

    for i in arr1:
        npa[:, index] = np.where(dftrain['Ethnicity'] == i, q, 0)
        npa1[:, index] = np.where(dftest['Ethnicity'] == i, q1, 0)
        names.append('Ethnicity' + '_ls_' + str(i))
        index = index+1
    dftrain.drop('Ethnicity', inplace=True, axis=1)
    dftest.drop('Ethnicity', inplace=True, axis=1)

    for i in arr2:
        npa[:, index] = np.where(dftrain['TypeofAdmission'] == i, q, 0)
        npa1[:, index] = np.where(dftest['TypeofAdmission'] == i, q1, 0)
        names.append('TypeofAdmission' + '_ls_' + str(i))
        index = index+1
    dftrain.drop('TypeofAdmission', inplace=True, axis=1)
    dftest.drop('TypeofAdmission', inplace=True, axis=1)

    for i in arr3:
        npa[:, index] = np.where(dftrain['AgeGroup'] == i, q, 0)
        npa1[:, index] = np.where(dftest['AgeGroup'] == i, q1, 0)
        names.append('AgeGroup' + '_ls_' + str(i))
        index = index + 1

    for i in arr4:
        npa[:, index] = np.where(dftrain['OperatingCertificateNumber'] == i, q, 0)
        npa1[:, index] = np.where(dftest['OperatingCertificateNumber'] == i, q1, 0)
        names.append('OperatingCertificateNumber' + '_ls_' + str(i))
        index = index + 1
    dftrain.drop('OperatingCertificateNumber', inplace=True, axis=1)
    dftest.drop('OperatingCertificateNumber', inplace=True, axis=1)

    for i in arr5:
        npa[:, index] = np.where(dftrain['PaymentTypology1'] == i, q, 0)
        npa1[:, index] = np.where(dftest['PaymentTypology1'] == i, q1, 0)
        names.append('PaymentTypology1' + '_ls_' + str(i))
        index = index + 1
    dftrain.drop('PaymentTypology1', inplace=True, axis=1)
    dftest.drop('PaymentTypology1', inplace=True, axis=1)

    for i in arr8:
        npa[:, index] = np.where(dftrain['APRSeverityofIllnessCode'] == i, q, 0)
        npa1[:, index] = np.where(dftest['APRSeverityofIllnessCode'] == i, q1, 0)
        names.append('APRSeverityofIllnessCode' + '_ls_' + str(i))
        index = index + 1
    dftrain.drop('APRSeverityofIllnessCode', inplace=True, axis=1)
    dftest.drop('APRSeverityofIllnessCode', inplace=True, axis=1)

    for i in arr9:
        npa[:, index] = np.where(dftrain['APRRiskofMortality'] == i, q, 0)
        npa1[:, index] = np.where(dftest['APRRiskofMortality'] == i, q1, 0)
        names.append('APRRiskofMortality' + '_ls_' + str(i))
        index = index + 1
    dftrain.drop('APRRiskofMortality', inplace=True, axis=1)
    dftest.drop('APRRiskofMortality', inplace=True, axis=1)

    for i in arr10:
        npa[:, index] = np.where(dftrain['APRMedicalSurgicalDescription'] == i, q, 0)
        npa1[:, index] = np.where(dftest['APRMedicalSurgicalDescription'] == i, q1, 0)
        names.append('APRMedicalSurgicalDescription' + '_ls_' + str(i))
        index = index + 1
    dftrain.drop('APRMedicalSurgicalDescription', inplace=True, axis=1)
    dftest.drop('APRMedicalSurgicalDescription', inplace=True, axis=1)

    for i in arr13:
        npa[:, index] = np.where(dftrain['APRMDCCode'] == i, q, 0)
        npa1[:, index] = np.where(dftest['APRMDCCode'] == i, q1, 0)
        names.append('APRMDCCode' + '_ls_' + str(i))
        index = index + 1
    dftrain.drop('APRMDCCode', inplace=True, axis=1)
    dftest.drop('APRMDCCode', inplace=True, axis=1)

    for i in arr15:
        npa[:, index] = np.where(dftrain['CCSDiagnosisCode'] == i, q, 0)
        npa1[:, index] = np.where(dftest['CCSDiagnosisCode'] == i, q1, 0)
        names.append('CCSDiagnosisCode' + '_ls_' + str(i))
        index = index + 1


    for i in arr14:
        npa[:, index] = np.where(dftrain['CCSProcedureCode'] == i, q, 0)
        npa1[:, index] = np.where(dftest['CCSProcedureCode'] == i, q1, 0)
        names.append('CCSProcedureCode' + '_ls_' + str(i))
        index = index + 1

    for i in arr16:
        npa[:, index] = np.where(dftrain['APRDRGCode'] == i, q, 0)
        npa1[:, index] = np.where(dftest['APRDRGCode'] == i, q1, 0)
        names.append('APRDRGCode' + '_ls_' + str(i))
        index = index + 1

    for i in arr17:
        npa[:, index] = np.where(dftrain['ZipCodedigits'] == i, q, 0)
        npa1[:, index] = np.where(dftest['ZipCodedigits'] == i, q1, 0)
        names.append('ZipCodedigits' + '_ls_' + str(i))
        index = index + 1

    for i in arr18:
        npa[:, index] = np.where(dftrain['FacilityName'] == i, q, 0)
        npa1[:, index] = np.where(dftest['FacilityName'] == i, q1, 0)
        names.append('FacilityName' + '_ls_' + str(i))
        index = index + 1

    for i in arr19:
        npa[:, index] = np.where(dftrain['HospitalCounty'] == i, q, 0)
        npa1[:, index] = np.where(dftest['HospitalCounty'] == i, q1, 0)
        names.append('HospitalCounty' + '_ls_' + str(i))
        index = index + 1

    for i in arr20:
        npa[:, index] = np.where(dftrain['HealthServiceArea'] == i, q, 0)
        npa1[:, index] = np.where(dftest['HealthServiceArea'] == i, q1, 0)
        names.append('HealthServiceArea' + '_ls_' + str(i))
        index = index + 1

    for i in arr21:
        npa[:, index] = np.where(dftrain['EmergencyDepartmentIndicator'] == i, q, 0)
        npa1[:, index] = np.where(dftest['EmergencyDepartmentIndicator'] == i, q1, 0)
        names.append('EmergencyDepartmentIndicator' + '_ls_' + str(i))
        index = index + 1

    dt = pd.DataFrame(npa)
    dt.columns = names
    dt1 = pd.DataFrame(npa1)
    dt1.columns = names

    arr1 = [x for x in dftrain.HealthServiceArea.value_counts().index]
    arr2 = [x for x in dftrain.EmergencyDepartmentIndicator.value_counts().index]
    nptr = np.zeros((r,len(arr1)*len(arr2)))
    nptr1 = np.zeros((r1,len(arr1)*len(arr2)))
    nam = []
    i = 0
    for var1 in arr1:
        for var2 in arr2:
            nptr[:, i] = np.where(((dftrain['HealthServiceArea'] == var1)&(dftrain['EmergencyDepartmentIndicator'] == var2)),q,0)
            nptr1[:, i] = np.where(((dftest['HealthServiceArea'] == var1)&(dftest['EmergencyDepartmentIndicator'] == var2)),q1,0)
            i = i+1
            nam.append(str(var1) + '_emer' + str(var2))
    d = pd.DataFrame(nptr)
    d.columns = nam
    d1 = pd.DataFrame(nptr1)
    d1.columns = nam

    arr1 = [x for x in dftrain.AgeGroup.value_counts().index]
    arr2 = [x for x in dftrain.EmergencyDepartmentIndicator.value_counts().index]
    nptr = np.zeros((r,len(arr1)*len(arr2)))
    nptr1 = np.zeros((r1,len(arr1)*len(arr2)))
    nam = []
    i = 0
    for var1 in arr1:
        for var2 in arr2:
            nptr[:, i] = np.where(((dftrain['AgeGroup'] == var1)&(dftrain['EmergencyDepartmentIndicator'] == var2)),q, 0)
            nptr1[:, i] = np.where(((dftest['AgeGroup'] == var1)&(dftest['EmergencyDepartmentIndicator'] == var2)),q1, 0)
            i = i+1
            nam.append(str(var1) + 'AgeGroup_emer' + str(var2))
    d2 = pd.DataFrame(nptr)
    d2.columns = nam
    dt2 = pd.DataFrame(nptr1)
    dt2.columns = nam

    dftrain = pd.concat([dftrain, dt, d, d2], axis=1)
    dftest = pd.concat([dftest,dt1,d1,dt2],axis=1)

    dftrain.drop('Race', inplace=True, axis=1)

    dftrain.drop('Gender', inplace=True, axis=1)

    dftrain.drop('AgeGroup', inplace=True, axis=1)

    dftrain.drop('FacilityId', inplace=True, axis=1)

    dftrain.drop('FacilityName', inplace=True, axis=1)

    dftrain.drop('ZipCodedigits', inplace=True, axis=1)

    dftrain.drop('CCSDiagnosisCode', inplace=True, axis=1)

    dftrain.drop('CCSProcedureCode', inplace=True, axis=1)

    dftrain.drop('PaymentTypology2', inplace=True, axis=1)

    dftrain.drop('PaymentTypology3', inplace=True, axis=1)

    dftrain.drop('APRDRGCode', inplace=True, axis=1)

    dftrain.drop('HospitalCounty', inplace=True, axis=1)

    dftrain.drop('HealthServiceArea', inplace=True, axis=1)

    dftrain.drop('EmergencyDepartmentIndicator', inplace=True, axis=1)

    dftest.drop('Race', inplace=True, axis=1)

    dftest.drop('Gender', inplace=True, axis=1)

    dftest.drop('AgeGroup', inplace=True, axis=1)

    dftest.drop('FacilityId', inplace=True, axis=1)

    dftest.drop('FacilityName', inplace=True, axis=1)

    dftest.drop('ZipCodedigits', inplace=True, axis=1)

    dftest.drop('CCSDiagnosisCode', inplace=True, axis=1)

    dftest.drop('CCSProcedureCode', inplace=True, axis=1)

    dftest.drop('PaymentTypology2', inplace=True, axis=1)

    dftest.drop('PaymentTypology3', inplace=True, axis=1)

    dftest.drop('APRDRGCode', inplace=True, axis=1)

    dftest.drop('HospitalCounty', inplace=True, axis=1)

    dftest.drop('HealthServiceArea', inplace=True, axis=1)

    dftest.drop('EmergencyDepartmentIndicator', inplace=True, axis=1)

    Xtrain = np.asarray(dftrain.values)
    Ytrain = np.asarray(ytrain.values)
    Xtest = np.asarray(dftest.values)
    Xt = Xtrain.T
    alpha = 0.45
    beta = 0.75
    k = 500
    i = 1
    num = 1000
    r,c = Xtrain.shape
    w = np.zeros((c,8))
    n = 2.5
    while(i <= num and c1+om > datetime.now()):
        X = np.matmul(Xtrain,w)
        yhat = softmax(X,axis=1)
        err = yhat - Ytrain
        g = np.matmul(Xt,err)
        g = g/r
        fn = np.square(np.linalg.norm(g))
        fn = alpha*fn
        lx = loglikeihood(w,Xtrain,Ytrain)
        while(loglikeihood(w-n*g, Xtrain,Ytrain) > (lx-n*fn)):
            n = n*beta
        j = 0
        while((j+1)*k < r):
            X = np.matmul(Xtrain[j*k:(j+1)*k, :],w)
            yhat = np.asarray(softmax(X,axis=1))
            err = yhat-Ytrain[j*k:(j+1)*k]
            g = np.matmul(Xtrain[j*k:(j+1)*k, :].T,err)
            g = g/k
            w = w-n*g
            j = j+1
        curr1 = datetime.now()
        onem = timedelta(minutes=1)
        final = curr+onem
        if(final <= curr1 and curr1 < c1+om):
            curr = curr1
            X = np.matmul(Xtest,w)
            X = np.asarray(softmax(X,axis=1))
            ans = np.argmax(X, axis=1)
            ans = ans+1
            w1 = np.asarray(w)
            w1 = w1.flatten()
            np.savetxt(outputfile,ans)
            np.savetxt(weightfile,w1)
        i = i+1
    curr1 = datetime.now()
    if(curr1 < c1+om):
        X = np.matmul(Xtest,w)
        X = np.asarray(softmax(X,axis=1))
        ans = np.argmax(X,axis=1)
        ans = ans+1
        w = np.asarray(w)
        w = w.flatten()
        np.savetxt(outputfile,ans)
        np.savetxt(weightfile,w)