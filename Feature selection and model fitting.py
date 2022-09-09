
import numpy as np
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import math
from sklearn.feature_selection import RFE
from sklearn import linear_model
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score

pd.set_option('display.max_rows', None) # The output in the console is not truncated (rows)
pd.set_option('display.expand_frame_repr', False) # The output in the console is not truncated (columns)


## Import the data

# X data
dataAbx = pd.read_csv("Data/ReNEW inferred latent variables_Antibiotics.csv", sep=',')
dataAbx = pd.DataFrame(dataAbx)
dataPPCP = pd.read_csv("Data/ReNEW inferred latent variables_PPCPs.csv", sep=',')
dataPPCP = pd.DataFrame(dataPPCP)
dataGenes = pd.read_csv("Data/ReNEW inferred latent variables_Genes.csv", sep=',')
dataGenes = pd.DataFrame(dataGenes)
dataPharm = pd.read_csv("Data/Pharmacy inferred latent variables.csv", sep=',')
dataPharm = pd.DataFrame(dataPharm)
dataWCDoH = pd.read_csv("Data/Hospital and clinic inferred latent variables.csv", sep=',')
dataWCDoH = pd.DataFrame(dataWCDoH)
dataVet = pd.read_csv("Data/Veterinary inferred latent variables.csv", sep=',')
dataVet = pd.DataFrame(dataVet)

# # Y data - Untransformed
dataResAbx = pd.read_csv("Data/Resistance rate_Abx.csv", sep=',')
dataResAbx = pd.DataFrame(dataResAbx)
dataResMic = pd.read_csv("Data/Resistance rate_Microbes.csv", sep=',')
dataResMic = pd.DataFrame(dataResMic)

# Y data - Untransformed
dataResIN = pd.read_csv("Data/AMR cases_hospital associated.csv", sep=',')
dataResIN = pd.DataFrame(dataResIN)
dataResOUT = pd.read_csv("Data/AMR cases_community associated.csv", sep=',')
dataResOUT = pd.DataFrame(dataResOUT)

## Convert dates to date_time objects
# X
dataAbx['Date'] = pd.to_datetime(dataAbx['Date'], utc=False)
dataPPCP['Date'] = pd.to_datetime(dataPPCP['Date'], utc=False)
dataGenes['Date'] = pd.to_datetime(dataGenes['Date'], utc=False)
dataPharm['Date'] = pd.to_datetime(dataPharm['Date'], utc=False)
dataWCDoH['Date'] = pd.to_datetime(dataWCDoH['Date'], utc=False)
dataVet['Date'] = pd.to_datetime(dataVet['Date'], utc=False)
# Y
dataResAbx['Date'] = pd.to_datetime(dataResAbx['Date'], utc=False)
dataResMic['Date'] = pd.to_datetime(dataResMic['Date'], utc=False)
# Y
dataResIN['Date'] = pd.to_datetime(dataResIN['Date'], utc=False)
dataResOUT['Date'] = pd.to_datetime(dataResOUT['Date'], utc=False)

# X
dataAbx.drop(dataAbx.columns[0], axis=1, inplace=True)
dataPPCP.drop(dataPPCP.columns[0], axis=1, inplace=True)
dataGenes.drop(dataGenes.columns[0], axis=1, inplace=True)
dataPharm.drop(dataPharm.columns[0], axis=1, inplace=True)
dataWCDoH.drop(dataWCDoH.columns[0], axis=1, inplace=True)
dataVet.drop(dataVet.columns[0], axis=1, inplace=True)
# Y
dataResAbx.drop(dataResAbx.columns[0], axis=1, inplace=True)
dataResMic.drop(dataResMic.columns[0], axis=1, inplace=True)

# Add underscore labels to the feature data _Pharm, _WCDoH, _Vet

def renameCols(dfname, name):
    for i in range(1, len(dfname.columns)):
        dfname = dfname.rename(columns={dfname.columns[i]: str(dfname.columns[i] + '_' + name)})
    return dfname

dataPharm = renameCols(dataPharm, 'Pvt_Ph')
dataWCDoH = renameCols(dataWCDoH, 'WCDoH')
dataVet = renameCols(dataVet, 'Vet')
#dataAbx = renameCols(dataAbx, 'Abx')
#dataPPCP = renameCols(dataPPCP, 'PPCP')
#dataGenes = renameCols(dataGenes, 'Gene')

# -------------------------------------------------
# Moving window mean
#---------------------------------------------------

days = 30

dataAbxwindow = pd.DataFrame()
dataAbxwindow[dataAbx.columns[0]] = dataAbx[dataAbx.columns[0]]
for j in dataAbx.columns[1:]:
    dataAbxwindow[j] = dataAbx[j].iloc[:].rolling(window=days).mean()  # simple moving average
dataAbxwindow.fillna(0, inplace=True)

dataPPCPwindow = pd.DataFrame()
dataPPCPwindow[dataPPCP.columns[0]] = dataPPCP[dataPPCP.columns[0]]
for j in dataPPCP.columns[1:]:
    dataPPCPwindow[j] = dataPPCP[j].iloc[:].rolling(window=days).mean()  # simple moving average
dataPPCPwindow.fillna(0, inplace=True)

dataGeneswindow = pd.DataFrame()
dataGeneswindow[dataGenes.columns[0]] = dataGenes[dataGenes.columns[0]]
for j in dataGenes.columns[1:]:
    dataGeneswindow[j] = dataGenes[j].iloc[:].rolling(window=days).mean()
dataGeneswindow.fillna(0, inplace=True)

dataPharmwindow = pd.DataFrame()
dataPharmwindow[dataPharm.columns[0]] = dataPharm[dataPharm.columns[0]]
for j in dataPharm.columns[1:]:
    dataPharmwindow[j] = dataPharm[j].iloc[:].rolling(window=days).mean()
dataPharmwindow.fillna(0, inplace=True)

dataWCDoHwindow = pd.DataFrame()
dataWCDoHwindow[dataWCDoH.columns[0]] = dataWCDoH[dataWCDoH.columns[0]]
for j in dataWCDoH.columns[1:]:
    dataWCDoHwindow[j] = dataWCDoH[j].iloc[:].rolling(window=days).mean()
dataWCDoHwindow.fillna(0, inplace=True)

dataVetwindow = pd.DataFrame()
dataVetwindow[dataVet.columns[0]] = dataVet[dataVet.columns[0]]
for j in dataVet.columns[1:]:
    dataVetwindow[j] = dataVet[j].iloc[:].rolling(window=days).mean()
dataVetwindow.fillna(0, inplace=True)

resdays = 30

dataResAbx.fillna(0, inplace=True)
dataResAbx.replace('None', 0.0, inplace=True)
dataResAbxwindow = pd.DataFrame()
dataResAbxwindowBefore = pd.DataFrame()
dataResAbxwindow[dataResAbx.columns[0]] = dataResAbx[dataResAbx.columns[0]]
for j in dataResAbx.columns[1:]:
    dataResAbxwindow[j] = dataResAbx[j].iloc[:].rolling(window=resdays).mean()
dataResAbxwindowBefore = dataResAbxwindow.copy()
dataResAbxBefore = dataResAbx.copy()

dataResMic.fillna(0, inplace=True)
dataResMic.replace('None', 0.0, inplace=True)
dataResMicwindow = pd.DataFrame()
dataResMicwindowBefore = pd.DataFrame()
dataResMicwindow[dataResMic.columns[0]] = dataResMic[dataResMic.columns[0]]
for j in dataResMic.columns[1:]:
    dataResMicwindow[j] = dataResMic[j].iloc[:].rolling(window=resdays).mean()
dataResMicwindowBefore = dataResMicwindow.copy()
dataResMicBefore = dataResMic.copy()

dataResIN.fillna(0, inplace=True)
dataResIN.replace('None', 0.0, inplace=True)
dataResINwindow = pd.DataFrame()
dataResINwindowBefore = pd.DataFrame()
dataResINwindow[dataResIN.columns[0]] = dataResIN[dataResIN.columns[0]]
for j in dataResIN.columns[1:]:
    dataResINwindow[j] = dataResIN[j].iloc[:].rolling(window=resdays).mean()
dataResINwindowBefore = dataResINwindow.copy()
dataResINBefore = dataResIN.copy()

dataResOUT.fillna(0, inplace=True)
dataResOUT.replace('None', 0.0, inplace=True)
dataResOUTwindow = pd.DataFrame()
dataResOUTwindowBefore = pd.DataFrame()
dataResOUTwindow[dataResOUT.columns[0]] = dataResOUT[dataResOUT.columns[0]]
for j in dataResOUT.columns[1:]:
    dataResOUTwindow[j] = dataResOUT[j].iloc[:].rolling(window=resdays).mean()
dataResOUTwindowBefore = dataResOUTwindow.copy()
dataResOUTBefore = dataResOUT.copy()

# Make sure that all dataframes have the date range 2018-07-18 to 2019-05-23
from dateutil.relativedelta import relativedelta

start_date = pd.to_datetime('2018-07-18') + relativedelta(days=days)
end_date = pd.to_datetime('2019-05-23')

# X
mask = (dataAbxwindow['Date'] >= start_date) & (dataAbxwindow['Date'] <= end_date)
dataAbx = dataAbxwindow.loc[mask]
mask = (dataPPCPwindow['Date'] >= start_date) & (dataPPCPwindow['Date'] <= end_date)
dataPPCP = dataPPCPwindow.loc[mask]
mask = (dataGeneswindow['Date'] >= start_date) & (dataGeneswindow['Date'] <= end_date)
dataGenes = dataGeneswindow.loc[mask]

mask = (dataPharmwindow['Date'] >= start_date) & (dataPharmwindow['Date'] <= end_date)
dataPharm = dataPharmwindow.loc[mask]
mask = (dataWCDoHwindow['Date'] >= start_date) & (dataWCDoHwindow['Date'] <= end_date)
dataWCDoH = dataWCDoHwindow.loc[mask]
mask = (dataVetwindow['Date'] >= start_date) & (dataVetwindow['Date'] <= end_date)
dataVet = dataVetwindow.loc[mask]
# Y
mask = (dataResAbx['Date'] >= start_date) & (dataResAbx['Date'] <= end_date)
dataResAbx = dataResAbxwindow.loc[mask]
mask = (dataResMic['Date'] >= start_date) & (dataResMic['Date'] <= end_date)
dataResMic = dataResMicwindow.loc[mask]
# Y
mask = (dataResIN['Date'] >= start_date) & (dataResIN['Date'] <= end_date)
dataResIN = dataResINwindow.loc[mask]
mask = (dataResOUT['Date'] >= start_date) & (dataResOUT['Date'] <= end_date)
dataResOUT = dataResOUTwindow.loc[mask]

# Add day before to features
start_dateBefore = pd.to_datetime('2018-07-17') + relativedelta(days=days)
end_dateBefore = pd.to_datetime('2019-05-22')

mask = (dataResAbxBefore['Date'] >= start_dateBefore) & (dataResAbxBefore['Date'] <= end_dateBefore)
dataResAbxBefore = dataResAbxwindowBefore.loc[mask]
mask = (dataResMicBefore['Date'] >= start_dateBefore) & (dataResMicBefore['Date'] <= end_dateBefore)
dataResMicBefore = dataResMicwindowBefore.loc[mask]
mask = (dataResINBefore['Date'] >= start_dateBefore) & (dataResINBefore['Date'] <= end_dateBefore)
dataResINBefore = dataResINwindowBefore.loc[mask]
mask = (dataResOUTBefore['Date'] >= start_dateBefore) & (dataResOUTBefore['Date'] <= end_dateBefore)
dataResOUTBefore = dataResOUTwindowBefore.loc[mask]

# Reset the index
# X
dataAbx.reset_index(drop=True, inplace=True)
dataPPCP.reset_index(drop=True, inplace=True)
dataGenes.reset_index(drop=True, inplace=True)

dataPharm.reset_index(drop=True, inplace=True)
dataWCDoH.reset_index(drop=True, inplace=True)
dataVet.reset_index(drop=True, inplace=True)
# Y
dataResAbx.reset_index(drop=True, inplace=True)
dataResMic.reset_index(drop=True, inplace=True)
dataResAbxBefore.reset_index(drop=True, inplace=True)
dataResMicBefore.reset_index(drop=True, inplace=True)
# Y
dataResIN.reset_index(drop=True, inplace=True)
dataResOUT.reset_index(drop=True, inplace=True)
dataResINBefore.reset_index(drop=True, inplace=True)
dataResOUTBefore.reset_index(drop=True, inplace=True)

##--------------------------------------------------------
# Split the datasets into training & validation sets
#---------------------------------------------------------
# Remove the last month's data to keep as a testing set

minval = 23

#dataResAbxTest = dataResAbx.iloc[-23:, :]
dataResAbxTest = dataResAbx.iloc[:, :]
dataResAbx = dataResAbx.iloc[:-minval, :]

#dataResMicTest = dataResMic.iloc[-23:, :]
dataResMicTest = dataResMic.iloc[:, :]
dataResMic = dataResMic.iloc[:-minval, :]

#dataAbxTest = dataAbx.iloc[-23:, :]
dataAbxTest = dataAbx.iloc[:, :]
dataAbx = dataAbx.iloc[:-minval, :]

#dataPPCPTest = dataPPCP.iloc[-23:, :]
dataPPCPTest = dataPPCP.iloc[:, :]
dataPPCP = dataPPCP.iloc[:-minval, :]

#dataGenesTest = dataGenes.iloc[-23:, :]
dataGenesTest = dataGenes.iloc[:, :]
dataGenes = dataGenes.iloc[:-minval, :]

#dataPharmTest = dataPharm.iloc[-23:, :]
dataPharmTest = dataPharm.iloc[:, :]
dataPharm = dataPharm.iloc[:-minval, :]

#dataWCDoHTest = dataWCDoH.iloc[-23:, :]
dataWCDoHTest = dataWCDoH.iloc[:, :]
dataWCDoH = dataWCDoH.iloc[:-minval, :]

#dataVetTest = dataVet.iloc[-23:, :]
dataVetTest = dataVet.iloc[:, :]
dataVet = dataVet.iloc[:-minval, :]

#dataResAbxTest = dataResAbx.iloc[-23:, :]
dataResINTest = dataResIN.iloc[:, :]
dataResIN = dataResIN.iloc[:-minval, :]

#dataResMicTest = dataResMic.iloc[-23:, :]
dataResOUTTest = dataResOUT.iloc[:, :]
dataResOUT = dataResOUT.iloc[:-minval, :]

print(dataResMicTest)

##  Number of Antibiotics and Microbes in the Y data

print('Number of Antibiotics in Y data:')
print(len(dataResAbx.columns.tolist())-1) # -1 to subtract the date column
print('')
print('Number of Microbes in Y data:')
print(len(dataResMic.columns.tolist())-1)

abxlist = dataResAbx.columns[1:].tolist()
miclist = dataResMic.columns[1:].tolist()

inlist = dataResIN.columns[1:].tolist()
outlist = dataResOUT.columns[1:].tolist()

print(miclist)

plotdf = pd.DataFrame()

##---------------------------------------------------------
#
#---------------------------------------------------------

plot_Acine = pd.DataFrame()
plot_Camp = pd.DataFrame()
plot_EntSpp = pd.DataFrame()
plot_EntFae = pd.DataFrame()
plot_Ecoli = pd.DataFrame()
plot_Haem = pd.DataFrame()
plot_Kleb = pd.DataFrame()
plot_Morg = pd.DataFrame()
plot_Prot = pd.DataFrame()
plot_Prov = pd.DataFrame()
plot_Pseudo = pd.DataFrame()
plot_Sal = pd.DataFrame()
plot_Ser = pd.DataFrame()
plot_Shi = pd.DataFrame()
plot_Staph = pd.DataFrame()
plot_Strep = pd.DataFrame()

plot_AcineTest = pd.DataFrame()
plot_CampTest = pd.DataFrame()
plot_EntSppTest = pd.DataFrame()
plot_EntFaeTest = pd.DataFrame()
plot_EcoliTest = pd.DataFrame()
plot_HaemTest = pd.DataFrame()
plot_KlebTest = pd.DataFrame()
plot_MorgTest = pd.DataFrame()
plot_ProtTest = pd.DataFrame()
plot_ProvTest = pd.DataFrame()
plot_PseudoTest = pd.DataFrame()
plot_SalTest = pd.DataFrame()
plot_SerTest = pd.DataFrame()
plot_ShiTest = pd.DataFrame()
plot_StaphTest = pd.DataFrame()
plot_StrepTest = pd.DataFrame()

def calcRFE(response, inputdf, term):
    df = inputdf[response]
    tempYdf = df[df != 'None']

    if len(tempYdf) >= 5:
        print(len(tempYdf))
        indexlist = tempYdf.index.tolist()
        tempXdf = Xdf.iloc[indexlist, 0:]
        y = tempYdf.to_numpy().reshape((-1, 1))

        X = tempXdf
        model = linear_model.PoissonRegressor(alpha=0).fit(X, y)
        numberOfFeatures = 1  # 100
        rfe = RFE(model, n_features_to_select=numberOfFeatures)
        fit = rfe.fit(X, y)

        df = pd.DataFrame()
        df['Features'] = X.columns
        df['Ranking'] = fit.ranking_
        df = df.sort_values('Ranking')
        df.to_csv('Output/' + str(folder) + '/Recursive FE_' + str(response) + '.csv', index=False)


def modelFit(response, inputdf, inputdfTest, term):
    global plot_Acine, plot_Camp, plot_EntSpp, plot_EntFae, plot_Ecoli, plot_Haem, plot_Kleb, plot_Morg, plot_Prot, plot_Pseudo, plot_Sal
    global plot_Ser, plot_Shi, plot_Staph, plot_Strep, plot_Prov
    global plot_AcineTest, plot_CampTest, plot_EntSppTest, plot_EntFaeTest, plot_EcoliTest, plot_HaemTest, plot_KlebTest, plot_MorgTest
    global plot_ProtTest, plot_PseudoTest, plot_SalTest, plot_SerTest, plot_ShiTest, plot_StaphTest, plot_StrepTest, plot_ProvTest
    numoffeaList = np.arange(1, 21, 1)
    print(response)
    df = inputdf[response]
    samplesize = sum(df)
    tempYdf = df[df != 'None']
    indexlist = tempYdf.index.tolist()
    y = tempYdf.to_numpy()
    y = y.astype(np.float)

    imp = pd.read_csv("Output/" + str(folder) + "/Recursive FE_" + str(response) + ".csv", sep=',')
    importdf = imp.head(8)
    features = importdf['Features'].tolist()
    tempXdf = Xdf.iloc[indexlist, 0:]
    tempXdf = tempXdf[features]
    X = tempXdf.to_numpy()
    X = X.astype(np.float)
    param_grid = {'alpha': [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]}
    model = linear_model.PoissonRegressor(max_iter=5000)
    search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', return_train_score=True, cv=TimeSeriesSplit(n_splits=5))
    search.fit(X, y)
    y_pred = search.predict(X)
    y_pred[:] = [number if number > 0 else 0.0001 for number in y_pred]
    r2Score = r2_score(y, y_pred)

    # Test the model
    dfTest = inputdfTest[response]
    tempYdfTest = dfTest[dfTest != 'None']
    indexlistTest = tempYdfTest.index.tolist()
    yTest = tempYdfTest.to_numpy()
    yTest = yTest.astype(np.float)

    # tempXdfTest = XdfTest.iloc[indexlistTest, 0:]
    tempXdfTest = XdfTest.iloc[range(len(inputdfTest)), 0:]
    tempXdfTest = tempXdfTest[features]
    XTest = tempXdfTest.to_numpy()
    XTest = XTest.astype(np.float)

    y_predTest = search.predict(XTest)
    y_predTest[:] = [number if number > 0 else 0.0001 for number in y_predTest]
    r2ScoreTest = r2_score(yTest, y_predTest)

    sigma = np.array([math.sqrt(number / (10 * len(y_pred) + 1)) for number in y_pred])  # λ ±1.96 * sqrt(λ / n)
    xCI = np.arange(1, len(y_pred) + 1, 1)
    yCImin = np.asarray([y_pred - 1.9600 * sigma]).reshape(-1)
    yCImax = np.asarray([y_pred + 1.9600 * sigma]).reshape(-1)
    plotdf = pd.DataFrame()
    plotdf['DataPoints'] = y
    plotdf[str(term)] = y_pred
    plotdf['xCI_' + str(term)] = xCI
    plotdf['yCImin_' + str(term)] = yCImin
    plotdf['yCImax_' + str(term)] = yCImax

    sigmaTest = np.array(
        [math.sqrt(number / (10 * len(y_predTest) + 1)) for number in y_predTest])  # λ ±1.96 * sqrt(λ / n)
    xCITest = np.arange(1, len(y_predTest) + 1, 1)
    yCIminTest = np.asarray([y_predTest - 1.9600 * sigmaTest]).reshape(-1)
    yCImaxTest = np.asarray([y_predTest + 1.9600 * sigmaTest]).reshape(-1)
    plotdfTest = pd.DataFrame()
    plotdfTest['DataPoints'] = yTest
    plotdfTest[str(term)] = y_predTest
    plotdfTest['xCITest_' + str(term)] = xCITest
    plotdfTest['yCIminTest_' + str(term)] = yCIminTest
    plotdfTest['yCImaxTest_' + str(term)] = yCImaxTest

    if response == 'Acinetobacter baumanii':
        plot_Acine = pd.concat([plot_Acine, plotdf], axis=1)
        plot_AcineTest = pd.concat([plot_AcineTest, plotdfTest], axis=1)
    elif response == 'Campylobacter spp.':
        plot_Camp = pd.concat([plot_Camp, plotdf], axis=1)
        plot_CampTest = pd.concat([plot_CampTest, plotdfTest], axis=1)
    elif response == 'Enterobacter spp.':
        plot_EntSpp = pd.concat([plot_EntSpp, plotdf], axis=1)
        plot_EntSppTest = pd.concat([plot_EntSppTest, plotdfTest], axis=1)
    elif response == 'Enterococcus faecium':
        plot_EntFae = pd.concat([plot_EntFae, plotdf], axis=1)
        plot_EntFaeTest = pd.concat([plot_EntFaeTest, plotdfTest], axis=1)
    elif response == 'Escherichia coli':
        plot_Ecoli = pd.concat([plot_Ecoli, plotdf], axis=1)
        plot_EcoliTest = pd.concat([plot_EcoliTest, plotdfTest], axis=1)
    elif response == 'Haemophilus influenzae':
        plot_Haem = pd.concat([plot_Haem, plotdf], axis=1)
        plot_HaemTest = pd.concat([plot_HaemTest, plotdfTest], axis=1)
    elif response == 'Klebsiella pneumoniae':
        plot_Kleb = pd.concat([plot_Kleb, plotdf], axis=1)
        plot_KlebTest = pd.concat([plot_KlebTest, plotdfTest], axis=1)
    elif response == 'Morganella morganii':
        plot_Morg = pd.concat([plot_Morg, plotdf], axis=1)
        plot_MorgTest = pd.concat([plot_MorgTest, plotdfTest], axis=1)
    elif response == 'Proteus mirabilis':
        plot_Prot = pd.concat([plot_Prot, plotdf], axis=1)
        plot_ProtTest = pd.concat([plot_ProtTest, plotdfTest], axis=1)
    elif response == 'Providencia spp.':
        plot_Prov = pd.concat([plot_Prov, plotdf], axis=1)
        plot_ProvTest = pd.concat([plot_ProvTest, plotdfTest], axis=1)
    elif response == 'Pseudomonas aeruginosa':
        plot_Pseudo = pd.concat([plot_Pseudo, plotdf], axis=1)
        plot_PseudoTest = pd.concat([plot_PseudoTest, plotdfTest], axis=1)
    elif response == 'Salmonella spp.':
        plot_Sal = pd.concat([plot_Sal, plotdf], axis=1)
        plot_SalTest = pd.concat([plot_SalTest, plotdfTest], axis=1)
    elif response == 'Serratia spp.':
        plot_Ser = pd.concat([plot_Ser, plotdf], axis=1)
        plot_SerTest = pd.concat([plot_SerTest, plotdfTest], axis=1)
    elif response == 'Shigella spp.':
        plot_Shi = pd.concat([plot_Shi, plotdf], axis=1)
        plot_ShiTest = pd.concat([plot_ShiTest, plotdfTest], axis=1)
    elif response == 'Staphylococcus aureus':
        plot_Staph = pd.concat([plot_Staph, plotdf], axis=1)
        plot_StaphTest = pd.concat([plot_StaphTest, plotdfTest], axis=1)
    elif response == 'Streptococcus pneumoniae':
        plot_Strep = pd.concat([plot_Strep, plotdf], axis=1)
        plot_StrepTest = pd.concat([plot_StrepTest, plotdfTest], axis=1)

    print(r2Score)
    print(r2ScoreTest)
    print('')

    lab1 = str("$CV-R^2$ = " + str("%.3f" % r2_score(y, y_pred)))
    lab2 = str("$CV-R^2$ = " + str("%.3f" % r2_score(yTest, y_predTest)))
    plt.ioff()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    plt.scatter(range(0, len(yTest[:-22])), yTest[:-22], label='True AMR incidence', s=15, c='k', marker='x')
    plt.plot(range(0, len(y_predTest[:-22])), y_predTest[:-22], label='Predicted AMR incidence (' + lab1 + ')', ls='-',
             c='tab:blue', lw=2)
    plt.fill_between(range(0, len(y_predTest[:-22])), yCIminTest[:-22], yCImaxTest[:-22], alpha=.3, ec='None',
                     label='95% Confidence Interval')

    plt.scatter(range(257, 257 + len(yTest[-23:])), yTest[-23:], s=15, c='red', marker='x')
    plt.plot(range(257, 257 + len(y_predTest[-23:])), y_predTest[-23:],
             label='Predicted AMR incidence TEST (' + lab2 + ')', ls='-', c='tab:red', lw=2)
    plt.fill_between(range(257, 257 + len(yTest[-23:])), yCIminTest[-23:], yCImaxTest[-23:], alpha=.3, ec='None')

    ax.set_xlabel('Date', fontsize=18)
    ax.set_xticks([-25, 35, 95, 160, 215, 270])
    ax.set_xticklabels(['Jul 2018', 'Sep 2018', 'Nov 2018', 'Jan 2019', 'Mar 2019', 'May 2019'], fontsize=14)
    ax.set_ylabel('Scaled daily load', fontsize=18)
    plt.suptitle(str(response) + '\n', fontsize=22) #plt.suptitle(response + '\n', fontsize=16)
    ax.legend(loc='upper right', fontsize=14)  # loc='upper left'
    ax.yaxis.set_tick_params(labelsize=14)

    leg = ax.get_legend()
    leg.legendHandles[3].set_color('tab:gray')

    plt.savefig('Output/' + str(folder) + '/' + str(response) + '.png')

codes = pd.read_csv("Data/Antibiotics and microbes.csv", sep=',')
codes.drop_duplicates(inplace=True)
print(codes.tail())

def renameFS(response):
    imp = pd.read_csv("Output/" + str(folder) + "/Recursive FE_" + str(response) + ".csv", sep=',')
    imp = pd.DataFrame(imp)
    importdf = imp.head(25)
    features = importdf['Features'].tolist()

    features = [s.replace('_site1', ' (Site 1)') for s in features]
    features = [s.replace('_site2', ' (Site 2)') for s in features]
    features = [s.replace('_site3', ' (Site 3)') for s in features]
    features = [s.replace('_site4', ' (Site 4)') for s in features]
    features = [s.replace('_site5', ' (Site 5)') for s in features]
    features = [s.replace('_site6', ' (Site 6)') for s in features]
    features = [s.replace('_site7', ' (Site 7)') for s in features]
    features = [s.replace('_site8', ' (Site 8)') for s in features]
    features = [s.replace('_site9', ' (Site 9)') for s in features]
    features = [s.replace('_WCDoH', ' (WCDoH)') for s in features]
    features = [s.replace('_Pvt_Ph', ' (Pharm)') for s in features]
    features = [s.replace('_Vet', ' (Vet)') for s in features]

    featuresdf = pd.DataFrame(features)

    features = [s.replace(' (Site 1)', '') for s in features]
    features = [s.replace(' (Site 2)', '') for s in features]
    features = [s.replace(' (Site 3)', '') for s in features]
    features = [s.replace(' (Site 4)', '') for s in features]
    features = [s.replace(' (Site 5)', '') for s in features]
    features = [s.replace(' (Site 6)', '') for s in features]
    features = [s.replace(' (Site 7)', '') for s in features]
    features = [s.replace(' (Site 8)', '') for s in features]
    features = [s.replace(' (Site 9)', '') for s in features]
    features = [s.replace(' (WCDoH)', '') for s in features]
    features = [s.replace(' (Pharm)', '') for s in features]
    features = [s.replace(' (Vet)', '') for s in features]
    features = [s.rstrip() for s in features]

    featuresdf['Chemical'] = features
    featuresdf = pd.merge(featuresdf, codes, on='Chemical', how='left')
    featuresdf.drop('Chemical', inplace=True, axis=1)
    print(featuresdf)

    featuresdf.to_csv('Output/' + str(folder) + '/Paper_Recursive FE_' + str(response) + '.csv', index=False)

## -----------------------------------------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------------------------------------------

responses = ['Acinetobacter baumanii', 'Campylobacter spp.', 'Enterobacter spp.', 'Enterococcus faecium', 'Escherichia coli',
             'Haemophilus influenzae', 'Klebsiella pneumoniae', 'Morganella morganii', 'Proteus mirabilis', 'Providencia spp.',
             'Pseudomonas aeruginosa', 'Salmonella spp.', 'Serratia spp.', 'Shigella spp.', 'Staphylococcus aureus', 'Streptococcus pneumoniae']

for response in responses:
    scaler = MinMaxScaler(feature_range=(0.,1.))

    Xdf = pd.concat([dataAbx], axis=1)
    df1 = Xdf.filter(regex='site2')
    df2 = Xdf.filter(regex='site4')
    df3 = Xdf.filter(regex='site6')
    Xdf = pd.concat([df1, df2, df3], axis=1)
    Xdf.iloc[:,:] = scaler.fit_transform(Xdf.iloc[:,:])

    XdfTest = pd.concat([dataAbxTest.iloc[:,1:]], axis=1)
    df1 = XdfTest.filter(regex='site2')
    df2 = XdfTest.filter(regex='site4')
    df3 = XdfTest.filter(regex='site6')
    XdfTest = pd.concat([df1, df2, df3], axis=1)
    XdfTest.iloc[:,0:] = scaler.fit_transform(XdfTest.iloc[:,0:])

    folder = 'Individual microbes/All'
    calcRFE(response, dataResMic, 'All')
    modelFit(response, dataResMic, dataResMicTest, 'All')
    renameFS(response)



##--------------------------------------------------------

#---------------------------------------------------------

folder1 = 'Individual microbes'

plot_Acine = pd.DataFrame()
plot_Camp = pd.DataFrame()
plot_EntSpp = pd.DataFrame()
plot_EntFae = pd.DataFrame()
plot_Ecoli = pd.DataFrame()
plot_Haem = pd.DataFrame()
plot_Kleb = pd.DataFrame()
plot_Morg = pd.DataFrame()
plot_Prot = pd.DataFrame()
plot_Prov = pd.DataFrame()
plot_Pseudo = pd.DataFrame()
plot_Sal = pd.DataFrame()
plot_Ser = pd.DataFrame()
plot_Shi = pd.DataFrame()
plot_Staph = pd.DataFrame()
plot_Strep = pd.DataFrame()

plot_AcineTest = pd.DataFrame()
plot_CampTest = pd.DataFrame()
plot_EntSppTest = pd.DataFrame()
plot_EntFaeTest = pd.DataFrame()
plot_EcoliTest = pd.DataFrame()
plot_HaemTest = pd.DataFrame()
plot_KlebTest = pd.DataFrame()
plot_MorgTest = pd.DataFrame()
plot_ProtTest = pd.DataFrame()
plot_ProvTest = pd.DataFrame()
plot_PseudoTest = pd.DataFrame()
plot_SalTest = pd.DataFrame()
plot_SerTest = pd.DataFrame()
plot_ShiTest = pd.DataFrame()
plot_StaphTest = pd.DataFrame()
plot_StrepTest = pd.DataFrame()

def calcRFE(response, dataSet, dataTest, foldername):
    dataResMic = dataSet
    dataResMicTest = dataTest
    folder2 = foldername
    df = dataResMic[response]
    tempYdf = df[df != 'None']

    if len(tempYdf) >= 5:
        print(len(tempYdf))
        indexlist = tempYdf.index.tolist()
        tempXdf = Xdf.iloc[indexlist, 0:]
        y = tempYdf.to_numpy().reshape((-1, 1))

        X = tempXdf
        model = linear_model.PoissonRegressor(alpha=0).fit(X, y)
        numberOfFeatures = 1  # 100
        rfe = RFE(model, n_features_to_select=numberOfFeatures)
        fit = rfe.fit(X, y)

        df = pd.DataFrame()
        df['Features'] = X.columns
        df['Ranking'] = fit.ranking_
        df = df.sort_values('Ranking')
        df.to_csv('Output/' + str(folder1) + '/' + str(folder2) + '/Recursive FE_' + str(response) + '.csv', index=False)


def modelFit(response, dataSet, dataTest, foldername):
    dataResMic = dataSet
    dataResMicTest = dataTest
    folder2 = foldername
    global plot_Acine, plot_Camp, plot_EntSpp, plot_EntFae, plot_Ecoli, plot_Haem, plot_Kleb, plot_Morg, plot_Prot, plot_Pseudo, plot_Sal
    global plot_Ser, plot_Shi, plot_Staph, plot_Strep, plot_Prov
    global plot_AcineTest, plot_CampTest, plot_EntSppTest, plot_EntFaeTest, plot_EcoliTest, plot_HaemTest, plot_KlebTest, plot_MorgTest
    global plot_ProtTest, plot_PseudoTest, plot_SalTest, plot_SerTest, plot_ShiTest, plot_StaphTest, plot_StrepTest, plot_ProvTest
    numoffeaList = np.arange(1, 21, 1)
    print(response)
    df = dataResMic[response]
    tempYdf = df[df != 'None']
    indexlist = tempYdf.index.tolist()
    y = tempYdf.to_numpy()
    y = y.astype(np.float)

    optnumfea = 8
    # Optimal feature num and alpha fit
    imp = pd.read_csv("Output/" + str(folder1) + '/' + str(folder2) + "/Recursive FE_" + str(response) + ".csv", sep=',')
    imp = pd.DataFrame(imp)
    importdf = imp.head(optnumfea)
    features = importdf['Features'].tolist()
    tempXdf = Xdf.iloc[indexlist, 0:]
    tempXdf = tempXdf[features]
    X = tempXdf.to_numpy()
    X = X.astype(np.float)

    param_grid = {'alpha': [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]}
    model = linear_model.PoissonRegressor(max_iter=5000)
    search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', return_train_score=True, cv=TimeSeriesSplit(n_splits=5))
    search.fit(X, y)
    y_pred = search.predict(X)
    y_pred[:] = [number if number > 0 else 0.0001 for number in y_pred]
    r2Score = r2_score(y, y_pred)

    # Test the model
    dfTest = dataResMicTest[response]
    tempYdfTest = dfTest[dfTest != 'None']
    indexlistTest = tempYdfTest.index.tolist()
    yTest = tempYdfTest.to_numpy()
    yTest = yTest.astype(np.float)

    tempXdfTest = XdfTest.iloc[range(len(dataResMicTest)), 0:]
    tempXdfTest = tempXdfTest[features]
    XTest = tempXdfTest.to_numpy()
    XTest = XTest.astype(np.float)

    y_predTest = search.predict(XTest)
    y_predTest[:] = [number if number > 0 else 0.0001 for number in y_predTest]
    r2ScoreTest = r2_score(yTest, y_predTest)

    sigma = np.array([math.sqrt(number / (10 * len(y_pred) + 1)) for number in y_pred])  # λ ±1.96 * sqrt(λ / n)
    xCI = np.arange(1, len(y_pred) + 1, 1)
    yCImin = np.asarray([y_pred - 1.9600 * sigma]).reshape(-1)
    yCImax = np.asarray([y_pred + 1.9600 * sigma]).reshape(-1)
    plotdf = pd.DataFrame()
    plotdf['DataPoints'] = y
    plotdf['AllFeatures'] = y_pred
    plotdf['xCI'] = xCI
    plotdf['yCImin'] = yCImin
    plotdf['yCImax'] = yCImax

    sigmaTest = np.array([math.sqrt(number / (10 * len(y_predTest) + 1)) for number in y_predTest])  # λ ±1.96 * sqrt(λ / n)
    xCITest = np.arange(1, len(y_predTest) + 1, 1)
    yCIminTest = np.asarray([y_predTest - 1.9600 * sigmaTest]).reshape(-1)
    yCImaxTest = np.asarray([y_predTest + 1.9600 * sigmaTest]).reshape(-1)
    plotdfTest = pd.DataFrame()
    plotdfTest['DataPoints'] = yTest
    plotdfTest['AllFeatures'] = y_predTest
    plotdfTest['xCITest'] = xCITest
    plotdfTest['yCIminTest'] = yCIminTest
    plotdfTest['yCImaxTest'] = yCImaxTest

    if response == 'Acinetobacter baumanii':
        plot_Acine = pd.concat([plot_Acine, plotdf], axis=1)
        plot_AcineTest = pd.concat([plot_AcineTest, plotdfTest], axis=1)
    elif response == 'Campylobacter spp.':
        plot_Camp = pd.concat([plot_Camp, plotdf], axis=1)
        plot_CampTest = pd.concat([plot_CampTest, plotdfTest], axis=1)
    elif response == 'Enterobacter spp.':
        plot_EntSpp = pd.concat([plot_EntSpp, plotdf], axis=1)
        plot_EntSppTest = pd.concat([plot_EntSppTest, plotdfTest], axis=1)
    elif response == 'Enterococcus faecium':
        plot_EntFae = pd.concat([plot_EntFae, plotdf], axis=1)
        plot_EntFaeTest = pd.concat([plot_EntFaeTest, plotdfTest], axis=1)
    elif response == 'Escherichia coli':
        plot_Ecoli = pd.concat([plot_Ecoli, plotdf], axis=1)
        plot_EcoliTest = pd.concat([plot_EcoliTest, plotdfTest], axis=1)
    elif response == 'Haemophilus influenzae':
        plot_Haem = pd.concat([plot_Haem, plotdf], axis=1)
        plot_HaemTest = pd.concat([plot_HaemTest, plotdfTest], axis=1)
    elif response == 'Klebsiella pneumoniae':
        plot_Kleb = pd.concat([plot_Kleb, plotdf], axis=1)
        plot_KlebTest = pd.concat([plot_KlebTest, plotdfTest], axis=1)
    elif response == 'Morganella morganii':
        plot_Morg = pd.concat([plot_Morg, plotdf], axis=1)
        plot_MorgTest = pd.concat([plot_MorgTest, plotdfTest], axis=1)
    elif response == 'Proteus mirabilis':
        plot_Prot = pd.concat([plot_Prot, plotdf], axis=1)
        plot_ProtTest = pd.concat([plot_ProtTest, plotdfTest], axis=1)
    elif response == 'Providencia spp.':
        plot_Prov = pd.concat([plot_Prov, plotdf], axis=1)
        plot_ProvTest = pd.concat([plot_ProvTest, plotdfTest], axis=1)
    elif response == 'Pseudomonas aeruginosa':
        plot_Pseudo = pd.concat([plot_Pseudo, plotdf], axis=1)
        plot_PseudoTest = pd.concat([plot_PseudoTest, plotdfTest], axis=1)
    elif response == 'Salmonella spp.':
        plot_Sal = pd.concat([plot_Sal, plotdf], axis=1)
        plot_SalTest = pd.concat([plot_SalTest, plotdfTest], axis=1)
    elif response == 'Serratia spp.':
        plot_Ser = pd.concat([plot_Ser, plotdf], axis=1)
        plot_SerTest = pd.concat([plot_SerTest, plotdfTest], axis=1)
    elif response == 'Shigella spp.':
        plot_Shi = pd.concat([plot_Shi, plotdf], axis=1)
        plot_ShiTest = pd.concat([plot_ShiTest, plotdfTest], axis=1)
    elif response == 'Staphylococcus aureus':
        plot_Staph = pd.concat([plot_Staph, plotdf], axis=1)
        plot_StaphTest = pd.concat([plot_StaphTest, plotdfTest], axis=1)
    elif response == 'Streptococcus pneumoniae':
        plot_Strep = pd.concat([plot_Strep, plotdf], axis=1)
        plot_StrepTest = pd.concat([plot_StrepTest, plotdfTest], axis=1)

    print(r2Score)
    print(r2ScoreTest)
    print('')

    lab1 = str("$CV-R^2$ = " + str("%.3f" % r2_score(y, y_pred)))
    lab2 = str("$CV-R^2$ = " + str("%.3f" % r2_score(yTest, y_predTest)))
    plt.ioff()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    print(len(yTest))
    print(len(y_predTest))

    plt.scatter(range(0,len(yTest[:-22])), yTest[:-22], label='True AMR incidence', s=15, c='k', marker='x')
    plt.plot(range(0,len( y_predTest[:-22])), y_predTest[:-22],label='Predicted AMR incidence (' + lab1 + ')', ls='-', c='tab:blue',lw=2)
    plt.fill_between(range(0,len( y_predTest[:-22])), yCIminTest[:-22], yCImaxTest[:-22], alpha=.3, ec='None',label='95% Confidence Interval')

    strt = 257
    plt.scatter(range(strt, strt + len(yTest[-23:])), yTest[-23:], s=15, c='red', marker='x')
    plt.plot(range(strt, strt + len(y_predTest[-23:])), y_predTest[-23:],label='Predicted AMR incidence TEST (' + lab2 + ')', ls='-', c='tab:red', lw=2)
    plt.fill_between(range(strt, strt + len(yTest[-23:])), yCIminTest[-23:], yCImaxTest[-23:], alpha=.3, ec='None')

    ax.set_xlabel('Date', fontsize=18)
    ax.set_xticks([-25, 35, 95, 160, 215, 270])
    ax.set_xticklabels(['Jul 2018', 'Sep 2018', 'Nov 2018', 'Jan 2019', 'Mar 2019', 'May 2019'], fontsize=14)
    ax.set_ylabel('Scaled daily load', fontsize=18)
    plt.title(str(response), fontsize=22)
    ax.legend(loc='upper right', fontsize=14)  # loc='upper left'
    ax.yaxis.set_tick_params(labelsize=14)
    leg = ax.get_legend()
    leg.legendHandles[3].set_color('tab:gray')

    plt.savefig('Output/' + str(folder1) + '/' + str(folder2) + '/' + str(response) + '.png')

codes = pd.read_csv("Data/Antibiotics and microbes.csv", sep=',')
codes.drop_duplicates(inplace=True)
print(codes.tail())

def renameFS(response, data, dataT, f2):
    imp = pd.read_csv("Output/" + str(folder1) + '/' + str(f2) + "/Recursive FE_" + str(response) + ".csv", sep=',')
    imp = pd.DataFrame(imp)
    importdf = imp.head(25)
    features = importdf['Features'].tolist()

    features = [s.replace('_site1', ' (Site 1)') for s in features]
    features = [s.replace('_site2', ' (Site 2)') for s in features]
    features = [s.replace('_site3', ' (Site 3)') for s in features]
    features = [s.replace('_site4', ' (Site 4)') for s in features]
    features = [s.replace('_site5', ' (Site 5)') for s in features]
    features = [s.replace('_site6', ' (Site 6)') for s in features]
    features = [s.replace('_site7', ' (Site 7)') for s in features]
    features = [s.replace('_site8', ' (Site 8)') for s in features]
    features = [s.replace('_site9', ' (Site 9)') for s in features]
    features = [s.replace('_WCDoH', ' (WCDoH)') for s in features]
    features = [s.replace('_Pvt_Ph', ' (Pharm)') for s in features]
    features = [s.replace('_Vet', ' (Vet)') for s in features]

    featuresdf = pd.DataFrame(features)

    features = [s.replace(' (Site 1)', '') for s in features]
    features = [s.replace(' (Site 2)', '') for s in features]
    features = [s.replace(' (Site 3)', '') for s in features]
    features = [s.replace(' (Site 4)', '') for s in features]
    features = [s.replace(' (Site 5)', '') for s in features]
    features = [s.replace(' (Site 6)', '') for s in features]
    features = [s.replace(' (Site 7)', '') for s in features]
    features = [s.replace(' (Site 8)', '') for s in features]
    features = [s.replace(' (Site 9)', '') for s in features]
    features = [s.replace(' (WCDoH)', '') for s in features]
    features = [s.replace(' (Pharm)', '') for s in features]
    features = [s.replace(' (Vet)', '') for s in features]
    features = [s.rstrip() for s in features]

    featuresdf['Chemical'] = features
    featuresdf = pd.merge(featuresdf, codes, on='Chemical', how='left')
    featuresdf.drop('Chemical', inplace=True, axis=1)
    print(featuresdf)

    featuresdf.to_csv('Output/' + str(folder1) + '/' + str(f2) + '/Paper_Recursive FE_' + str(response) + '.csv', index=False)

## -----------------------------------------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------------------------------------------------------------

responses = ['Acinetobacter baumanii', 'Campylobacter spp.', 'Enterobacter spp.', 'Enterococcus faecium', 'Escherichia coli',
             'Haemophilus influenzae', 'Klebsiella pneumoniae', 'Morganella morganii', 'Proteus mirabilis', 'Providencia spp.',
             'Pseudomonas aeruginosa', 'Salmonella spp.', 'Serratia spp.', 'Shigella spp.', 'Staphylococcus aureus', 'Streptococcus pneumoniae']

for response in responses:
    scaler = MinMaxScaler(feature_range=(0.,1.))

    Xdf = pd.concat([dataAbx.iloc[:,1:]], axis=1)
    colList = Xdf.columns
    Xdf = pd.concat([dataAbx.iloc[:,1:]], axis=1)
    Xdf.iloc[:,0:] = scaler.fit_transform(Xdf.iloc[:,0:])

    XdfTest = pd.concat([dataAbxTest.iloc[:,1:]], axis=1)
    XdfTest.iloc[:,0:] = scaler.fit_transform(XdfTest.iloc[:,0:])

    calcRFE(response, dataResIN, dataResINTest, 'Hospital')
    modelFit(response, dataResIN, dataResINTest, 'Hospital')
    renameFS(response, dataResIN, dataResINTest, 'Hospital')

    calcRFE(response, dataResOUT, dataResOUTTest, 'Community')
    modelFit(response, dataResOUT, dataResOUTTest, 'Community')
    renameFS(response, dataResOUT, dataResOUTTest, 'Community')

##

##---------------------------------------------------

#----------------------------------------------------
plot_Acine = pd.DataFrame()
plot_Camp = pd.DataFrame()
plot_EntSpp = pd.DataFrame()
plot_EntFae = pd.DataFrame()
plot_Ecoli = pd.DataFrame()
plot_Haem = pd.DataFrame()
plot_Kleb = pd.DataFrame()
plot_Morg = pd.DataFrame()
plot_Prot = pd.DataFrame()
plot_Prov = pd.DataFrame()
plot_Pseudo = pd.DataFrame()
plot_Sal = pd.DataFrame()
plot_Ser = pd.DataFrame()
plot_Shi = pd.DataFrame()
plot_Staph = pd.DataFrame()
plot_Strep = pd.DataFrame()

plot_AcineTest = pd.DataFrame()
plot_CampTest = pd.DataFrame()
plot_EntSppTest = pd.DataFrame()
plot_EntFaeTest = pd.DataFrame()
plot_EcoliTest = pd.DataFrame()
plot_HaemTest = pd.DataFrame()
plot_KlebTest = pd.DataFrame()
plot_MorgTest = pd.DataFrame()
plot_ProtTest = pd.DataFrame()
plot_ProvTest = pd.DataFrame()
plot_PseudoTest = pd.DataFrame()
plot_SalTest = pd.DataFrame()
plot_SerTest = pd.DataFrame()
plot_ShiTest = pd.DataFrame()
plot_StaphTest = pd.DataFrame()
plot_StrepTest = pd.DataFrame()

codes = pd.read_csv("Data/Antibiotics and microbes.csv", sep=',')
codes.drop_duplicates(inplace=True)
print(codes.tail())

def modelFit(response, inputdf, inputdfTest, term):
    global plot_Acine, plot_Camp, plot_EntSpp, plot_EntFae, plot_Ecoli, plot_Haem, plot_Kleb, plot_Morg, plot_Prot, plot_Pseudo, plot_Sal
    global plot_Ser, plot_Shi, plot_Staph, plot_Strep, plot_Prov
    global plot_AcineTest, plot_CampTest, plot_EntSppTest, plot_EntFaeTest, plot_EcoliTest, plot_HaemTest, plot_KlebTest, plot_MorgTest
    global plot_ProtTest, plot_PseudoTest, plot_SalTest, plot_SerTest, plot_ShiTest, plot_StaphTest, plot_StrepTest, plot_ProvTest
    print(response)
    df = inputdf[response]
    tempYdf = df[df != 'None']
    indexlist = tempYdf.index.tolist()
    y = tempYdf.to_numpy()
    y = y.astype(np.float)

    imp = pd.read_csv("Output/" + str(folder) + "/Recursive FE_all.csv", sep=',')
    imp = pd.DataFrame(imp)
    importdf = imp.head(8)
    features = importdf['Features'].tolist()
    tempXdf = Xdf.iloc[indexlist, 0:]
    tempXdf = tempXdf[features]
    X = tempXdf.to_numpy()
    X = X.astype(np.float)

    param_grid = {'alpha': [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]}
    model = linear_model.PoissonRegressor(max_iter=5000)

    search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', return_train_score=True, cv=TimeSeriesSplit(n_splits=5))
    search.fit(X, y)
    y_pred = search.predict(X)
    y_pred[:] = [number if number > 0 else 0.0001 for number in y_pred]
    r2Score = r2_score(y, y_pred)

    # Test the model
    dfTest = inputdfTest[response]
    tempYdfTest = dfTest[dfTest != 'None']
    indexlistTest = tempYdfTest.index.tolist()
    yTest = tempYdfTest.to_numpy()
    yTest = yTest.astype(np.float)

    # tempXdfTest = XdfTest.iloc[indexlistTest, 0:]
    tempXdfTest = XdfTest.iloc[range(len(inputdfTest)), 0:]
    tempXdfTest = tempXdfTest[features]
    XTest = tempXdfTest.to_numpy()
    XTest = XTest.astype(np.float)

    y_predTest = search.predict(XTest)
    y_predTest[:] = [number if number > 0 else 0.0001 for number in y_predTest]
    r2ScoreTest = r2_score(yTest, y_predTest)

    sigma = np.array([math.sqrt(number / (10 * len(y_pred) + 1)) for number in y_pred])  # λ ±1.96 * sqrt(λ / n)
    xCI = np.arange(1, len(y_pred) + 1, 1)
    yCImin = np.asarray([y_pred - 1.9600 * sigma]).reshape(-1)
    yCImax = np.asarray([y_pred + 1.9600 * sigma]).reshape(-1)
    plotdf = pd.DataFrame()
    plotdf['DataPoints'] = y
    plotdf[str(term)] = y_pred
    plotdf['xCI_' + str(term)] = xCI
    plotdf['yCImin_' + str(term)] = yCImin
    plotdf['yCImax_' + str(term)] = yCImax

    sigmaTest = np.array([math.sqrt(number / (10 * len(y_predTest) + 1)) for number in y_predTest])  # λ ±1.96 * sqrt(λ / n)
    xCITest = np.arange(1, len(y_predTest) + 1, 1)
    yCIminTest = np.asarray([y_predTest - 1.9600 * sigmaTest]).reshape(-1)
    yCImaxTest = np.asarray([y_predTest + 1.9600 * sigmaTest]).reshape(-1)
    plotdfTest = pd.DataFrame()
    plotdfTest['DataPoints'] = yTest
    plotdfTest[str(term)] = y_predTest
    plotdfTest['xCITest_' + str(term)] = xCITest
    plotdfTest['yCIminTest_' + str(term)] = yCIminTest
    plotdfTest['yCImaxTest_' + str(term)] = yCImaxTest

    if response == 'Acinetobacter baumanii':
        plot_Acine = pd.concat([plot_Acine, plotdf], axis=1)
        plot_AcineTest = pd.concat([plot_AcineTest, plotdfTest], axis=1)
    elif response == 'Campylobacter spp.':
        plot_Camp = pd.concat([plot_Camp, plotdf], axis=1)
        plot_CampTest = pd.concat([plot_CampTest, plotdfTest], axis=1)
    elif response == 'Enterobacter spp.':
        plot_EntSpp = pd.concat([plot_EntSpp, plotdf], axis=1)
        plot_EntSppTest = pd.concat([plot_EntSppTest, plotdfTest], axis=1)
    elif response == 'Enterococcus faecium':
        plot_EntFae = pd.concat([plot_EntFae, plotdf], axis=1)
        plot_EntFaeTest = pd.concat([plot_EntFaeTest, plotdfTest], axis=1)
    elif response == 'Escherichia coli':
        plot_Ecoli = pd.concat([plot_Ecoli, plotdf], axis=1)
        plot_EcoliTest = pd.concat([plot_EcoliTest, plotdfTest], axis=1)
    elif response == 'Haemophilus influenzae':
        plot_Haem = pd.concat([plot_Haem, plotdf], axis=1)
        plot_HaemTest = pd.concat([plot_HaemTest, plotdfTest], axis=1)
    elif response == 'Klebsiella pneumoniae':
        plot_Kleb = pd.concat([plot_Kleb, plotdf], axis=1)
        plot_KlebTest = pd.concat([plot_KlebTest, plotdfTest], axis=1)
    elif response == 'Morganella morganii':
        plot_Morg = pd.concat([plot_Morg, plotdf], axis=1)
        plot_MorgTest = pd.concat([plot_MorgTest, plotdfTest], axis=1)
    elif response == 'Proteus mirabilis':
        plot_Prot = pd.concat([plot_Prot, plotdf], axis=1)
        plot_ProtTest = pd.concat([plot_ProtTest, plotdfTest], axis=1)
    elif response == 'Providencia spp.':
        plot_Prov = pd.concat([plot_Prov, plotdf], axis=1)
        plot_ProvTest = pd.concat([plot_ProvTest, plotdfTest], axis=1)
    elif response == 'Pseudomonas aeruginosa':
        plot_Pseudo = pd.concat([plot_Pseudo, plotdf], axis=1)
        plot_PseudoTest = pd.concat([plot_PseudoTest, plotdfTest], axis=1)
    elif response == 'Salmonella spp.':
        plot_Sal = pd.concat([plot_Sal, plotdf], axis=1)
        plot_SalTest = pd.concat([plot_SalTest, plotdfTest], axis=1)
    elif response == 'Serratia spp.':
        plot_Ser = pd.concat([plot_Ser, plotdf], axis=1)
        plot_SerTest = pd.concat([plot_SerTest, plotdfTest], axis=1)
    elif response == 'Shigella spp.':
        plot_Shi = pd.concat([plot_Shi, plotdf], axis=1)
        plot_ShiTest = pd.concat([plot_ShiTest, plotdfTest], axis=1)
    elif response == 'Staphylococcus aureus':
        plot_Staph = pd.concat([plot_Staph, plotdf], axis=1)
        plot_StaphTest = pd.concat([plot_StaphTest, plotdfTest], axis=1)
    elif response == 'Streptococcus pneumoniae':
        plot_Strep = pd.concat([plot_Strep, plotdf], axis=1)
        plot_StrepTest = pd.concat([plot_StrepTest, plotdfTest], axis=1)

    print(r2Score)
    print(r2ScoreTest)
    print('')

    lab1 = str("$CV-R^2$ = " + str("%.3f" % r2_score(y, y_pred)))
    lab2 = str("$CV-R^2$ = " + str("%.3f" % r2_score(yTest, y_predTest)))
    plt.ioff()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    plt.scatter(range(0, len(yTest[:-22])), yTest[:-22], label='True AMR incidence', s=15, c='k', marker='x')
    plt.plot(range(0, len(y_predTest[:-22])), y_predTest[:-22], label='Predicted AMR incidence (' + lab1 + ')', ls='-',
             c='tab:blue', lw=2)
    plt.fill_between(range(0, len(y_predTest[:-22])), yCIminTest[:-22], yCImaxTest[:-22], alpha=.3, ec='None',
                     label='95% Confidence Interval')

    plt.scatter(range(257, 257 + len(yTest[-23:])), yTest[-23:], s=15, c='red', marker='x')
    plt.plot(range(257, 257 + len(y_predTest[-23:])), y_predTest[-23:],
             label='Predicted AMR incidence TEST (' + lab2 + ')', ls='-', c='tab:red', lw=2)
    plt.fill_between(range(257, 257 + len(yTest[-23:])), yCIminTest[-23:], yCImaxTest[-23:], alpha=.3, ec='None')

    ax.set_xlabel('Date', fontsize=18)
    ax.set_xticks([-25, 35, 95, 160, 215, 270])
    ax.set_xticklabels(['Jul 2018', 'Sep 2018', 'Nov 2018', 'Jan 2019', 'Mar 2019', 'May 2019'], fontsize=14)
    ax.set_ylabel('Scaled daily load', fontsize=18)
    plt.suptitle(str(response) + '\n', fontsize=22) #plt.suptitle(response + '\n', fontsize=16)
    ax.legend(loc='upper right', fontsize=14)  # loc='upper left'
    ax.yaxis.set_tick_params(labelsize=14)

    leg = ax.get_legend()
    leg.legendHandles[3].set_color('tab:gray')

    plt.savefig('Output/' + str(folder) + '/' + str(response) + '.png')

def renameFS(response):
    imp = pd.read_csv("Output/" + str(folder) + "/Recursive FE_all.csv", sep=',')
    imp = pd.DataFrame(imp)
    importdf = imp.head(25)
    features = importdf['Features'].tolist()

    features = [s.replace('_site1', ' (Site 1)') for s in features]
    features = [s.replace('_site2', ' (Site 2)') for s in features]
    features = [s.replace('_site3', ' (Site 3)') for s in features]
    features = [s.replace('_site4', ' (Site 4)') for s in features]
    features = [s.replace('_site5', ' (Site 5)') for s in features]
    features = [s.replace('_site6', ' (Site 6)') for s in features]
    features = [s.replace('_site7', ' (Site 7)') for s in features]
    features = [s.replace('_site8', ' (Site 8)') for s in features]
    features = [s.replace('_site9', ' (Site 9)') for s in features]
    features = [s.replace('_WCDoH', ' (WCDoH)') for s in features]
    features = [s.replace('_Pvt_Ph', ' (Pharm)') for s in features]
    features = [s.replace('_Vet', ' (Vet)') for s in features]

    featuresdf = pd.DataFrame(features)

    features = [s.replace(' (Site 1)', '') for s in features]
    features = [s.replace(' (Site 2)', '') for s in features]
    features = [s.replace(' (Site 3)', '') for s in features]
    features = [s.replace(' (Site 4)', '') for s in features]
    features = [s.replace(' (Site 5)', '') for s in features]
    features = [s.replace(' (Site 6)', '') for s in features]
    features = [s.replace(' (Site 7)', '') for s in features]
    features = [s.replace(' (Site 8)', '') for s in features]
    features = [s.replace(' (Site 9)', '') for s in features]
    features = [s.replace(' (WCDoH)', '') for s in features]
    features = [s.replace(' (Pharm)', '') for s in features]
    features = [s.replace(' (Vet)', '') for s in features]
    features = [s.rstrip() for s in features]

    featuresdf['Chemical'] = features
    featuresdf = pd.merge(featuresdf, codes, on='Chemical', how='left')
    featuresdf.drop('Chemical', inplace=True, axis=1)
    print(featuresdf)

    featuresdf.to_csv('Output/' + str(folder) + '/Paper_Recursive FE_all.csv', index=False)

##----------------------------------------------------------

scaler = MinMaxScaler(feature_range=(0.,1.))

Xdf = pd.concat([dataAbx.iloc[:,1:]], axis=1)
df1 = Xdf.filter(regex='site2')
df2 = Xdf.filter(regex='site4')
df3 = Xdf.filter(regex='site6')
Xdf = pd.concat([df1, df2, df3], axis=1)
colList = Xdf.columns
Xdf.iloc[:,0:] = scaler.fit_transform(Xdf.iloc[:,0:])

Xforall = pd.DataFrame()
Yforall = pd.DataFrame()

XdfTest = pd.concat([dataAbxTest.iloc[:,1:]], axis=1)
df1 = XdfTest.filter(regex='site2')
df2 = XdfTest.filter(regex='site4')
df3 = XdfTest.filter(regex='site6')
XdfTest = pd.concat([df1, df2, df3], axis=1)
XdfTest.iloc[:,0:] = scaler.fit_transform(XdfTest.iloc[:,0:])

XforallTest = pd.DataFrame()
YforallTest = pd.DataFrame()


##----------------------------------------------------------

responses = ['Acinetobacter baumanii', 'Enterobacter spp.', 'Escherichia coli',
             'Haemophilus influenzae', 'Klebsiella pneumoniae', 'Morganella morganii', 'Proteus mirabilis',
             'Pseudomonas aeruginosa',  'Serratia spp.', 'Staphylococcus aureus']


folder = 'Combined microbes/ALL'
adf = dataResMic
adfTest = dataResMicTest

for response in responses:
    df = adf[response]
    tempYdf = df[df != 'None']
    indexlist = tempYdf.index.tolist()
    tempXdf = Xdf.iloc[indexlist, 0:]
    Xforall = pd.concat([Xforall, tempXdf])
    Yforall = pd.concat([Yforall, tempYdf])

    tempXdfTest = XdfTest.iloc[indexlist, 0:]
    XforallTest = pd.concat([XforallTest, tempXdfTest])
    YforallTest = pd.concat([YforallTest, tempYdf])

y = Yforall.to_numpy().reshape((-1, 1))
X = Xforall
model = linear_model.PoissonRegressor(alpha=0).fit(X, y)
numberOfFeatures = 1  # 100
rfe = RFE(model, n_features_to_select=numberOfFeatures)
fit = rfe.fit(X, y)

df = pd.DataFrame()
df['Features'] = X.columns
df['Ranking'] = fit.ranking_
df = df.sort_values('Ranking')
df.to_csv('Output/' + str(folder) + '/Recursive FE_all.csv', index=False)

yTest = YforallTest.to_numpy().reshape((-1, 1))
XTest = XforallTest
model = linear_model.PoissonRegressor(alpha=0).fit(XTest, yTest)
numberOfFeatures = 1  # 100
rfe = RFE(model, n_features_to_select=numberOfFeatures)
fit = rfe.fit(XTest, yTest)

df = pd.DataFrame()
df['Features'] = XTest.columns
df['Ranking'] = fit.ranking_
df = df.sort_values('Ranking')
df.to_csv('Output/' + str(folder) + '/Recursive FE_all_TEST.csv', index=False)

optlist = []
for response in responses:
    scaler = MinMaxScaler(feature_range=(0., 1.))

    Xdf = pd.concat([dataAbx.iloc[:, 1:]], axis=1)
    df1 = Xdf.filter(regex='site2')
    df2 = Xdf.filter(regex='site4')
    df3 = Xdf.filter(regex='site6')
    Xdf = pd.concat([df1, df2, df3], axis=1)
    colList = Xdf.columns
    Xdf.iloc[:, 0:] = scaler.fit_transform(Xdf.iloc[:, 0:])

    XdfTest = pd.concat([dataAbxTest.iloc[:, 1:]], axis=1)
    df1 = XdfTest.filter(regex='site2')
    df2 = XdfTest.filter(regex='site4')
    df3 = XdfTest.filter(regex='site6')
    XdfTest = pd.concat([df1, df2, df3], axis=1)
    XdfTest.iloc[:, 0:] = scaler.fit_transform(XdfTest.iloc[:, 0:])

    modelFit(response, adf, adfTest, 'ComAll')
    renameFS(response)


##
folder = 'Combined microbes/Hospital'
adf = dataResIN
adfTest = dataResINTest

for response in responses:
    df = adf[response]
    tempYdf = df[df != 'None']
    indexlist = tempYdf.index.tolist()
    tempXdf = Xdf.iloc[indexlist, 0:]
    Xforall = pd.concat([Xforall, tempXdf])
    Yforall = pd.concat([Yforall, tempYdf])

    tempXdfTest = XdfTest.iloc[indexlist, 0:]
    XforallTest = pd.concat([XforallTest, tempXdfTest])
    YforallTest = pd.concat([YforallTest, tempYdf])

y = Yforall.to_numpy().reshape((-1, 1))
X = Xforall
model = linear_model.PoissonRegressor(alpha=0).fit(X, y)
numberOfFeatures = 1  # 100
rfe = RFE(model, n_features_to_select=numberOfFeatures)
fit = rfe.fit(X, y)

df = pd.DataFrame()
df['Features'] = X.columns
df['Ranking'] = fit.ranking_
df = df.sort_values('Ranking')
df.to_csv('Output/' + str(folder) + '/Recursive FE_all.csv', index=False)

yTest = YforallTest.to_numpy().reshape((-1, 1))
XTest = XforallTest
model = linear_model.PoissonRegressor(alpha=0).fit(XTest, yTest)
numberOfFeatures = 1  # 100
rfe = RFE(model, n_features_to_select=numberOfFeatures)
fit = rfe.fit(XTest, yTest)

df = pd.DataFrame()
df['Features'] = XTest.columns
df['Ranking'] = fit.ranking_
df = df.sort_values('Ranking')
df.to_csv('Output/' + str(folder) + '/Recursive FE_all_TEST.csv', index=False)

optlist = []
for response in responses:
    scaler = MinMaxScaler(feature_range=(0., 1.))

    Xdf = pd.concat([dataAbx.iloc[:, 1:]], axis=1)
    df1 = Xdf.filter(regex='site2')
    df2 = Xdf.filter(regex='site4')
    df3 = Xdf.filter(regex='site6')
    Xdf = pd.concat([df1, df2, df3], axis=1)
    colList = Xdf.columns
    Xdf.iloc[:, 0:] = scaler.fit_transform(Xdf.iloc[:, 0:])

    XdfTest = pd.concat([dataAbxTest.iloc[:, 1:]], axis=1)
    df1 = XdfTest.filter(regex='site2')
    df2 = XdfTest.filter(regex='site4')
    df3 = XdfTest.filter(regex='site6')
    XdfTest = pd.concat([df1, df2, df3], axis=1)
    XdfTest.iloc[:, 0:] = scaler.fit_transform(XdfTest.iloc[:, 0:])

    modelFit(response, adf, adfTest, 'ComIN')
    renameFS(response)

# ----------------------------------------------------------

##----------------------------------------------------------

folder = 'Combined microbes/Community'
adf = dataResOUT
adfTest = dataResOUTTest

for response in responses:
    df = adf[response]
    tempYdf = df[df != 'None']
    indexlist = tempYdf.index.tolist()
    tempXdf = Xdf.iloc[indexlist, 0:]
    Xforall = pd.concat([Xforall, tempXdf])
    Yforall = pd.concat([Yforall, tempYdf])

    tempXdfTest = XdfTest.iloc[indexlist, 0:]
    XforallTest = pd.concat([XforallTest, tempXdfTest])
    YforallTest = pd.concat([YforallTest, tempYdf])

y = Yforall.to_numpy().reshape((-1, 1))
X = Xforall
model = linear_model.PoissonRegressor(alpha=0).fit(X, y)
numberOfFeatures = 1  # 100
rfe = RFE(model, n_features_to_select=numberOfFeatures)
fit = rfe.fit(X, y)

df = pd.DataFrame()
df['Features'] = X.columns
df['Ranking'] = fit.ranking_
df = df.sort_values('Ranking')
df.to_csv('Output/' + str(folder) + '/Recursive FE_all.csv', index=False)

yTest = YforallTest.to_numpy().reshape((-1, 1))
XTest = XforallTest
model = linear_model.PoissonRegressor(alpha=0).fit(XTest, yTest)
numberOfFeatures = 1  # 100
rfe = RFE(model, n_features_to_select=numberOfFeatures)
fit = rfe.fit(XTest, yTest)

df = pd.DataFrame()
df['Features'] = XTest.columns
df['Ranking'] = fit.ranking_
df = df.sort_values('Ranking')
df.to_csv('Output/' + str(folder) + '/Recursive FE_all_TEST.csv', index=False)

optlist = []
for response in responses:
    scaler = MinMaxScaler(feature_range=(0., 1.))

    Xdf = pd.concat([dataAbx.iloc[:, 1:]], axis=1)
    df1 = Xdf.filter(regex='site2')
    df2 = Xdf.filter(regex='site4')
    df3 = Xdf.filter(regex='site6')
    Xdf = pd.concat([df1, df2, df3], axis=1)
    colList = Xdf.columns
    Xdf.iloc[:, 0:] = scaler.fit_transform(Xdf.iloc[:, 0:])

    XdfTest = pd.concat([dataAbxTest.iloc[:, 1:]], axis=1)
    df1 = XdfTest.filter(regex='site2')
    df2 = XdfTest.filter(regex='site4')
    df3 = XdfTest.filter(regex='site6')
    XdfTest = pd.concat([df1, df2, df3], axis=1)
    XdfTest.iloc[:, 0:] = scaler.fit_transform(XdfTest.iloc[:, 0:])

    modelFit(response, adf, adfTest, 'ComOUT')
    renameFS(response)


##

