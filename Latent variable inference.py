## Load packages

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

#------------------------------------------------
# Import the data
#------------------------------------------------

# Pharmacies
dataPharm = pd.read_csv("Data/Antibiotics_Pharmacies.csv", sep=',')
dataPharm = pd.DataFrame(dataPharm)
# WCDoH
dataHosp = pd.read_csv("Data/Antibiotics_Hospital_Clinics.csv", sep=',')
dataHosp = pd.DataFrame(dataHosp)
# Vet Abx
dataVet = pd.read_csv("Data/Antibiotics_Veterinary.csv", sep=',')
dataVet = pd.DataFrame(dataVet)

print(dataPharm.head())
print(dataHosp.head())
print(dataVet.head())


##__________________________________________________________________
# Convert the Pharmacy, WCDoH and Vet Abx Dataframe months to a single date
#___________________________________________________________________

dataPharm.replace({'18_Jan': pd.to_datetime('15 Jan 2018'), '18_Feb': pd.to_datetime('15 Feb 2018'), '18_Mar': pd.to_datetime('15 Mar 2018'),
                   '18_Apr': pd.to_datetime('15 Apr 2018'), '18_May': pd.to_datetime('15 May 2018'), '18_Jun': pd.to_datetime('15 Jun 2018'),
                   '18_Jul': pd.to_datetime('15 Jul 2018'), '18_Aug': pd.to_datetime('15 Aug 2018'), '18_Sep': pd.to_datetime('15 Sep 2018'),
                   '18_Oct': pd.to_datetime('15 Oct 2018'), '18_Nov': pd.to_datetime('15 Nov 2018'), '18_Dec': pd.to_datetime('15 Dec 2018'),
                   '19_Jan': pd.to_datetime('15 Jan 2019'), '19_Feb': pd.to_datetime('15 Feb 2019'), '19_Mar': pd.to_datetime('15 Mar 2019'),
                   '19_Apr': pd.to_datetime('15 Apr 2019'), '19_May': pd.to_datetime('15 May 2019'), '19_Jun': pd.to_datetime('15 Jun 2019'),
                   '19_Jul': pd.to_datetime('15 Jul 2019'), '19_Aug': pd.to_datetime('15 Aug 2019'), '19_Sep': pd.to_datetime('15 Sep 2019'),
                   '19_Oct': pd.to_datetime('15 Oct 2019'), '19_Nov': pd.to_datetime('15 Nov 2019'), '19_Dec': pd.to_datetime('15 Dec 2019'),
                   },inplace=True)

dataHosp.replace({'18_Jan': pd.to_datetime('15 Jan 2018'), '18_Feb': pd.to_datetime('15 Feb 2018'), '18_Mar': pd.to_datetime('15 Mar 2018'),
                   '18_Apr': pd.to_datetime('15 Apr 2018'), '18_May': pd.to_datetime('15 May 2018'), '18_Jun': pd.to_datetime('15 Jun 2018'),
                   '18_Jul': pd.to_datetime('15 Jul 2018'), '18_Aug': pd.to_datetime('15 Aug 2018'), '18_Sep': pd.to_datetime('15 Sep 2018'),
                   '18_Oct': pd.to_datetime('15 Oct 2018'), '18_Nov': pd.to_datetime('15 Nov 2018'), '18_Dec': pd.to_datetime('15 Dec 2018'),
                   '19_Jan': pd.to_datetime('15 Jan 2019'), '19_Feb': pd.to_datetime('15 Feb 2019'), '19_Mar': pd.to_datetime('15 Mar 2019'),
                   '19_Apr': pd.to_datetime('15 Apr 2019'), '19_May': pd.to_datetime('15 May 2019'), '19_Jun': pd.to_datetime('15 Jun 2019'),
                   '19_Jul': pd.to_datetime('15 Jul 2019'), '19_Aug': pd.to_datetime('15 Aug 2019'), '19_Sep': pd.to_datetime('15 Sep 2019'),
                   '19_Oct': pd.to_datetime('15 Oct 2019'), '19_Nov': pd.to_datetime('15 Nov 2019'), '19_Dec': pd.to_datetime('15 Dec 2019'),
                   },inplace=True)

dataVet.replace({'18_Jan': pd.to_datetime('15 Jan 2018'), '18_Feb': pd.to_datetime('15 Feb 2018'), '18_Mar': pd.to_datetime('15 Mar 2018'),
                   '18_Apr': pd.to_datetime('15 Apr 2018'), '18_May': pd.to_datetime('15 May 2018'), '18_Jun': pd.to_datetime('15 Jun 2018'),
                   '18_Jul': pd.to_datetime('15 Jul 2018'), '18_Aug': pd.to_datetime('15 Aug 2018'), '18_Sep': pd.to_datetime('15 Sep 2018'),
                   '18_Oct': pd.to_datetime('15 Oct 2018'), '18_Nov': pd.to_datetime('15 Nov 2018'), '18_Dec': pd.to_datetime('15 Dec 2018'),
                   '19_Jan': pd.to_datetime('15 Jan 2019'), '19_Feb': pd.to_datetime('15 Feb 2019'), '19_Mar': pd.to_datetime('15 Mar 2019'),
                   '19_Apr': pd.to_datetime('15 Apr 2019'), '19_May': pd.to_datetime('15 May 2019'), '19_Jun': pd.to_datetime('15 Jun 2019'),
                   '19_Jul': pd.to_datetime('15 Jul 2019'), '19_Aug': pd.to_datetime('15 Aug 2019'), '19_Sep': pd.to_datetime('15 Sep 2019'),
                   '19_Oct': pd.to_datetime('15 Oct 2019'), '19_Nov': pd.to_datetime('15 Nov 2019'), '19_Dec': pd.to_datetime('15 Dec 2019'),
                   },inplace=True)

dataPharm.sort_values(by='Date', inplace=True)
dataHosp.sort_values(by='Date', inplace=True)
dataVet.sort_values(by='Date', inplace=True)

print(dataPharm)
print(dataHosp)
print(dataVet)


##__________________________________________________________________
# GPR and calculate latent variables for Pharmacy antibiotic data
#___________________________________________________________________

days = pd.date_range(start="1 jan 2018",end="31 dec 2019")
print(len(days))

# Set the date range over which to infer
exportdf = pd.DataFrame()
exportdf['Date'] = days

# Annotate the dates as numbers: day of the year over two years 1 to 730
xVals = []
for i in dataPharm['Date']:
    if i < pd.to_datetime('1 January 2019'):
        xVals.append(i.timetuple().tm_yday)
    else:
        xVals.append(i.timetuple().tm_yday + 365)
xVals[:] = [number - xVals[0] +1 for number in xVals]

# Define the range from 1 to the length of the days over which to predict
x_pred = np.array(range(1, len(days)+1, 1)).reshape(-1, 1)

# Get feature names
cols = dataPharm.columns[2:]

scaler = MinMaxScaler(feature_range=(0.,1))

for i in cols:
    yVals1 = np.array(dataPharm[i]).reshape(-1, 1)
    yVals2 = np.array(dataPharm[i]).reshape(-1, 1)

    # Scale the data
    yVals1 = scaler.fit_transform(yVals1)
    yVals2 = scaler.fit_transform(yVals2)

    # Calculate mean and std dev
    yValsMean = []
    yValsSD = []
    for j in range(len(yVals1)):
        yValsMean.append(np.mean([yVals1[j], yVals2[j]]))
        yValsSD.append(np.std([yVals1[j], yVals2[j]]))

    # Scale the mean data again after calculating the means to get the max=1
    yValsMean = np.array(yValsMean).reshape(-1, 1)
    yValsMean = scaler.fit_transform(yValsMean)
    yValsMean = yValsMean.ravel()

    print(max(yValsMean))
    # Transform the y values to log(y + 1)
    yValsMean[:] = [math.log(number + 1) for number in yValsMean]
    yValsSD[:] = [math.log(number + 1) for number in yValsSD]

    yValsMean = np.array(yValsMean)  # .reshape(-1, 1)
    yValsSD = np.array(yValsSD)  # .reshape(-1, 1)
    xVals = np.array(xVals).reshape(-1, 1)

    kernel = RBF(10, (1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=yValsSD ** 2, n_restarts_optimizer=0,
                                  normalize_y=False)  # Toggle True and False

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(xVals, yValsMean)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    print(y_pred)

    # Transform the y values back from the transformed space
    y_pred[:] = [np.exp(number) - 1 for number in y_pred]
    sigma[:] = [np.exp(number) - 1 for number in sigma]  # NEW
    yValsMean[:] = [np.exp(number) - 1 for number in yValsMean]  # NEW
    yValsSD[:] = [np.exp(number) - 1 for number in yValsSD]  # NEW

    # Convert inferred y_pred to all positive values
    y_pred[:] = [number if number >= 0 else 0.0 for number in y_pred]

    exportdf[i] = y_pred

    # Plot CIs
    xCI = np.concatenate([x_pred, x_pred[::-1]])
    yCI = np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]])
    yCI[:] = [number if number >= 0 else 0.0 for number in yCI]

    # Plot the function, the prediction and the 95% confidence interval based on the MSE
    plt.ioff()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # main axes
    ax.errorbar(xVals, yValsMean, yValsSD, fmt='k.', markersize=10, label='ReNEW Sampling Data')
    ax.plot(x_pred, y_pred, '-', label='Inferred Latent Variable')
    ax.fill(xCI, yCI, alpha=.3, ec='None', label='95% Confidence Interval')  # fc='b'
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Scaled Daily Load', fontsize=18)
    ax.set_title(i, fontsize=22)
    ax.set_xticks([5, 180, 370, 545, 700])
    ax.set_xticklabels(['Jan 2018', 'Jul 2018', 'Jan 2019', 'Jul 2019', 'Dec 2019'], fontsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(loc='upper left', fontsize=14)
    plt.savefig('Plots/Inferred latent variables/Pharmacies/' + str(i) + '.png')
    plt.close()

exportdf.to_csv('Data/Pharmacy inferred latent variables.csv')


##__________________________________________________________________
# GPR and calculate latent variables for Hospital and Clinic antibiotic data
#___________________________________________________________________

days = pd.date_range(start="1 jan 2018",end="31 dec 2019")
print(len(days))

# Set the date range over which to infer
exportdf = pd.DataFrame()
exportdf['Date'] = days

# Annotate the dates as numbers: day of the year over two years 1 to 730
xVals = []
for i in dataHosp['Date']:
    if i < pd.to_datetime('1 January 2019'):
        xVals.append(i.timetuple().tm_yday)
    else:
        xVals.append(i.timetuple().tm_yday + 365)
xVals[:] = [number - xVals[0] +1 for number in xVals]

# Define the range from 1 to the length of the days over which to predict
x_pred = np.array(range(1, len(days)+1, 1)).reshape(-1, 1)

# Get feature names
cols = dataHosp.columns[2:]

scaler = MinMaxScaler(feature_range=(0.,1))

for i in cols:
    yVals1 = np.array(dataHosp[i]).reshape(-1, 1)
    yVals2 = np.array(dataHosp[i]).reshape(-1, 1)

    # Scale the data
    yVals1 = scaler.fit_transform(yVals1)
    yVals2 = scaler.fit_transform(yVals2)

    # Calculate mean and std dev
    yValsMean = []
    yValsSD = []
    for j in range(len(yVals1)):
        yValsMean.append(np.mean([yVals1[j], yVals2[j]]))
        yValsSD.append(np.std([yVals1[j], yVals2[j]]))

    # Scale the mean data again after calculating the means to get the max=1
    yValsMean = np.array(yValsMean).reshape(-1, 1)
    yValsMean = scaler.fit_transform(yValsMean)
    yValsMean = yValsMean.ravel()

    print(max(yValsMean))
    # Transform the y values to log(y + 1)
    yValsMean[:] = [math.log(number + 1) for number in yValsMean]
    yValsSD[:] = [math.log(number + 1) for number in yValsSD]

    yValsMean = np.array(yValsMean)  # .reshape(-1, 1)
    yValsSD = np.array(yValsSD)  # .reshape(-1, 1)
    xVals = np.array(xVals).reshape(-1, 1)

    kernel = RBF(10, (1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=yValsSD ** 2, n_restarts_optimizer=0,
                                  normalize_y=False)  # Toggle True and False

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(xVals, yValsMean)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    print(y_pred)

    # Transform the y values back from the transformed space
    y_pred[:] = [np.exp(number) - 1 for number in y_pred]
    sigma[:] = [np.exp(number) - 1 for number in sigma]  # NEW
    yValsMean[:] = [np.exp(number) - 1 for number in yValsMean]  # NEW
    yValsSD[:] = [np.exp(number) - 1 for number in yValsSD]  # NEW

    # Convert inferred y_pred to all positive values
    y_pred[:] = [number if number >= 0 else 0.0 for number in y_pred]

    exportdf[i] = y_pred

    # Plot CIs
    xCI = np.concatenate([x_pred, x_pred[::-1]])
    yCI = np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]])
    yCI[:] = [number if number >= 0 else 0.0 for number in yCI]

    # Plot the function, the prediction and the 95% confidence interval based on the MSE
    plt.ioff()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # main axes
    ax.errorbar(xVals, yValsMean, yValsSD, fmt='k.', markersize=10, label='ReNEW Sampling Data')
    ax.plot(x_pred, y_pred, '-', label='Inferred Latent Variable')
    ax.fill(xCI, yCI, alpha=.3, ec='None', label='95% Confidence Interval')  # fc='b'
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Scaled Daily Load', fontsize=18)
    ax.set_title(i, fontsize=22)
    ax.set_xticks([5, 180, 370, 545, 700])
    ax.set_xticklabels(['Jan 2018', 'Jul 2018', 'Jan 2019', 'Jul 2019', 'Dec 2019'], fontsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(loc='upper left', fontsize=14)
    plt.savefig('Plots/Inferred latent variables/Hospital and Clinics/' + str(i) + '.png')
    plt.close()

exportdf.to_csv('Data/Hospital and clinic inferred latent variables.csv')


##__________________________________________________________________
# GPR and calculate latent variables for Veterinary antibiotic data
#___________________________________________________________________

days = pd.date_range(start="1 jan 2018",end="31 dec 2019")
print(len(days))

# Set the date range over which to infer
exportdf = pd.DataFrame()
exportdf['Date'] = days

# Annotate the dates as numbers: day of the year over two years 1 to 730
xVals = []
for i in dataVet['Date']:
    if i < pd.to_datetime('1 January 2019'):
        xVals.append(i.timetuple().tm_yday)
    else:
        xVals.append(i.timetuple().tm_yday + 365)
xVals[:] = [number - xVals[0] +1 for number in xVals]

# Define the range from 1 to the length of the days over which to predict
x_pred = np.array(range(1, len(days)+1, 1)).reshape(-1, 1)

# Get feature names
cols = dataVet.columns[2:]

scaler = MinMaxScaler(feature_range=(0.,1))

for i in cols:
    yVals1 = np.array(dataVet[i]).reshape(-1, 1)
    yVals2 = np.array(dataVet[i]).reshape(-1, 1)

    # Scale the data
    yVals1 = scaler.fit_transform(yVals1)
    yVals2 = scaler.fit_transform(yVals2)

    # Calculate mean and std dev
    yValsMean = []
    yValsSD = []
    for j in range(len(yVals1)):
        yValsMean.append(np.mean([yVals1[j], yVals2[j]]))
        yValsSD.append(np.std([yVals1[j], yVals2[j]]))

    # Scale the mean data again after calculating the means to get the max=1
    yValsMean = np.array(yValsMean).reshape(-1, 1)
    yValsMean = scaler.fit_transform(yValsMean)
    yValsMean = yValsMean.ravel()

    print(max(yValsMean))
    # Transform the y values to log(y + 1)
    yValsMean[:] = [math.log(number + 1) for number in yValsMean]
    yValsSD[:] = [math.log(number + 1) for number in yValsSD]

    yValsMean = np.array(yValsMean)  # .reshape(-1, 1)
    yValsSD = np.array(yValsSD)  # .reshape(-1, 1)
    xVals = np.array(xVals).reshape(-1, 1)

    kernel = RBF(10, (1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=yValsSD ** 2, n_restarts_optimizer=0,
                                  normalize_y=False)  # Toggle True and False

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(xVals, yValsMean)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    print(y_pred)

    # Transform the y values back from the transformed space
    y_pred[:] = [np.exp(number) - 1 for number in y_pred]
    sigma[:] = [np.exp(number) - 1 for number in sigma]  # NEW
    yValsMean[:] = [np.exp(number) - 1 for number in yValsMean]  # NEW
    yValsSD[:] = [np.exp(number) - 1 for number in yValsSD]  # NEW

    # Convert inferred y_pred to all positive values
    y_pred[:] = [number if number >= 0 else 0.0 for number in y_pred]

    exportdf[i] = y_pred

    # Plot CIs
    xCI = np.concatenate([x_pred, x_pred[::-1]])
    yCI = np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]])
    yCI[:] = [number if number >= 0 else 0.0 for number in yCI]

    # Plot the function, the prediction and the 95% confidence interval based on the MSE
    plt.ioff()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # main axes
    ax.errorbar(xVals, yValsMean, yValsSD, fmt='k.', markersize=10, label='ReNEW Sampling Data')
    ax.plot(x_pred, y_pred, '-', label='Inferred Latent Variable')
    ax.fill(xCI, yCI, alpha=.3, ec='None', label='95% Confidence Interval')  # fc='b'
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Scaled Daily Load', fontsize=18)
    ax.set_title(i, fontsize=22)
    ax.set_xticks([5, 180, 370, 545, 700])
    ax.set_xticklabels(['Jan 2018', 'Jul 2018', 'Jan 2019', 'Jul 2019', 'Dec 2019'], fontsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(loc='upper left', fontsize=14)
    plt.savefig('Plots/Inferred latent variables/Veterinary/' + str(i) + '.png')
    plt.close()

exportdf.to_csv('Data/Veterinary inferred latent variables.csv')


##

