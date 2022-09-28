from pandas.core.frame import DataFrame
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLSResults
from statsmodels.stats.outliers_influence import variance_inflation_factor
sns.set()

def dropcolumns(data,drop) -> DataFrame:
    """Returns the parsed dataframe with the columns specified dropped.
    Mostly used in other functions"""
    CD = data.copy()
    if drop:
        CD.drop(drop,axis=1,inplace=True)
    return CD

def vif_scores(data) -> DataFrame:
    """Calculates the VIF scores for each indep variable and returns them in a dataframe"""
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independent Features"] = data.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    return VIF_Scores

def perform_regression(data,target,drop=None,cov_type=None) -> OLSResults:
    """Performs regression and returns the model"""
    CD = dropcolumns(data,drop)
    if cov_type:
        model = sm.OLS(target,sm.add_constant(CD)).fit(cov_type=cov_type)
        return model
    else:
        model = sm.OLS(target,sm.add_constant(CD)).fit()
        return model

def check_multi(data,target,drop=None,cov_type=None) -> None:
    """Check the endog variables for multicolinearity, if present runs vif_scores"""
    if perform_regression(data,target,drop=drop,cov_type=cov_type).condition_number > 200:
        print('Consider removing variables with high VIF scores.')
        return vif_scores(dropcolumns(data,drop))
    else:
        print('No strong multicolinearity present')

def checkhs(data,target,drop=None,cov_type=None) -> plt.plot:
    """Checks for HS by plotting the residuals"""
    CD = dropcolumns(data,drop)
    X = range(len(CD.index))
    y = perform_regression(CD,target,cov_type=cov_type).resid
    fig, ax = plt.subplots(figsize=(12,6))
    ax.scatter(X,y)
    ax.set(xlabel='Fitted Line',ylabel='Residuals',title='A conelike shape indicates heteroscedacity')
    plt.show()

def allchecks(data,target,drop=None,cov_type=None) -> plt.plot:
    """Performs checks for multicolinearity and Heteroschedacity"""
    return checkhs(data,target,drop=drop,cov_type=cov_type), check_multi(data,target,drop=drop,cov_type=cov_type)

def dummify(data,drop=None) -> DataFrame:
    """Finds columns eligible to OHE and returns a dataframe with the operation performed on all viable columns"""
    CD = dropcolumns(data, drop=drop)
    temp_df = pd.DataFrame()
    for i in CD:
        if type(CD[i][0]) == str and len(i.split())==1:
            temp = pd.get_dummies(CD[i],drop_first=True)
            temp_df = pd.concat([temp_df,temp],axis=1)
            CD.drop(i,axis=1,inplace=True)
    return pd.concat([CD, temp_df],axis=1)

def polynomial_regression(data,target,order,drop=None,cov_type=None):
    temp = pd.DataFrame()
    for j in range(2,order+1):
        for i in data:
            temp[i+f'**{j}'] = data[i].apply(lambda x:x**j)
    data = pd.concat([data,temp],axis=1)
    return perform_regression(data,target,drop=drop,cov_type=cov_type)

def partialplots(data,target,drop=None,cov_type=None):
    fig, axes = plt.subplots(data.shape[1],1,figsize=(12,6))
    for i,key in enumerate(data):
        model = perform_regression(data[key],target,cov_type=cov_type)
        Xfit = np.linspace(min(data[key]),max(data[key]))
        yfit = model.predict(sm.add_constant(Xfit))
        axes[i].scatter(data[key],target,label=f'{key}')
        axes[i].plot(Xfit,yfit)
        axes[i].legend()
    plt.show()