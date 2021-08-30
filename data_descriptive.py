from scipy.stats import skew,kurtosis,iqr, median_abs_deviation as mad
from statistics import mode 
import pandas as pd
import numpy as np


def Missing_values(dataframe):
    dictio = {}
    
    cols = sorted(dataframe.columns.tolist())
    dtypes = dataframe[cols].dtypes
    
    for i in range(len(cols)):
        dictio[cols[i]] = [ dtypes[i], dataframe[cols[i]].count(), 0 ]
    temp_df = pd.DataFrame.from_dict(dictio, columns=['Dtype','Count','Missing_Values'], orient='index')
    
    missing_values = dataframe.isnull().sum()
    a = missing_values[ missing_values > 0]
    temp_df['Missing_Values'] = (temp_df['Missing_Values'] + a).replace(np.nan,0).astype(int)

    temp_df['% Missing_Values'] = round(100*temp_df['Missing_Values']/len(dataframe),2)
    
    return temp_df



def numeric_cols(dataframe):
    res= dataframe.select_dtypes(include=np.number).columns.tolist()
    return res
#[is_numeric(dataframe[x]) for x in dataframe.columns.tolist()]

def Statistical_description(dataframe):
    dictio = {}
    
    cols = numeric_cols(dataframe)
    cols.sort()
    
    for i in range(len(cols)):
        x = dataframe[cols[i]]
        dictio[cols[i]] = [len(x),np.max(x) - np.min(x),iqr(x),mode(x),mad(x),
                           kurtosis(x),skew(x),np.mean(x),np.std(x),np.min(x),
                           np.quantile(x, 0.25),np.quantile(x, 0.5),np.quantile(x, 0.75),np.max(x)]
    stat_metrics = ['count','range','IQR','mode','mad','kurtosis','skewness','mean','std','min','25%','50%','75%','max']
    temp_df = pd.DataFrame.from_dict(dictio, columns=stat_metrics, orient='index')
    
    return temp_df.apply(lambda x : round(x,2))