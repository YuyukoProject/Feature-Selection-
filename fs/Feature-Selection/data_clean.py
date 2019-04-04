
# coding: utf-8

# In[1]:


import numpy as np,pandas as pd
import math,time
from sklearn.feature_selection import variance_threshold
import warnings
warnings.filterwarnings('ignore')


# In[2]:


class data_clean():
    ''' cheak the data before use the feature selection.
        1. Find columns with a missing percentage greater than a specified threshold
        2. Find columns with a single unique value
        3. Find columns with a lower variance
    '''
    def __init__(self,selection_params,features=None):
        '''
        Args:
            selection_params:dict,Parameters to use in the third feature selection methhods.
                                must contain the keys ['missing_threshold','variance_threshold']
        '''
        for param in ['missing_threshold','variance_threshold']:
            if param not in selection_params.keys():
                raise ValueError('%s is  a required parameter for this method.' % param)
        
        self.identify_missing=selection_params['missing_threshold']
        self.identify_single_unique=1
        self.identify_variance=selection_params['variance_threshold']
        self.temp=features
        self.logfile = 'record_clean.log'
                
    def fit(self,data):
        self.df = df
        if not self.temp:
            self.temp=list(self.df.columns)
        for feature in self.temp:
            if feature not in self.df.columns:
                raise ValueError('Features Must Be In Data')                 
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n{}\n%{}%\n'.format('Start!',time.ctime(),'-'*100))
        result=_data_cheak(data=self.df,start=self.temp,
                              missing=self.identify_missing,lower_var=self.identify_variance,
                              single_unique=self.identify_single_unique)
        result.select()
        self.ops=result.ops
        self.data=result.data_retain
        self.miss=result.miss_col
        self.single=result.single_col
        self.var=result.var_col
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n miss:{}\n single:{}\n lower variance:{}\n%{}%\n'.format('Done',self.miss,self.single,self.var,'-'*100))


# In[3]:


class _data_cheak(object):    
    def __init__(self,data,start,missing,lower_var,single_unique):
        self._df=data
        self._temp=start
        self._miss=missing
        self._lower_var=lower_var
        self._single_unique=single_unique
        self.ops={}
        
    def select(self):
        self._identify_missing()
        self._identify_single_unique()
        self._identify_lower_var()
        self._remove=np.array([_ for _ in self.ops.values() if len(_)>0]).flatten()
        self._use_col=[col for col in self._temp if col not in self._remove]
        self.data_retain=self._df[self._use_col]
        
    def _identify_missing(self):
        '''Find the features with a fraction of missing values above `missing_threshold'''
        miss_series = self._df[self._temp].isnull().sum() / self._df.shape[0]
        self.miss_col = list(miss_series[miss_series>=self._miss].index)
        self.ops['missing'] = self.miss_col
        
    def _identify_single_unique(self):
        '''Finds features with only a single unique value.'''
        unique_counts = self._df[self._temp].nunique()
        self.single_col=list(unique_counts[unique_counts<=1].index)
        self.ops['single_unique'] = self.single_col
        
    def _identify_lower_var(self):
        '''Finds features with low variance '''
        selector =variance_threshold.VarianceThreshold(self._lower_var)
        try:
            selector.fit_transform(self._df[self._temp])
        except:
            raise ValueError('No feature in data meets the variance threshold {}'.format(self._lower_var))
        self.var_col=[col for col in self._temp if col not in np.array(self._temp)[selector.get_support()]]
        self.ops['lower_var']=self.var_col

