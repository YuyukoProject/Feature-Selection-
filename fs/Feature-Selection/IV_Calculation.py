
# coding: utf-8

# In[94]:


import numpy as np ,pandas as pd,math,time
from Split import *
import warnings
warnings.filterwarnings('ignore')


# In[95]:


class IV_Select(object):
    
    def __init__(self,features=None,bins=5,woe_limit=5,min_sample = 1):
        '''
        Args:
            features:list,Feature of IV(WOE) needed to be calculated 
            bins:int,Indicate the number of data boxes
            woe_limit: numerical,set the limit of woe ,sometimes we must set this.
        '''
        self.temp = features
        self.min_sample=min_sample
        if isinstance(bins,int):
            self.bins=bins
        else:
            raise ValueError('Bins Must Be Int') 
        self.woe_limit=woe_limit
        self.logfile = 'record_IV.log'

    def fit(self,df,label,positive=None):
        '''Import pandas dataframe to the class
        Args:
            df: pandas dataframe include all features and label.
            label: str, label name
            positive: choose the label value to be positive ,others would be negative
        '''
        self.df = df
        self.label = label
        if not self.temp:
            self.temp=list(self.df.columns)
            self.temp.remove(self.label)
        for feature in self.temp:
            if feature not in self.df.columns:
                raise ValueError('Features Must Be In Data') 
        if (len(np.unique(self.df[self.label])) != 2)&(positive==None):
            raise ValueError('label Must Be Binary') 
        self.positive=positive
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n{}\n%{}%\n'.format('Start!',time.ctime(),'-'*100))
        
        a = _IV_calculated(df = self.df, label = self.label,
                                    start = self.temp,bins=self.bins,min_sample=self.min_sample,
                                     positive=self.positive,woe_limit=self.woe_limit)
        a._woe_iv()
        self.iv_dict = a._iv_dict
        self.woe_dict = a._woe_dict
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n{}\n%{}%\n'.format('Done',self.iv_dict,self.woe_dict,'-'*100))
        return self.iv_dict


# In[96]:


class _IV_calculated(object):
    
    def __init__(self,df,label,start,bins,min_sample,positive,woe_limit):
        self._df=df 
        self._label =label
        self._temp =start 
        self._bins=bins
        self.min_sample=min_sample
        self._positive=positive 
        self._woe_limit=woe_limit
        
    def _woe_iv(self):
        ''' use the Split to divide the data into bins ,then calculated  the WOE and IV for the feature .
            Args:
                data:dataframe like, the data need to calculated woe and iv
                features:the features in the data columns 
                label: point the label,Must Be Binary(if not,with positive we could choose one value to be 1 others be 0)
                positive: choose the positive of the label 
                woe_limit :the limit of the woe 
        '''
        self._woe_dict,self._iv_dict={},{}
        self._data_temp=self._df.copy()
        if self._positive:
            self._data_temp['temp']=0
            self._data_temp['temp'][self._data_temp[self._label]==self._positive]=1
            self._data_temp['temp'][self._data_temp[self._label]!=self._positive] =0
        else:
            self._data_temp['temp']=self._data_temp['lable']
        event_total = self._data_temp['temp'].sum()
        non_event_total = len(self._data_temp['temp']) - event_total
        self._split_bins()
        for col in self._temp:
            woe_dict_k={}
            iv=0
            bins_col=self.bins[col]
            bins_col.extend([-np.inf,np.inf])
            self._data_temp[col]=pd.cut(self._df[col],bins=sorted(bins_col))
            col_dic=self._event_dict(self._data_temp,col,'temp',event_total,non_event_total)
            for k,(rate_event, rate_non_event) in col_dic.items():
                if rate_event == 0:
                    woe_k = -1*self._woe_limit
                elif rate_non_event == 0:
                    woe_k = self._woe_limit
                else:
                    woe_k = np.log(rate_event / rate_non_event)
                woe_dict_k[k]=woe_k
                iv += (rate_event - rate_non_event) * woe_k
            self._woe_dict[col]=woe_dict_k
            self._iv_dict[col]=iv
            self._col_table=pd.DataFrame([woe_dict_k],index=['woe'])
    def _event_dict(self,data,col,label,et,net):
        '''Statistical data by value according to discrete classes
        '''
        val=np.unique(data[col])
        event_dict={}
        for i in val:
            event_count=data[data[col]==i][label].sum()
            non_event_count=len(data[data[col]==i][label])-event_count
            rate_event = 1.0 * event_count / et
            rate_non_event = 1.0 * non_event_count / net
            event_dict[i]=[rate_event,rate_non_event]
        return event_dict
    def _split_bins(self):
        split_clf=Split(feature=self._temp,min_sample=self.min_sample,max_node_number=self._bins)
        split_clf.fit(self._df,self._label)
        self.bins=split_clf.bins

