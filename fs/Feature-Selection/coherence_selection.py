
# coding: utf-8

# In[73]:


from collections import OrderedDict
# from scipy.spatial.distance import correlation
import numpy as np ,pandas as pd
import math,time
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from  IV_Calculation import *
import warnings
warnings.filterwarnings('ignore')


# In[108]:


class Corr_Select(object):
    def __init__(self,features=None,iv=None,method= 1, CorrLimit=0.7,plot=0):
        '''
            use Pearsonr/Distance Correlation to calculated correlation coefficient,
            calculated iv and drop the lower iv between the two features with high correlation
            
        Args:
            features:list,feature of correlation coefficient needed to be calculated 
            iv:bool, if 1 ,calculated the iv for each feature and choose the lower to drop
            method:bool, if 1 ,use Pearsonr to calculated correlation coefficient/0 for Distance Correlation
            CorrLimit: float, the threshold of correlation coefficent.
        '''
        self.logfile = 'record.log'
        self.temp = features
        self.iv= iv
        self.method=method
        self.plot=plot
        self.logfile =  'record_correlation.log'
        if (CorrLimit>0) &( CorrLimit<=1):
            self.CorrLimit = CorrLimit
        else:
            raise ValueError('The Corrlimit Is Incorrect')
        
    def fit(self,df, label=None,positive=None):
        '''Import pandas dataframe to the class
        Args:
            df: pandas dataframe include all features and label.
            label: str, label name
            positive: default None , for IV calculated,choose the label value to be positive ,others would be negative
        '''
        self.df = df
        self.label = label
        if not self.temp:
            self.temp=self.df.columns
            if self.label:
                self.temp=list(self.temp)
                self.temp.remove(self.label)
        if self.iv and (not self.label):
            raise ValueError('Need A label For Calculate IV')
        elif (positive==None) and self.label and self.iv :
            if (len(np.unique(self.df[self.label])) != 2):
                raise ValueError('label Must Be Binary') 
        self.positive=positive    
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n{}\n%{}%\n'.format('Start!',time.ctime(),'-'*100))
        a = _coherence_selection(df = self.df, label = self.label,
                                    start = self.temp,all_col=self.df.columns,CorrLimit=self.CorrLimit,
                                    iv=self.iv,method=self.method,positive=self.positive,plot=self.plot)
        a.select()
        self.best_features = a._best
        self._tril=a._tril
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n{}\n%{}%\n'.format('Done',self.best_features,'-'*100))
        return self.best_features


# In[120]:


class _coherence_selection(object):
    def __init__(self,df, start,all_col,label,CorrLimit,iv,method,positive,plot):
        self._df = df
        self._temp, self._label = start, label
        self._Startcol = ['None']
        self._all=all_col
        self._CorrLimit=CorrLimit
        self._iv=iv
        self._method=method
        self._positive=positive
        self._plot=plot
        
    def select(self):
        print(self._temp)
        self._keepdowner(self._df[self._temp])  
        drop_list,save_list,save_list_temporary=[],[],[]
        if self._iv:          
            iv_arg=IV_Select(bins=5,woe_limit=5,min_sample = 1)
            iv_arg.fit(self._df[self._all],self._label,self._positive)
            IV_dict=iv_arg.iv_dict
            list_dict={key: value for key, value in IV_dict.items() if key in self._to_drop}
            self._to_drop=dict(sorted(list_dict.items(),key = lambda x:abs(x[1]),reverse = True)).keys()
            for column in self._to_drop:
                corr_features = list(self._downer.index[self._downer[column].abs() > self._CorrLimit])
                if column in save_list:
                    drop_list.extend(corr_features)
                elif column not in drop_list:
                    save_list.append(column)
                    drop_list.extend(corr_features)
                else:                    
                    for col in corr_features: 
                        if col not in drop_list:  
                            save_list_temporary.append(col)  
                if column in save_list_temporary:
                    drop_list.extend(corr_features)
        else:
            corr_matrix=self._tril.copy()
            save_list=list(corr_matrix.columns)
            while corr_matrix.abs().max().max() > self._CorrLimit:
                col_delete = corr_matrix[corr_matrix.abs().max() == corr_matrix.abs().max().max()].abs().sum(axis = 1).argmax()
                drop_list.append(col_delete)
                save_list.remove(col_delete)
                corr_matrix=corr_matrix.loc[save_list,save_list]
        self._drop_list=drop_list
        self._best=[column for column in self._downer.columns if column not in self._drop_list]
        if self._plot:
            self._tril.fillna(0,inplace=True)
            self._plot_collinear(self._tril)
            self._plot_collinear(self._tril.loc[self._best,self._best])
            
    def _keepdowner(self,data):
        '''use pandas to calculated the feature with high correlation coefficent
           Preserving upper triangular matrix,Select features need to be deleted based on the threshold 
        '''
        if self._method:
            self._corr_matrix=data.corr() # Compute pairwise correlation of columns
        else :
            self._CorrDist(data)
        self._downer = self._corr_matrix.where(np.tril(np.ones(self._corr_matrix.shape), k = -1).astype(np.bool)) 
        self._tril=self._corr_matrix.where(np.eye(self._corr_matrix.shape[0])!=1) 
        self._to_drop = [column for column in self._tril.columns if any(self._tril[column].abs() >= self._CorrLimit)]

    def _CorrDist(self,data):
        temp_downer=pd.DataFrame(index=self._df[self._temp].columns,columns=self._df[self._temp].columns)
        for i in range(data.shape[1]):
            for j in range (i,data.shape[1]):
                dis_corr=self._distance(data.iloc[:,i],data.iloc[:,j])
                temp_downer.iloc[j,i]=dis_corr
                temp_downer.iloc[i,j]=dis_corr
        self._corr_matrix=temp_downer
    
    def _distance(self,X,Y):
        ''' Calculated the Correlation Distance Between Two 1-D arrays.
        '''
        X ,Y= X[:, None],Y[:, None]
        n = X.shape[0]
        a ,b= squareform(pdist(X)),squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
        dcov2_xy,dcov2_xx,dcov2_yy = (A * B).sum()/float(n * n),(A * A).sum()/float(n * n),(B * B).sum()/float(n * n)
        dcorr = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcorr
    def _plot_collinear(self,data):
        '''Heatmap of the correlation values. 
        '''        
        corr_matrix_plot = data
        title = 'Correlations'     
        f, ax = plt.subplots(figsize=(10, 8))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,linewidths=.25, cbar_kws={"shrink": 0.6})
        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size = int(160 / corr_matrix_plot.shape[0]));
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size = int(160 / corr_matrix_plot.shape[1]));
        plt.title(title, size = 14)
        plt.show()

