
# coding: utf-8

# In[1]:


from collections import OrderedDict
import numpy as np ,pandas as pd,math
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[106]:


class Select(object):
    def __init__(self,features=None,FeatureNum=0.8,task='classification'):
        '''
        Args:
            features:list,feature of correlation coefficient needed to be calculated 
            FeatureLimit:bool, if 1 ,calculated the iv for each feature and choose the lower to drop
            
        '''
        self.logfile = 'record.log'
        self.temp = features
        self.task=task
        self.FeatureNum=FeatureNum        
        self.logfile = 'record_importance.log'
    
    def _CheakLimit(self):
        '''Cheak the Feature Number We Want To Preserve
        '''
        if  self.FeatureNum > len(self.temp):
            raise ValueError('Do Not Need To Calculation')
        elif   self.FeatureNum < 0:
            raise ValueError('FeatureNum Could Not Be Negative')
        elif self.FeatureNum > 1:
            self.FeatureLimit= self.FeatureNum
        else:
            self.FeatureLimit=math.ceil( self.FeatureNum * len(self.temp))
        
    def _SetClassifier(self, clf=None):
        '''Set the classifier, Default use lightgbm
            Args:
                clf:the classifier, Default use lightgbm( we can use xgboost,ensamble tree ,etc.)
        '''
        if clf != None:
            self.clf = clf
        else:
            if self.task =='classification':
                self.clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1)
            elif task == 'regression':
                self.clf = lgb.LGBMRegressor(n_estimators=1000, learning_rate = 0.05, verbose = -1)
            else:
                raise ValueError('Task must be either "classification" or "regression"')
                
    def fit(self,df, lable,clf=None):
        '''
        Args:
            df: pandas dataframe include all features and lable.
            lable: str, lable name
        '''
        self.df = df
        self.lable = lable
        if not self.temp:
            self.temp=list(self.df.columns)
            self.temp.remove(self.lable)
        for feature in self.temp:
            if feature not in self.df.columns:
                raise ValueError('Features Must Be In Data') 
        if (self.lable is None) | (self.lable not in self.df.columns):
            raise ValueError('Input Data Incorrect. Importance Methods Are Not available')

        self. _CheakLimit()
        self._SetClassifier(clf)
        
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n%{}%\n'.format('Start!','-'*60))
        self._CheakLimit
        a = _importsance_selection(df = self.df, clf = self.clf,
                                    lable = self.lable,
                                    start = self.temp,FeatureLimit=self.FeatureLimit)
        best_features_comb=a.select()
        self.best_features_comb = a._selectcol
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n{}\n%{}%\n'.format('Done',self.best_features_comb,'-'*60))
        return self.best_features_comb


# In[111]:


class  _importsance_selection(object):
    def __init__(self,df,lable,clf,start,FeatureLimit):
        '''
        Args:
        '''
        self._df= df
        self._lable=lable
        self._clf=clf
        self._temp = start
        self._featureNum=FeatureLimit
        self._selectcol=[]
        self._zero=np.pi
    def select(self):
        while (len(self._selectcol) < self._featureNum):# | (self._zero != len(self._temp)):
            temp = self._temp[:]  
            self._clf.fit(self._df[temp],self._df[self._lable])
            importances = sorted([[i,j] for i,j in zip(self._clf.feature_importances_,list(OrderedDict.fromkeys(temp)))])   
            self._zero=(self._clf.feature_importances_ ==0).sum()
            self._temp.remove(importances[-1][-1])
            self._selectcol.append(importances[-1][-1])
            #print(self._clf.feature_importances_,)
        return self._selectcol

