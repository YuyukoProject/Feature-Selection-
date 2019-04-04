
# coding: utf-8

# In[150]:


import numpy as np,pandas as pd
import time,math
from sklearn.feature_selection import  RFE,RFECV,chi2
from sklearn.feature_selection import mutual_info_,mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import SelectFdr,SelectFpr,SelectFwe,VarianceThreshold
from sklearn.feature_selection import f_classif,f_oneway,f_regression
from sklearn.feature_selection import  SelectKBest ,SelectPercentile,SelectFromModel
import warnings
warnings.filterwarnings('ignore')




''' use sklearn.feature_selection ,
'''


class RFE_Select(RFE):
    def __init__(self,estimator, n_features_to_select=None, step=1, verbose=0):
        '''
            Feature ranking with recursive feature elimination.

        '''
        self.estimator=estimator
        self.n_features_to_select=n_features_to_select
        self.step=step
        self.verbose=verbose



class RFECV_Select(RFECV):
    def __init__(self,estimator, step=1, min_features_to_select=1, cv='warn', scoring=None, verbose=0, n_jobs=None):
        self.estimator=estimator
        self.min_features_to_select=min_features_to_select
        self.step=step
        self.cv=cv
        self.scoring=scoring
        self.verbose=verbose
        self.n_jobs=n_jobs




class F_class_Select(object):
    def __init__(self,k=None):
        '''
        k :int /float , The number of feature you want to select.
        '''
        self._k=k
        
    def fit(self,x,y):
        '''x:dataframe like.
           y:
        '''
        assert x.shape[0]==y.shape[0]
        self._columns=x.columns
        if self._k:
            if (self._k > 0 )& (self._k < 1):
                self.n= self._k
            elif not isinstance(self._k,int):
                raise ValueError('K maybe need to be int')
            elif self._k >= x.shape[1] :
                raise ValueError('K is too large')
            elif (self._k >= 1) & (self._k < x.shape[1]):
                self.n=self._k
            else:
                raise ValueError('K is wrong')
        self._x=x
        self._y=y
        if self.n:
            if isinstance(self.n,int):
                clf=SelectKBest(f_classif,self.n)
                clf.fit_transform(self._x.values,self._y)
                self.col=self._x.columns[clf.get_support()]
                self.F,self.P=clf.scores_,clf.pvalues_
            else:
                clf=SelectPercentile(f_classif,self.n*100)
                clf.fit_transform(self._x.values,self._y)
                self.col=self._x.columns[clf.get_support()]   
                self.F,self.P=clf.scores_,clf.pvalues_
        else:
            self.F,self.P=f_classif(self._x,self._y)
            self.col=self.x.columns[self.P<0.05]



class Chi2_Select(object):
    def __init__(self,k=None,sig=0):
        self._k=k
        self._sig=sig
        
    def fit(self,x,y):
        '''x:dataframe like.
           y:
        '''
        assert x.shape[0]==y.shape[0]
        self._columns=x.columns
        if self._k:
            if (self._k > 0 )& (self._k < 1):
                self.n= self._k
            elif not isinstance(self._k,int):
                raise ValueError('K maybe need to be int')
            elif self._k >= x.shape[1] :
                raise ValueError('K is too large')
            elif (self._k >= 1) & (self._k < x.shape[1]):
                self.n=self._k
            else:
                raise ValueError('K is wrong')
        self._x=x
        self._y=y
        if self.n:
            if isinstance(self.n,int):
                clf=SelectKBest(chi2,self.n)
                clf.fit_transform(self._x.values,self._y)
                self.col=self._x.columns[clf.get_support()]
                self.F,self.P=clf.scores_,clf.pvalues_
            else:
                clf=SelectPercentile(chi2,self.n*100)
                clf.fit_transform(self._x.values,self._y)
                self.col=self._x.columns[clf.get_support()]   
                self.F,self.P=clf.scores_,clf.pvalues_
        else:
            self.F,self.P=chi2(self._x,self._y)
            self.col=self._x.columns[self.P<0.05]


class F_regression_Select(object):
    def __init__(self,k=None,sig=0):
        self._k=k
        self._sig=sig
        
    def fit(self,x,y):
        '''x:dataframe like.
           y:
        '''
        assert x.shape[0]==y.shape[0]
        self._columns=x.columns
        if self._k:
            if (self._k > 0 )& (self._k < 1):
                self.n= self._k
            elif not isinstance(self._k,int):
                raise ValueError('K maybe need to be int')
            elif self._k >= x.shape[1] :
                raise ValueError('K is too large')
            elif (self._k >= 1) & (self._k < x.shape[1]):
                self.n=self._k
            else:
                raise ValueError('K is wrong')
        self._x=x
        self._y=y
        if self.n:
            if isinstance(self.n,int):
                clf=SelectKBest(f_regression,self.n)
                clf.fit_transform(self._x.values,self._y)
                self.col=self._x.columns[clf.get_support()]
                self.F,self.P=clf.scores_,clf.pvalues_
            else:
                clf=SelectPercentile(f_regression,self.n*100)
                clf.fit_transform(self._x.values,self._y)
                self.col=self._x.columns[clf.get_support()]      
                self.F,self.P=clf.scores_,clf.pvalues_
        else:
            self.F,self.P=f_regression(self._x,self._y)
            self.col=self._x.columns[self.P<0.05]

class MIC_Select(object):
    def __init__(self,k=None,sig=0):
        self._k=k
        self._sig=sig
        
    def fit(self,x,y):
        '''x:dataframe like.
           y:
        '''
        assert x.shape[0]==y.shape[0]
        self._columns=x.columns
        if self._k:
            if (self._k > 0 )& (self._k < 1):
                self.n= self._k
            elif not isinstance(self._k,int):
                raise ValueError('K maybe need to be int')
            elif self._k >= x.shape[1] :
                raise ValueError('K is too large')
            elif (self._k >= 1) & (self._k < x.shape[1]):
                self.n=self._k
            else:
                raise ValueError('K is wrong')
        self._x=x
        self._y=y
        if self.n:
            if isinstance(self.n,int):
                clf=SelectKBest(mutual_info_classif,self.n)
                clf.fit_transform(self._x.values,self._y)
                self.col=self.x.columns[clf.get_support()]
                self.F,self.P=clf.scores_,clf.pvalues_
            else:
                clf=SelectPercentile(mutual_info_classif,self.n*100)
                clf.fit_transform(self._x.values,self._y)
                self.col=self._x.columns[clf.get_support()]   
                self.F,self.P=clf.scores_,clf.pvalues_
        else:
            self.F,self.P=mutual_info_classif(self._x,self._y)
            self.col=self._x.columns[self.P<0.05]


class From_model_Select(object):
    def __init__(self,task,model_name,):
        self.task=task
        self.model_name=model_name
        

    def get_feature_selection_model_from_name(self,type_of_estimator):
        model_map = {
            'classifier': {
                'RandomForest': RandomForestClassifier(n_jobs=-1, max_depth=10, n_estimators=15,),
                'a': GenericUnivariateSelect(),
                },
            'regressor': {
                'SelectFromModel': SelectFromModel(RandomForestRegressor(n_jobs=-1, max_depth=10, n_estimators=15), threshold='0.7*mean'),
                'GenericUnivariateSelect': GenericUnivariateSelect(),
            }
                }
        self.model=model_map[type_of_estimator][self.model_name] 

    def fit(self, X, Y):
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.feature_selection import SelectFromModel

            num_features = X.shape[1]
            max_features = int(
                float(self.max_features) * (np.log(num_features) + 1))
            # Use at most half of the features
            max_features = max(1, min(int(X.shape[1] / 2), max_features))
            preprocessor = ExtraTreesRegressor(
                n_estimators=self.n_estimators, criterion=self.criterion,
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap,
                max_features=max_features, max_leaf_nodes=self.max_leaf_nodes,
                oob_score=self.oob_score, n_jobs=self.n_jobs, verbose=self.verbose,
                random_state=self.random_state)
            preprocessor.fit(X, Y)
            self.preprocessor = SelectFromModel(preprocessor, prefit=True)

            return self 


# In[ ]:


def fit(self, X, Y, sample_weight=None):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        num_features = X.shape[1]
        max_features = int(
            float(self.max_features) * (np.log(num_features) + 1))
        # Use at most half of the features
        max_features = max(1, min(int(X.shape[1] / 2), max_features))
        preprocessor = ExtraTreesClassifier(
            n_estimators=self.n_estimators, criterion=self.criterion,
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap,
            max_features=max_features, max_leaf_nodes=self.max_leaf_nodes,
            oob_score=self.oob_score, n_jobs=self.n_jobs, verbose=self.verbose,
            random_state=self.random_state, class_weight=self.class_weight
        )
        preprocessor.fit(X, Y, sample_weight=sample_weight)
        self.preprocessor = SelectFromModel(preprocessor, prefit=True)
        return self 


# In[ ]:


def analyseReasonWithTreeBaesd(anamolySample,normalSample,name):
    data = anamolySample
    target = []
    for i in range(0,len(anamolySample)):
        target.append(1)
    data.extend(normalSample)
    for i in range(0,len(normalSample)):
        target.append(0)
    clf = ExtraTreesClassifier()
    clf = clf.fit(data,target)   
    model = SelectFromModel(clf,prefit=True) 
    outcome = model.get_support()
    for i in range(0,len(name)):
        if outcome[i]:
            print (name[i])


def analyseReasonWithTreeBaesd(anamolySample,normalSample,name):
    data = anamolySample
    target = []
    for i in range(0,len(anamolySample)):
        target.append(1)
    data = data.append(normalSample)
    for i in range(0,len(normalSample)):
        target.append(0)
    clf = ExtraTreesClassifier()
    clf = clf.fit(data,target)   
    model = SelectFromModel(clf,prefit=True) 
    outcome = model.get_support()
    for i in range(0,len(name)):
        if outcome[i]:
            print (name[i])




def analyseReasonWithTreeBaesd(anamolySample,normalSample):
    data = anamolySample
    target = []
    for i in range(0,len(anamolySample)):
        target.append(1)
    data = data.append(normalSample)
    for i in range(0,len(normalSample)):
        target.append(0)
    name = []
    for i in data.columns:
        name.append(i)
    clf = ExtraTreesClassifier()
    clf = clf.fit(data,target)   
    model = SelectFromModel(clf,prefit=True) 
    outcome = model.get_support()
    for i in range(0,len(name)):
        if outcome[i]:
            print (name[i]) 

