
# coding: utf-8

# In[1]:


from collections import OrderedDict
from sklearn.tree import DecisionTreeClassifier
import numpy as np ,pandas as pd,math
import warnings
warnings.filterwarnings('ignore')


# In[134]:


class Split(object):
    '''
        Separate box operation based on gini coefficient,Applicable to classification tasks
    '''
    def __init__(self,feature=None,frac=0.05,min_sample = 1,threshold=None , max_node_number = 6):
        '''
        Args:
            features: list, the starting features 
            frac: float,The minimum size of a sample to be sectioned
            min_sample : int, The minimum size of a sample to be sectioned ,if set min_sample, ignore frac
            threshold: float,The threshold of the next split .if less then ,stop
            max_node_number:int,The number of bins
        '''
        self.temp=feature       
        self.threshold=threshold
        self.nodenumber=max_node_number
        self.logfile = 'record_split.log'
        if isinstance(frac,float) & (frac<1) & (frac>0) & min_sample == 1:
            self.frac=frac
            self.leafnumber= None
        elif isinstance(min_sample,int):
            self.leafnumber=min_sample
            self.frac=None
        else:
            raise ValueError('The minimum size of  sample is Incorrect ')
        
    def fit(self,df,lable):
        '''Import pandas dataframe to the class
        Args:
            df: pandas dataframe ,include all features and lable.
            lable: str, lable name
        '''
        self.df = df
        self.lable = lable
        self.shape = self.df.shape
        if not self.temp:
            self.temp=list(self.df.columns)
            self.temp.remove(lable)
        for feature in self.temp:
            if feature not in self.df.columns:
                raise ValueError('Features Must Be In Data') 
        if (self.lable is None) | (self.lable not in self.df.columns):
            raise ValueError('Input Data Incorrect. Gini Based Methods Are Not available')
        if self.frac:
            self.minsample=self.frac*self.shape[0]
        else:
            self.minsample=self.leafnumber
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n%{}%\n'.format('Start!','-'*60))
        self.bins={}
        for _ in self.temp:
            self.data=self.df[_].values
            result=_find_node(data=self.data,lable=self.df[lable],threshold=self.threshold,minsample=self.minsample,nodenumber=self.nodenumber)
            best_node=result.select()
            self.bins[_]=result._node
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n{}\n%{}%\n'.format('Done',self.bins,'-'*60))
        return self.bins


# In[142]:


class _find_node():
    '''use sklearn.tree ,base on cart 
    '''
    def __init__(self,data,lable,threshold,minsample,nodenumber):
        '''Args:
                data: np.array like, the data need to be dividem, shape like (-1,1)
                lable: np.array like ,shape like (-1,1)
                number: int,How many bins are divided
                threshold: float,the gini  impurity threshold, if less then just stop 
                frac: float:the is a fraction and`ceil(min_samples_leaf * n_samples)`are the minimum
      number of samples for each node.
                leafnumber:int,The minimum number of samples required to be at a leaf node.
        '''    
        self._data=data
        self._lable=lable
        self._threshold=threshold
        self._leaflimit=math.ceil(minsample)
        self._nodenumber=nodenumber
        self._len=self._data.shape[0]
        
    def Calculation(self,data,lable):        
        '''Divide the data 
        Attributes:
                self.node: the node that divide the data
                self.impurity: the impurity of two node
        '''
        clf=DecisionTreeClassifier(max_depth=self._deep,min_samples_leaf=self._leaflimit).fit(data.reshape(-1,1),lable)
        self._impurity=clf.tree_.impurity
        self._node=clf.tree_.threshold
        return self._impurity , self._node
    
    def select(self):
        '''Determine the number of sub-boxes
                Attributes:
                self.n:the number of the leaflimit
        '''
        self._n=self._nodenumber
        self._deep=math.ceil(math.log2(self._n))
        impurity,node=self.Calculation(self._data,self._lable)# -2 in the node mean the leaf
        self._node=list(node[node!=-2])
        leaf_impurity=impurity[node==-2] # the leaf's impurity
        impurity=list(impurity)
        while len(self._node)>self._n:
            gini_score=np.inf
            impurity=self._cheakpurity(impurity,node,gini_score)
            
    def _datasplit(self,node,data,leble):
        '''Segmentation of data based on nodes 
        '''
        data_left,data_right=data[data<=node],data[data>node]
        lable_left,lable_right=lable[data<=node],lable[data>node]
        return data_left,lable_left,data_right,lable_right
        
    def _cheakpurity(self,impurity,node,gini_score):
        '''cheak the impurity if the diff impurity is to big ,drop the node
        '''
        for _ in range(len(impurity)):
            if node[_]== -2 and node[_ -1] !=-2:
                father_node,left_child_node,right_child_node=node[_ -1],node[_],node[_+1]
                father_impurity,left_child_impurity,right_child_impurity=impurity[_ -1],impurity[_],impurity[_+1]
                cheak_impurity=min((father_impurity-left_child_impurity),(father_impurity-right_child_impurity))
                if gini_score>cheak_impurity:
                    gini_score =cheak_impurity
                    remove_node= _-1
        self._node.remove(node[remove_node])
        impurity.pop(remove_node),impurity.pop(remove_node),impurity.pop(remove_node)
        return impurity

