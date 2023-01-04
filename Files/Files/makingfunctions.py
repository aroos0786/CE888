
import pandas as pd
import numpy as np
import csv
import sklearn
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import model_selection 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold
from sklearn import linear_model
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score # MSE and R squared
from numpy import std
from sklearn.metrics import confusion_matrix


# data viz
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

#from google.colab import drive
#drive.mount('/PythonProject')

############### Reading DataSet #######
class Dataset:
  def read_file(self):
   df = pd.read_csv('../../audit_data/audit_data/audit_risk.csv')
   describe_data = df.describe()
   header = df.columns.values.tolist()
   data = df.values
   self.dataframe = pd.DataFrame(data=df.values,columns=df.columns) #### Making DataFrame
   print(self.dataframe)

########### Pre-Processing DataSet ############
  def data_preprocessing(self):
    ############ Data Pre-Processing #########
     ##### Dropping Duplicates Value ######
    df1 = self.dataframe.drop_duplicates()
    df1
    df2 = df1.drop(['PROB','Prob'], axis = 1)
    df2.fillna(999, inplace=True)
    df2
       #Replacing the missing value by the median of the column
    df2['Money_Value'] = df2['Money_Value'].fillna(df2['Money_Value'].median())

    #Unique values in LOCATION_ID column
    df2["LOCATION_ID"].unique()
    print("These are the number of non-numeric values in LOCATION_ID: ", len(df2[(df2["LOCATION_ID"] == 'LOHARU') | (df2["LOCATION_ID"] ==  'NUH') | (df2["LOCATION_ID"] == 'SAFIDON')]))
    df2 = df2[(df2.LOCATION_ID != 'LOHARU')]
    df2 = df2[(df2.LOCATION_ID != 'NUH')]
    df2 = df2[(df2.LOCATION_ID != 'SAFIDON')]
    self.df2 = df2.astype(float)
    print("Updated number of rows in the dataset: ",len(df2))


    #Number of unique values in each columns
    for i in range(0, len(df2.columns)):
     print(df2.columns[i], ":", df2.iloc[:,i].nunique())

    #### Printing Duplicated Rows #####
    duplicatedrows = df2.duplicated().sum()
    print('Duplicated Rows:',duplicatedrows)

############# Plotting Dependent and Independent Variables #######
  def plot_variables(self):

   #### Plotting Dependent Variable Data ######
   target = self.df2.Risk.value_counts()
   targetplot = self.df2.Risk.value_counts().plot(kind="bar")
   targetplot
  
  def show_dataset_shape(self):
     ##### Showing Dataset Shape #####
    shape = self.df2.shape
    print('Dataset Shape:',shape)

  def separated_dependent_independent_variables(self):
     ###### Separating Dependent and Independent Varaibles #######
    self.X = self.df2.drop(["Risk"], axis = 1)
    self.Y = self.df2.drop(self.df2.iloc[:, :24], inplace=False, axis=1)
    print('Independent Variables:',self.X)
    print('Dependent Variables:',self.Y)

  def feature_scaling(self):
     #####Feature Scaling #########
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(self.X)
    print(X_scaled)

  def splitting_data(self):
      ###### Splitting Dataset ##########
    train_x,self.test_x,train_y,self.test_y = train_test_split(self.X, self.Y, test_size = 0.2)
    self.test_x,valid_x,self.test_y,valid_y = train_test_split(self.test_x,self.test_y, test_size = 0.5)
    print('Training Independent_Variable Shape:',train_x)
    print('\nTraining Dependent_Variable Shape:',train_y)
    print('\nValidation Independent_Variable Shape:',valid_x)
    print('\nValidation Dependent_Variable Shape:',valid_y)
    print('\nTesting Independent_Variable Shape:',self.test_x)
    print('\nTesting Dependent_Variable Shape:',self.test_y)



  def stratified_crossvalidation_randomforest(self):
       # Create StratifiedKFold for Random Forest #####
    clf = RandomForestClassifier(
     n_estimators=50,
     criterion='gini',
     max_depth=5,
     min_samples_split=2,
     min_samples_leaf=1,
     min_weight_fraction_leaf=0.0,
     max_features='auto',
     max_leaf_nodes=None,
     min_impurity_decrease=0.0,
     bootstrap=True,
     oob_score=False,
     n_jobs=-1,
     random_state=0,
     verbose=0,
     warm_start=False,
     class_weight='balanced'
)
    cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    lst_accu_stratified_new = []
  
    for train_index, test_index in cv.split(self.X,self.Y):     
     self.train_x, self.test_x = self.X.iloc[train_index], self.X.iloc[test_index]
     self.train_y, self.test_y= self.Y.iloc[train_index], self.Y.iloc[test_index]
     clf.fit(self.train_x, self.train_y.values.ravel())
     lst_accu_stratified_new.append(clf.score(self.test_x, self.test_y))
     #scores = cross_val_score(clf, self.X, np.ravel(self.Y), cv=cv)
     print('Stratified CV for Random Forest')
     print('\nList of possible accuracy:', lst_accu_stratified_new)
     print('\nMaximum Accuracy That can be obtained from this model is:',
      max(lst_accu_stratified_new)*100, '%')
     print('\nMinimum Accuracy:',
      min(lst_accu_stratified_new)*100, '%')
     print('\nOverall Accuracy/Mean:',
      mean(lst_accu_stratified_new)*100, '%')
     #print('\nStandard Deviation is:', lst_accu_stratified_new.std())
     print('\nStandard Deviation is:', std(lst_accu_stratified_new))
   
  def gridsearch_randomforest(self):

     #######Grid Search for Random Forest ######
   lst_accu_stratified_grid = []
      # Create a based model
   self.rf = RandomForestRegressor()
    # Create the parameter grid based on the results of random search 
   param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    }
   self.kf = KFold(n_splits=10)
   for train_index, test_index in self.kf.split(self.X,self.Y):
      x_train_fold, x_test_fold = self.X.iloc[train_index], self.X.iloc[test_index]
      y_train_fold, y_test_fold = self.Y.iloc[train_index], self.Y.iloc[test_index]
      grid_search = GridSearchCV(estimator = self.rf, param_grid = param_grid, cv=10)
  
    # Fit the grid search to the data
      grid_search.fit(self.train_x, np.ravel(self.train_y))
      grid_search.best_params_
      #y_pred = grid_search.best_estimator_.predict(self.train_x)
      lst_accu_stratified_grid.append(grid_search.score(self.test_x, self.test_y))
      accuracies = cross_val_score(estimator=grid_search, X=self.test_x, y=np.ravel(self.test_y))

     
      print('Grid Search 10-Fold CV')
      print('\nRandom Forest Grid Results')
      print('\nList of possible accuracy:', lst_accu_stratified_grid)
      print("tuned hpyerparameters :(best parameters) ",grid_search.best_params_)
      print('\nOverall Accuracy/Mean of Random Forest:',
      mean(lst_accu_stratified_grid)*100, '%')
      print("Standard Deviation:",accuracies.std())
      print("Mean:",accuracies.mean())
      #print("confusion_matrix:",(self.test_y, y_pred))


  def gridsearch_logisticregression(self):
     ###########Grid Search for Logistic Regression ########
     grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}
     logreg=LogisticRegression(solver='lbfgs', max_iter=1000)
     logreg_cv=GridSearchCV(logreg,grid,cv=10)
     logreg_cv.fit(self.train_x, np.ravel(self.train_y))
     logreg_cv.best_params_
     logistic_accuracy = cross_val_score(estimator=logreg_cv, X=self.test_x, y=np.ravel(self.test_y),cv=2)

   
     print('\nLogistic Regression Grid Results')
     print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
     print("accuracy :",logreg_cv.best_score_)
     print('\nOverall Accuracy/Mean of Logistic Regression:',
      (logreg_cv.best_score_), '%')
     print("Standard Deviation:",logistic_accuracy.std())

  def gridsearch_svm(self):
        ######## Grid Search for Support Vector Classifier ######

    # Set the parameters by cross-validation
    tuned_parameters =  [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000] },{'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]              

    for train_index, test_index in self.kf.split(self.X):
       train_x, test_x = self.X.iloc[train_index], self.X.iloc[test_index]
       train_y, valid_y= self.Y.iloc[train_index], self.Y.iloc[test_index]
       svc = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring='accuracy')
       svc.fit(self.train_x, np.ravel(self.train_y))
       svm_pred=svc.predict(self.test_x)
       svm_accuracy = cross_val_score(estimator=svc, X=self.test_x, y=np.ravel(self.test_y),cv=10)

    print('\n Support Vector Classifier Results')
    print("\t\taccuracy: {}".format(metrics.accuracy_score(self.test_y, svm_pred)))
    print('\nOverall Accuracy/Mean of Support Vector Classifier:',
      (svc.best_score_), '%')
    print('\nBest Parameters:',
      (svc.best_params_), '%')
    print("Standard Deviation:",svm_accuracy.std())







################ Class Object ########
c = Dataset()
### Run Functions ####
c.read_file()
c.data_preprocessing()
c.plot_variables()
c.show_dataset_shape()
c.separated_dependent_independent_variables()
c.feature_scaling()
c.splitting_data()
c.stratified_crossvalidation_randomforest()
c.gridsearch_randomforest()
c.gridsearch_logisticregression()
c.gridsearch_svm()

















