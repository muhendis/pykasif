import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

class DataCleaning:
  
  """
  Bir veri setinin temizlenmesi için kullanılan bir sınıf

  """
  def __init__(self,df_train:pd.core.frame.DataFrame=None,
                    df_test:pd.core.frame.DataFrame=None):
    """
    Parameters
    ----------
    df_train : pd.core.frame.DataFrame
        kullanılacak model eğitiminde kullanılacak veri seti
    df_test : pd.core.frame.DataFrame
        kullanılacak model testinde kullanılacak veri seti
    """

    self.df_train = df_train.copy()

    self.df_test = df_test.copy()

    self.variables = df_train.columns.to_list()

  def show_duplicate_observations(self):
    """
    Yinelenen satırları analiz eder.

    Returns
    -------
    dict
        df_train ve df_test Veri Çerçevesindeki yinenlenen satırları gösterir.

    """

    df_train_duplicate = self.df_train.duplicated()

    df_test_duplicate = self.df_test.duplicated()

    return {"df_train" : self.df_train[df_train_duplicate],
    "df_test" :self.df_test[df_test_duplicate]}


  def remove_duplicate_observations(self):
    """
    Yinelenen satırları siler.
    """

    self.df_train.drop_duplicates(inplace = True)

    self.df_test.drop_duplicates(inplace = True)


  def show_missing_values(self):
    """
    Kayıp değerleri sütun bazında miktar verir.

    Returns
    -------
    dict
        df_train ve df_test Veri Çerçevesindeki sütunlardaki kayıp hücre sayı

    """

    df_train_missing_cell = self.df_train.isna().sum()
    
    df_test_missing_cell = self.df_test.isna().sum()

    return {"df_train" : df_train_missing_cell,
    "df_test" :df_test_missing_cell}



  def missing_values_treatment(self,feature=None,strategy="delete",n_neighbors=3):
    """
    Kayıp hücreleri tedavi ediyor. Kullanılan yöntemler:
      - Sütunu silme
      - Ortalama ile doldurma tedavisi
      - Mod  ile doldurma tedavisi
      - Medyan  ile doldurma tedavisi
      - K-NN Algoritmasının tahmini ile doldurma tedavisi


    Parameters
    ----------
    feature : string
        tedavi edilecek sütun/özellik ismi
    strategy : string
        tedavi yöntemi
    """


    train_notna_index= None
    test_na_index = None
    train_na_index = None

    if strategy != "KNN":
    
      train_notna_index = self.df_train.loc[:,feature].notna().values

      test_na_index = self.df_test.loc[:,feature].isna().values

      train_na_index = self.df_train.loc[:,feature].isna().values

    if strategy == "delete":

      self.df_train=self.df_train.drop(columns=feature)

      self.df_test=self.df_test.drop(columns=feature)

    elif strategy == "mean":

      mean_for_missing_values = self.df_train.loc[train_notna_index,feature].mean()

      print("Mean : ",mean_for_missing_values)

      self.df_train.loc[train_na_index,feature] = mean_for_missing_values

      self.df_test.loc[test_na_index,feature] = mean_for_missing_values
    
    elif strategy == "mode":

      mode_for_missing_values = self.df_train.loc[train_notna_index,feature].mode()[0]
      
      print("Mode : ",mode_for_missing_values)

      self.df_train.loc[train_na_index,feature] = mode_for_missing_values

      self.df_test.loc[test_na_index,feature] = mode_for_missing_values

    elif strategy == "median":

      median_for_missing_values = self.df_train.loc[train_notna_index,feature].median()

      print("Median : ",median_for_missing_values)

      self.df_train.loc[train_na_index,feature] = median_for_missing_values

      self.df_test.loc[test_na_index,feature] = median_for_missing_values
    
    elif strategy =="KNN" :
      imputer = KNNImputer(n_neighbors=n_neighbors)

      imputer.fit(self.df_train)

      self.df_train=pd.DataFrame(data=imputer.transform(self.df_train),columns=self.variables)

      self.df_test=pd.DataFrame(data=imputer.transform(self.df_test),columns=self.variables)


  def outlier_detection(self,feature=None,
                        strategy="inter_quartile_range",
                        n_estimators=50,
                        max_samples='auto',
                        contamination=0.10):
    """
    Aykırı değerleri tespit etme.Kullanılan yöntemler:
      - Çeyrekler arası aralık
      - İzolasyon Ormanı

    Parameters
    ----------
    feature : string
        tedavi edilecek sütun/özellik ismi
    strategy : string
        tespit yöntemi    
    n_estimators : int
        ormanda kurulacak ağaç sayısını ifade eder.(izolasyon Ormanı yöntemi ile ilgili)
    max_samples : string
        her bir ağacı eğitmek için çekilecek örnek sayısını belirtir.(izolasyon Ormanı yöntemi ile ilgili)
    contamination : float
        veri setindeki aykırı değerlerin beklenen oran(izolasyon Ormanı yöntemi ile ilgili)

    Returns
    -------
    dict
        df_train ve df_test Veri Çerçevesindeki  aykırı satırlar / örnekler

    """

    df_train_outliers = None
    df_test_outliers = None

    if strategy == "inter_quartile_range":

        Q1 = self.df_train.loc[:,feature].quantile(0.25)
        Q3 = self.df_train.loc[:,feature].quantile(0.75)     
        IQR = Q3 - Q1
        lower_limit = Q1-1.5*IQR
        upper_limit = Q3+1.5*IQR

        print("Alt sınır : ",lower_limit,"\nÜst sınır : ",upper_limit)

        out_train = np.array(self.df_train[feature] > lower_limit) & (self.df_train[feature] < upper_limit) == False

        out_test = np.array(self.df_test[feature] > lower_limit) & (self.df_test[feature] < upper_limit) == False
        
        #aykırı değerler  
        df_train_outliers = self.df_train.loc[out_train]
        df_test_outliers = self.df_test.loc[out_test]

    
    elif strategy == "isolation_forest":

        forest_model=IsolationForest(n_estimators=n_estimators, 
        contamination=contamination)

        forest_model.fit(self.df_train.loc[:,feature].values.reshape(-1,1))

        pred_value_for_train = forest_model.predict(self.df_train.loc[:,feature].values.reshape(-1,1))
        pred_value_for_test = forest_model.predict(self.df_test.loc[:,feature].values.reshape(-1,1))

        # eğer pred_value 1 ise normal, -1 ise anormal 
        out_train_ind=np.where(pred_value_for_train==-1)[0].tolist()
        out_test_ind=np.where(pred_value_for_test==-1)[0].tolist()
                
        #aykırı değerler  
        df_train_outliers = self.df_train.iloc[out_train_ind,:]
        df_test_outliers = self.df_test.iloc[out_test_ind,:]

    
    return {"df_train" : df_train_outliers,
    "df_test" :df_test_outliers}

 
    """
    Mu metodları siz araştırın.....
    ---> Z-Score ile outlier tespiti 
    ---> strategy == "DBSCAN":
    # bu metouda siz araştırın
    #https://towardsdatascience.com/dbscan-a-density-based-unsupervised-algorithm-for-fraud-detection-887c0f1016e9
    model = DBSCAN(eps = 2, min_samples = 2).fit(df_train.loc[:,"v_1"].values.reshape(-1,1))
    outliers = df_train[model.labels_ == -1]
    model.labels_ 
    """
