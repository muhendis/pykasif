import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway  # for ANOVA


class FeatureSelection:
  """
  Bir veri setinin özniteliklerini seçmek için kullanılan bir sınıf
  
  """


  def __init__(self, df: pd.core.frame.DataFrame = None,
              continuous_variables: list = None,
              categorical_variables: list = None,
              target_variable: str = None):
      """
      Parameters
      ----------
      df : pd.core.frame.DataFrame
          kullanıcalak veri seti
      continuous_variables : list
          sürekli değişkenler
      categorical_variables : list
          kategorik değişkenler
      target_variable : list
          hedef değişkenler
      """

      self.df = df

      self.variables = df.columns.to_list()

      self.__continuous_variables = self.set_continuous_variables(continuous_variables)

      self.__categorical_variables = self.set_categorical_variables(categorical_variables)

      self.__target_variable = self.set_target_variable(target_variable)

  def __check_it_includes(self, main_list: list = None, sub_list: list = None):
      """ Bir alt listede yer alan bütün elemanların, ana listede olup
      olmadığını kontrol eder.

      Parameters
      ----------
      main_list : list
          ana liste
      sub_list : list
          alt liste

      Returns
      -------
      boolean
          Alt listedeki tüm elemanlar ana listedeyse True, değilse False
      """

      result = True

      for sub_i in sub_list:
          if not (sub_i in main_list):
              result = False
              break
      return result
      
  def set_continuous_variables(self, continuous_variables):
      """Sürekli değişkenleri ayarlamada kullanılır

      Parameters
      ----------
      continuous_variables : list
          sürekli değişkenler

      Assertions
      ------
      AssertionError
          continuous_variables degiskenin değerleri,  veri setinde tanımlanmamışsa.

      Returns
      -------
      list
          sürekli olan değişkenlerin listesi
      """

      assert (self.__check_it_includes(self.variables,
                                      continuous_variables)), "continuous_variables degiskenin değerleri,  veri setinde tanımlanmalıdır."

      return continuous_variables


  def get_continuous_variables(self):
      """Sürekli değişkenleri dönderir

      Returns
      -------
      list
          sürekli olan değişkenlerin listesi
      """

      return self.__continuous_variables


  def set_categorical_variables(self, categorical_variables):
      """Kategorik değişkenleri ayarlamada kullanılır

      Parameters
      ----------
      categorical_variables : list
          kategorik değişkenler

      Assertions
      ------
      AssertionError
          categorical_variables degiskenin değerleri,  veri setinde tanımlanmamışsa.


      Returns
      -------
      list
          kategorik olan değişkenlerin listesi
      """

      assert (self.__check_it_includes(self.variables,
                                      categorical_variables)), "categorical_variables degiskenin değerleri,  veri setinde tanımlanmalıdır."

      return categorical_variables


  def get_categorical_variables(self):
      """Kategorik değişkenleri dönderir

      Returns
      -------
      list
          kategorik olan değişkenlerin listesi
      """

      return self.__categorical_variables

      #


  def set_target_variable(self, target_variable):
      """Hedef değişkenleri ayarlamada kullanılır

      Parameters
      ----------
      target_variable : list
          hedef değişkenler

      Assertions
      ------
      AssertionError
          target_variable degiskenin değerleri,  veri setinde tanımlanmamışsa.


      Returns
      -------
      list
          hedeflenen değişkenlerin listesi
      """

      assert (self.__check_it_includes(self.variables,
                                      [target_variable])), "target_variable degiskenin değerleri,  veri setinde tanımlanmalıdır."

      return target_variable


  def get_target_variable(self):
      """Hedef değişkenleri dönderir

      Returns
      -------
      list
          hedeflenen değişkenlerin listesi
      """

      return self.__target_variable




  def correlation(self, threshold_for_target=0.5):
      """Hedef değişken ile diğer değişkenler arasındaki korelasyon katsayısı 
      bulunur.Belli bir eşik değer üstündeki (varsayılan , 0.5'dir.) katsayıya 
      sahip tahmin edici değişkenler tespit edilir. Ve bu tespit edilen tahmin 
      edici değişkenler arasındaki korelasyon katsayısına bakılır.

      (Not: Hedef değişkenin bir sürekli değişken olduğu kabul edilerek yazılmıştır.)

      Parameters
      ----------
      threshold_for_target : int
          ilişkinin olup olmadığını belirleyen katsayı değeri
      """

      # Korelasyon matrisini oluşturma
      correlation_data = pd.concat([self.df[self.__continuous_variables], self.df[self.__target_variable]], axis=1).corr()
      # Yalnızca Hedef Değişken ile mutlak korelasyonun > threshold_for_target (0.5) olduğu sütunları filtreleme
      selected_corr_list = correlation_data[self.__target_variable][abs(correlation_data[self.__target_variable]) > threshold_for_target][:-1]
      print("Hedef değişken ile korelasyonu yüksek olan değişkenler :\n",selected_corr_list,"\n")
      print("\nDiğer değişkenlerin kendi aralarındaki korelasyon : \n", self.df[selected_corr_list.index.values.tolist()].corr())


  def ANOVA_test(self, variable=None):
      """Hedef değişken ile kategorik değişken arasında bir ilişki olup olmadığını 
      bulur. Analiz sonucunda elde edilen P-değeri 0.05'den küçük ise ilgili
      değer ile hedef değişken arasında bir ilişki vardır.

      (Not: Hedef değişkenin bir sürekli değişken olduğu kabul edilerek yazılmıştır.)

      Parameters
      ----------
      variable : str
          ANOVA_test'i için bir kategorik değişken
      """

      print('##### ANOVA Sonucu ##### \n')

      category_group_list = self.df.groupby(variable)[self.__target_variable].apply(list)

      anova_result = f_oneway(*category_group_list)

      # ANOVA P-Değeri <0,05 ise, bu, H0'ı reddettiğimiz anlamına gelir
      if (anova_result[1] < 0.05):

          print(variable,", ",self.__target_variable,'ile ilişkilidir.', '| P-Value:', anova_result[1])

      else:
          print(variable,", ",self.__target_variable,'ile ilişkili değildir.', '| P-Value:', anova_result[1])


  def chi2_contingency(self,categorical_variable=None):
      """Hedef değişken ile kategorik değişken arasında bir ilişki olup olmadığını 
      bulur. Analiz sonucunda elde edilen P-değeri 0.05'den küçük ise ilgili
      değer ile hedef değişken arasında bir ilişki vardır.

      (Not: Hedef değişkenin bir kategorik değişken olduğu kabul edilerek yazılmıştır.)

      Parameters
      ----------
      categorical_variable : str
          Ki-Kare testi için bir kategorik değişken
      """
      
      df_crosstab=pd.crosstab(index=self.df[self.__target_variable], columns=self.df[categorical_variable])
 
      chi_square_result = chi2_contingency(df_crosstab)
      # # Ki kare P-Değeri <0,05 ise, bu, H0'ı reddettiğimiz anlamına gelir
      if (chi_square_result[1]< 0.05):

          print(categorical_variable,", ",self.__target_variable,'ile ilişkilidir.', '| P-Value:', chi_square_result[1])

      else:
          print(categorical_variable,", ",self.__target_variable,'ile ilişkili değildir.', '| P-Value:', chi_square_result[1])
