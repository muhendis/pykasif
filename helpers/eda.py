
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ProfillingReport:

  """

  Bir veri setinin keşifçi veri analizi için kullanılan bir sınıf

  ...

  Attributes
  ----------
  df : pandas.core.frame.DataFrame
      kullanılacak veri seti
  variables : list
      veri setinin özellikleri


  Methods
  -------
  says(sound=None)
      Prints the animals name and what sound it makes


  """


  def __init__(self,df:pd.core.frame.DataFrame=None,
               continuous_variables:list=None,
               categorical_variables:list=None,
               target_variables:list=None):
    """
    Parameters
    ----------
    df : pd.core.frame.DataFrame
        kullanıcalak veri seti
    continuous_variables : list
        sürekli değişkenler
    categorical_variables : list
        kategorik değişkenler
    target_variables : list
        hedef değişkenler
    """

    self.df = df

    self.variables = df.columns.to_list()

    self.__continuous_variables = self.set_continuous_variables(continuous_variables)

    self.__categorical_variables = self.set_categorical_variables(categorical_variables)

    self.__target_variables= self.set_target_variables(target_variables)

  def __check_it_includes(self,main_list:list=None,sub_list:list=None):
  
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

    result=True

    for sub_i in sub_list :
      if not(sub_i in main_list):
        result = False
        break
    return result

  
  def set_continuous_variables(self,continuous_variables):

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

    assert ( self.__check_it_includes(self.variables, continuous_variables)), "continuous_variables degiskenin değerleri,  veri setinde tanımlanmalıdır."

    return continuous_variables

  def get_continuous_variables(self):

    """Sürekli değişkenleri dönderir

    Returns
    -------
    list
        sürekli olan değişkenlerin listesi
    """

    return self.__continuous_variables
  
  

  def set_categorical_variables(self,categorical_variables):

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

    assert ( self.__check_it_includes(self.variables, categorical_variables)), "categorical_variables degiskenin değerleri,  veri setinde tanımlanmalıdır."
    

    return categorical_variables

  def get_categorical_variables(self):

    """Kategorik değişkenleri dönderir

    Returns
    -------
    list
        kategorik olan değişkenlerin listesi
    """

    return self.__categorical_variables

  
  def set_target_variables(self,target_variables):
  
    """Hedef değişkenleri ayarlamada kullanılır

    Parameters
    ----------
    target_variables : list
         hedef değişkenler

    Assertions
    ------
    AssertionError
        target_variables degiskenin değerleri,  veri setinde tanımlanmamışsa.


    Returns
    -------
    list
        hedeflenen değişkenlerin listesi
    """

    assert ( self.__check_it_includes(self.variables,target_variables)), "target_variables degiskenin değerleri,  veri setinde tanımlanmalıdır."
    

    return target_variables

  def get_target_variables(self):

    """Hedef değişkenleri dönderir

    Returns
    -------
    list
        hedeflenen değişkenlerin listesi
    """

    return self.__target_variables

  def __pie_plot_and_table(self,names:list=None,values:list=None):
    """Dairesel grafik ve tablo oluşturur.

    Parameters
    ----------
    names : list
         dairesel grafik ve tabloda yer alacak değerlerin
         isimleri
    values : list
         dairesel grafik ve tabloda yer alacak değerler
    """
    plt.clf()

    plt.pie(values, labels=names,  autopct='%1.1f%%')

    table=plt.table(cellText=np.asarray(values).reshape(-1,1),
                    rowLabels=names,fontsize=180,
                    bbox = [2, 0.8/len(names), 0.5, 0.15*len(names)]) # bbox = xmin, ymin, xmax, ymax
    
    plt.show()

  def data_types(self):
    """ Veri setindeki veri tiplerini gösteriyor.
    """
    self.__pie_plot_and_table(names=self.df.dtypes.value_counts().index.astype(str).tolist(),
                              values=self.df.dtypes.value_counts().values.tolist())
  
  def missing_cell_count(self):
    """Veri setindeki kayıp hücre miktarını gösteriyor.
    """
    missing_cell_count = self.df.isna().sum().sum()
    filled_cell_count = self.df.size-self.df.isna().sum().sum()

    self.__pie_plot_and_table(names=['Kayıp hücreler', 'Dolu hücreler'],
                              values=[missing_cell_count, filled_cell_count])

  def duplicate_row_count(self):
    """Veri setindeki tekrarlayan satır miktarını gösteriyor.
    """

    duplicated_row_count=self.df.duplicated().sum()
    unduplicated_row_count=self.df.shape[0]-self.df.duplicated().sum()

    self.__pie_plot_and_table(names=['Tekrarlanan satırlar','Tekrarlanmayan satırlar'],
                              values=[duplicated_row_count,unduplicated_row_count])


  def visualize_distribution(self):
    """Veri setindeki değişkenlerinin dağılım grafiğini oluştur
    """
    for var in self.variables:
      fig=self.df.loc[:,var].hist()
      plt.title(var)
      plt.show()
  
  def __create_dispersion_measures(self,feature:str=None):
    
    """Bir özelliğin dağılım ölçülerini olustur.
       Bunlar :
         * count
         * std
         * min
         * 25%
         * 50%
         * 75%
         * max
         * Skewness
         * Kurtosis

    Parameters
    ----------
    feature : str
         veri setindeki özellik

    Returns
    -------
    pandas.core.series.Series
         dağılım ölçülerini içerir
    """
    df_d=self.df.loc[:,feature].describe().T
    df_d=df_d.drop(index=['mean'])								
    df_d["Skewness"] = self.df.loc[:,feature].skew(axis = 0) 
    df_d["Kurtosis"] = self.df.loc[:,feature].kurt(axis = 0) 

    return df_d.T

  def dispersion_measures_of_a_feature(self,feature:str=None):

    """Veri setindeki bir özelliğin dağılım ölçülerinin grafiğini oluştur.

    Parameters
    ----------
    feature : str
         sürekli değişkenlerde yer alan bir değişken

    Assertions
    ------
    AssertionError
        feature degiskenin değerleri,  sürekli değişkenlerde tanımlanmamışsa.

    """

  
    assert (self.__check_it_includes(self.__continuous_variables,[feature])),  "feature degiskenin değeri,  sürekli değişkenlerde tanımlanılmalıdır."
   

    f, (ax_box, ax_hist) = plt.subplots(2, gridspec_kw={"height_ratios": ( .30, .70)})
    

    sns.boxplot(self.df[feature], ax=ax_box)

    ax_box.set(xlabel='')

    sns.histplot(data=self.df, x=feature, ax=ax_hist,kde=True)

    table = self.__create_dispersion_measures(feature)

    plt.table(cellText=table.values.reshape(-1,1),
              rowLabels=table.index,colWidths=[0.4],fontsize=180,
              bbox = [1.2, 0.05, 0.5, 1.5]) # bbox = [1.0, 0.0, 0.35, 1]

    plt.show()

  def central_tendency_measures_of_a_feature(self,feature:str=None): 

    """ Veri setindeki bir özelliğin merkezi dağılımını gösteren 
        değerlerin grafiğini oluştur.
        Bunlar:
          * mod (mode)
          * medyan (median)
          * ortalama (mean)
    Parameters
    ----------
    feature : str
         sürekli değişkenlerde yer alan bir değişken

    Assertions
    ------
    AssertionError
        feature degiskenin değerleri,  sürekli değişkenlerde tanımlanmamışsa.

    """

    assert (self.__check_it_includes(self.__continuous_variables,[feature])), "feature degiskenin değeri,  sürekli değişkenlerde tanımlanılmalıdır."
     

    df_ct=pd.DataFrame()

    df_ct["mode"]=self.df.loc[:,[feature]].mode().iloc[0,:]
    df_ct["median"]=self.df.loc[:,[feature]].median()
    df_ct["mean"]=self.df.loc[:,[feature]].mean()

    df_ct.plot.bar()
    plt.xticks([])

    table=plt.table(cellText=df_ct.T.values, colLabels=df_ct.T.columns,
                    rowLabels=df_ct.T.index,fontsize=180)
    plt.show()


                
  def covariance_matrix(self):
    """Veri setindeki sürekli değişkenlerin kovaryans grafiğini oluşturur
    """

    df_cov = self.df.loc[:,self.__continuous_variables].cov()

    
    mask_cov = np.triu(np.ones_like(df_cov, dtype=bool))

    plt.figure(figsize=(df_cov.shape[0],df_cov.shape[1]))

    sns.heatmap(df_cov,
                mask=mask_cov,annot=True, cmap="inferno", 
                cbar_kws={"shrink": .5})
    plt.show()
  

  def correlation_analysis(self):
    """Veri setindeki  sürekli olan değişkenlerin korelasyon grafiğini oluştur
    """

    df_corr = self.df.loc[:,self.__continuous_variables].corr()
    
    mask_corr = np.triu(np.ones_like(df_corr, dtype=bool))

    
    plt.figure(figsize=(df_corr.shape[0],df_corr.shape[1]))

    sns.heatmap(df_corr, fmt=".2%", 
                  mask=mask_corr,annot=True, cmap="Blues", 
                  cbar_kws={"shrink": .5})
    plt.show()

  def principal_component_analysis_2d(self,feature_color:np.array=None):
    """ Veri setini iki boyutta incelemeyi sağlar.

    Parameters
    ----------
    feature_color : np.array
         veri setindeki her bir örneği renklendirmek için kullan br dizi

    """

    from sklearn.decomposition import PCA

    pca = PCA(2)  
    data = self.df.drop(columns=self.__target_variables)
    projected = pca.fit_transform(data)

    plt.scatter(projected[:, 0], projected[:, 1],c=feature_color)
    ratios =pca.explained_variance_ratio_
    ratio_for_2d=ratios[0]+ratios[1]
    plt.title(f"Açıklanan varyans oranı : {ratio_for_2d}")
    plt.xlabel('Bileşen 1')
    plt.ylabel('Bileşen 2')
    plt.colorbar();
    plt.show()

  def jointplot(self,x_axis:str,y_axis:str):
    """Veri setinde bulunan iki sürekli değişkenin ilişkisini inceler.
    Parameters
    ----------
    x_axis : str
         x ekseni için sürekli değişkenlerde yer alan bir değişken
    y_axis : str
         y ekseni için sürekli değişkenlerde yer alan bir değişken

    Assertions
    ------
    AssertionError
        x_axis ve y_axis  veri setinde  tanımlanmamışsa.

    """

    assert (x_axis in self.variables and y_axis in self.variables), "x_axis ve y_axis  veri setinde tanımlanmalıdır."

    sns.jointplot(x=x_axis, y=y_axis, data=self.df,kind="reg")
