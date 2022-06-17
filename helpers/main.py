################################################################################

class DataCleaning:

  def __init__(self,df):

    self.df = df

  def remove_duplicate_observation(self):

    pass

  def remove_variable_with_low_variance(self):

    """
    Tüm değerleri aynı olan kategorik değişken
    """

    pass

  def remove_variable_with_high_variance(self):

    """
    Tüm değerleri farklı olan kategorik değişken
    """

    pass
  
  def outlier_treatment(self):

    """
    Cut-Off or Delete
    Natural Log
    Binning
    Assign Weights
    Mean/Mode/Median Imputation
    Build Predictive Model
    Treat them separately
    """

    pass

  def missing_values_treatment(self):

    """
    Delete
    Mean/Mode/Median Imputation
    Prediction Model
    KNN Imputation
    """

    pass


  def variable_transformation(self):

    """
    Logarithm
    Square / Cube root
    Binning / Discretization
    Dummies
    Factorization
    Other Data Type
    """

    pass

################################################################################


class FeatureCreation:

  def __init__(self,df):

    self.df = df

  def indicator_features(self):

    """
    Threshold (ex. below certain price = poor)
    Combination of features (ex. premium house if 2B,2Bth)
    Special Events (ex. christmas day or blackfriday)
    Event Type (ex. paid vs unpaid based on traffic source)
    """

    pass
  
  def representation_features(self):

    """
    Domain and Time Extractions (ex.purchase_day_of_week)
    Numeric to Categorical (ex. years in school to “elementary”)
    Grouping sparse classes (ex. sold, all other are “other”)
    """

    pass

  def interaction_features(self):

    """
    Sum of Features
    Difference of Features
    Product of Features
    Quotient of Features
    Unique Formula
    """
    pass

  def conjunctive_features(self):
    """
    Markov Blanket
    Linear Predictor
    """
    pass

