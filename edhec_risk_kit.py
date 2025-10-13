import pandas as pd
import scipy.stats
import numpy as np

def drawdown(return_series: pd.Series): #inputs. :pd.Series expresses the type
  """
  Takes a time series of returns
  Computes and returns a DataFrame that contains:
  wealth index
  previous peaks
  percent drawdowns
  """

  wealth_index = 1000*(1+return_series).cumprod()
  previous_peaks = wealth_index.cummax()
  drawdowns = (wealth_index-previous_peaks)/previous_peaks

  return pd.DataFrame({
    "Wealth": wealth_index,
    "previous_peaks": previous_peaks,
    "drawdowns": drawdowns
  })


def get_ffme_returns():
  """
  Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by
  MarketCap
  """
  me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",header=0, index_col=0, na_values=-99.99)
  rets = me_m[['Lo 10', 'Hi 10']]
  rets.columns = ['SmallCap', 'LargeCap']
  rets = rets/100
  rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
  return rets


def get_hfi_returns():
  """
  Load and format the EDHEC Hedge Fund Index Returns
  """
  hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                    header=0, index_col=0, parse_dates=True)
  hfi = hfi / 100
  hfi.index = hfi.index.to_period('M')
  return hfi


def semideviation(r):
  """
  Returns the semideviation aka negative semideviation of r
  r must be a Series or a Dataframe
  """
  is_negative = r < 0 #boolean mask
  return r[is_negative].std(ddof=0)


def var_historic(r, level = 5):
  """
  VaR historic
  """
  if isinstance(r, pd.DataFrame):
    return r.aggregate(var_historic, level = level)
  elif isinstance(r, pd.Series):
    return -np.percentile(r,level) #it is semi deviation, so all are understood to be negative. We present them as positive
  else:
    raise TypeError("Expected r to be Series or DataFrame")

def skewness(r):
  """
  Alternative to scipy.stats.skew()
  Computes the skewness of the supplied Series or DataFrame
  Returns a float or a Series
  """
  demeaned_r = r - r.mean()
  # use the population standard deviation, so set ddof=0
  sigma_r = r.std(ddof=0)
  exp = (demeaned_r**3).mean()
  return exp / sigma_r**3

def kurtosis(r):
  """
  Alternative to scipy.stats.kurtosis()
  Computes the kurtosis of the supplied Series or DataFrame
  Returns a float or a Series
  """
  demeaned_r = r - r.mean()
  # use the population standard deviation, so set ddof=0
  sigma_r = r.std(ddof=0)
  exp = (demeaned_r**4).mean()
  return exp / sigma_r**4



def is_normal(r, level=0.01):
  """
  Applies the Jarque Bera test to dermine if a series is normal or not
  Test is applied at the 1% level by default
  Returns true if the hypothesis of normality is accepted, false otherwise
  """

  statistic, p_value = scipy.stats.jarque_bera(r)

  return p_value>level