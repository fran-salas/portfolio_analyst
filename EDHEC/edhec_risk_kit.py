import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm

# ========================
# DATA LOADING FUNCTIONS
# ========================

def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ffme_returns():
    """
    Load the Fama-French Dataset for returns of Top and Bottom Market Cap Deciles
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
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

# ========================
# DRAWDOWN & RISK MEASURES
# ========================

def drawdown(return_series: pd.Series):
    """
    Takes a time series of returns and calculates drawdowns
    Returns DataFrame with: wealth index, previous peaks, percent drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "previous_peaks": previous_peaks,
        "drawdowns": drawdowns
    })

def semideviation(r):
    """
    Returns the negative semideviation of r
    r must be a Series or DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def semideviation3(r):
    """
    Returns the semideviation (returns below the mean)
    """
    excess = r - r.mean()
    excess_negative = excess[excess < 0]
    excess_negative_square = excess_negative**2
    n_negative = (excess < 0).sum()
    return (excess_negative_square.sum()/n_negative)**0.5

# ========================
# VALUE AT RISK & EXTREME RISK
# ========================

def var_historic(r, level=5):
    """
    Historic Value at Risk (VaR)
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def var_gaussian(r, level=5, modified=False):
    """
    Parametric Gaussian VaR (with Cornish-Fisher adjustment)
    """
    z = norm.ppf(level/100)
    if modified:
        # Cornish-Fisher adjustment
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z**2 - 1)*s/6 +
             (z**3 - 3*z)*(k-3)/24 -
             (2*z**3 - 5*z)*(s**2)/36
        )
    return abs(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Conditional Value at Risk (Expected Shortfall)
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

# ========================
# DISTRIBUTIONAL STATISTICS
# ========================

def skewness(r):
    """
    Compute skewness for Series/DataFrame
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / sigma_r**3

def kurtosis(r):
    """
    Compute kurtosis for Series/DataFrame
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / sigma_r**4

def is_normal(r, level=0.01):
    """
    Applies the Jarque Bera test for normality at default 1% significance
    Returns True if normal, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


# ========================
# OPTIMAL PORTFOLIO and EFFICIENT FRONTIER
# ========================

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5


def plot_ef2(n_points, er, cov, style = ".-"):
    """
    Plots the 2-asset efficient frontier
    """

    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("Plot_ef2 can only plot 2-asset frontiers")
    
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns":rets,
        "Volatility":vols
    })

    return ef.plot.line(x="Volatility", y="Returns", style=style)


from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
  """
  target_ret -> w
  """
  n = er.shape[0] #number of assets. It is number of rows of expected returns
  init_guess = np.repeat(1/n, n)
  bounds = ((0.0, 1.0),)*n #n copies of A tuple with no less than zero and 1.0 as max
  return_is_target = { #to provide constraints to the weights
    'type': 'eq', #we want it to be equal
    'args': (er,),
    'fun': lambda weights, er: target_return - portfolio_return(weights, er) #function to explain whether the constraint is met
    # Same as saying 
    # def target_is_met(w,er):
    # return target_return - erk.portfolio_return(w,er)
  }
  weights_sum_to_1 = {
    'type': 'eq',
    'fun': lambda weights: np.sum(weights) - 1
  }

  results = minimize(portfolio_vol, init_guess,
  args = (cov,), method = "SLSQP", options = {'disp':False},
  constraints =(return_is_target, weights_sum_to_1),
  bounds = bounds)

  return results.x


def optimal_weights(n_points, er, cov):
  """
  -> List of weights to run the optimizer on to minimize the vol
  """  
  target_returns = np.linspace(er.min(), er.max(), n_points)
  weights = [minimize_vol(target_return,er,cov) for target_return in target_returns]
  return weights

def plot_ef(n_points, er, cov, style = ".-"):
    """
    Plots the n-asset efficient frontier
    """

    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns":rets,
        "Volatility":vols
    })

    return ef.plot.line(x="Volatility", y="Returns", style=style)