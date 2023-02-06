# validation_util.py
#
# Ben Roberts
#
# A collection of common functions for analyzing model skill against
# observations
#
# run_stats and plot_fit are copied almost entirely from code I wrote in
# summer 2020 as part of a research assistanceship funded by King County.

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import db

# Dataframe that maps observation database parameter ID to NetCDF variable
# name
_parameter_map = pd.DataFrame({
    # FIXME handle chla: I think just take B1 and multiply by the assumed
    # chla to C ratio
    'param': ('temp','salt','o2','nh4','no23','ph'),
    'output_var': ('temp','salinity','DOXG','NH4','NO3','pH')
}).set_index('param')

class ModelValidator:

    def __init__(self, start, model_output, engine=None, end_date=None):
        self.start_date = pd.Timestamp(start).tz_localize('US/Pacific').tz_convert('GMT')
        if end_date is None:
            self.end_date = self.start_date + pd.to_timedelta(model_output['time'][-1], 'S')
        else:
            self.end_date = pd.Timestamp(end_date).tz_localize('US/Pacific').tz_convert('GMT')
        self.model_output = model_output
        self.engine = db.connect() if engine is None else engine

        # Mapping observation times to model times is most efficiently done
        # by pre-computing the times in between each model output time. So,
        # for example, if the model outputs every six hours with
        # observations at 6am, noon, 6pm, midnight, ... then we want to know
        # the times that fall exactly halfway between those times: 9am, 3pm,
        # 9pm, 3am, ...

        # This will be used as the argument to `np.searchsorted` to identify
        # where in this list a given date/time could be inserted in the
        # series, and then we know the time index to extract from the NetCDF
        # file.
        self.model_t_midpoints = self.start_date + pd.to_timedelta((model_output['time'][:-1] + model_output['time'][1:]) / 2, 'S')

        # Suppress annoying warnings
        self.model_output['siglev'].set_auto_mask(False)
        self.model_output['siglay'].set_auto_mask(False)

    def get_obsdata(self, param, exclude_stations=None):
        global _parameter_map
        if param in _parameter_map:
            self.model_output[_parameter_map.loc[param, 'output_var']].set_auto_mask(False)

        query = "SELECT id, datetime, depth, value, location_id, cast_id "\
            "FROM obsdata.observations obs WHERE parameter_id=%s "\
            "AND datetime BETWEEN %s AND %s "
        query_params = [param, self.start_date, self.end_date]
        if exclude_stations is not None and len(exclude_stations):
            query += "AND location_id NOT IN (" + ','.join(['%s' for s in exclude_stations]) + ") "
            query_params.extend(exclude_stations)
        query += "ORDER BY datetime, depth"

        obsdata = pd.read_sql(query, con=self.engine, params=query_params, index_col='id')
        # This is more efficient than querying it from the DB
        obsdata['parameter_id'] = param
        return obsdata

    def process_cast(self, cast_id, castdata):
        """Create a DataFrame populated with model results matching cast data"""
        global _parameter_map
        # get the parameter ID
        param = _parameter_map.loc[castdata['parameter_id'].iloc[0], 'output_var']
        # Interpolate each cast
        profile = interp1d(castdata['depth'], castdata['value'],
                bounds_error=False)

        # Extract the temperature profiles from each matching node and time in
        # the model output
        t_index = np.searchsorted(self.model_t_midpoints,
                castdata['datetime'].mean())
        n = castdata['node'].iloc[0]
        n_index = n - 1
        model_data = self.model_output[param][t_index,:,n_index]
        model_depths = ((self.model_output['h'][n_index] +
                self.model_output['zeta'][t_index,n_index]) *
            self.model_output['siglay'][:] * -1)

        # pull corresponding observed temperatures from the interpolated cast,
        # removing model results outside the interpolation range
        observed_data = profile(model_depths)

        df = pd.DataFrame({
            'location': castdata['location_id'].iloc[0],
            'node': n_index,
            'datetime': castdata['datetime'].iloc[0],
            'depth': model_depths,
            'sigma': np.arange(1, 11),
            'model': model_data,
            'observed': observed_data
        }).dropna()
        df['cast_id'] = cast_id
        df['t'] = t_index
        return df[['location','node','cast_id','datetime','t','depth','sigma','observed','model']]

    def process_nocast(self, nocasts):
        global _parameter_map
        # get the parameter ID
        param = _parameter_map.loc[nocasts['parameter_id'].iloc[0], 'output_var']

        siglay_ct = self.model_output.dimensions['siglay'].size

        df = pd.DataFrame({
            'location': nocasts['location_id'],
            'node': nocasts['node'],
            'datetime': nocasts['datetime'],
            'depth': nocasts['depth'],
            'observed': nocasts['value'],
            't': np.searchsorted(self.model_t_midpoints, nocasts['datetime'])
        })
        # Initialize new columns we're going to compute
        df['sigma'] = -1
        df['model'] = np.nan

        # We already have the corresponding node and time index for every 
        # observation. Extraction of the data also requires computing
        # corresponding sigma layer (which, thanks to tide, is partially time
        # dependent). Due to NetCDF4 library limitations we can also only
        # extract data with a single dimension as a sequence, so multiple
        # groupby's are required.
        for n,group in df.groupby('node'):
            node_selector = (df['node'] == n)
            for t,group2 in group.groupby('t'):
                model_layer_bounds = (self.model_output['h'][n-1] +
                        self.model_output['zeta'][t,n-1])\
                    * self.model_output['siglev'][:] * -1
                sig_is = np.searchsorted(model_layer_bounds, group2['depth'])
                # The outer group is a copy and not a view of the underlying
                # DataFrame. Both need to be updated here so we can do the
                # groupby on sigma next
                siglays = np.where(sig_is < siglay_ct, sig_is, siglay_ct)
                df.loc[node_selector & (df['t'] == t), 'sigma'] = siglays
                group.loc[group['t'] == t, 'sigma'] = siglays
            for s,group2 in group.groupby('sigma'):
                df.loc[node_selector & (df['sigma'] == s), 'model'] = self.model_output[param][group2['t'],s-1,n-1]

        return df[['location','node','datetime','t','depth','sigma','observed','model']]

def tsplot_zs(location_data, model_output, weight_obs=6.8):
    """Get max and min sigma plot values for a dataset.

    Find the optimal max and min sigma value to use in constructing a
    timeseries plot of data for a specific location.

    Arguments:
    location_data -- a DataFrame containing observations from the same location
    model_output -- a NetCDF Dataset of model results to plot
    weight_obs -- optional parameter to tune the selection algorithm. Larger
        values favor sigma layers with more observations over time at the
        expense of extremity in the water column (very shallow/deep).

    Returns:
        An size 2 list of the minimum and maximum z, one-indexed.
    """
    sigma_counts = location_data.groupby('sigma')['observed'].count()
    if len(sigma_counts) == 0:
        # No observations given, so just pick the most extreme values
        return [1, model_output.dimensions['siglay'].size]
    sigmas = sigma_counts.index.values
    # Define a value function for showing a given layer based on:
    # - how close that layer is to the surface/bottom
    distance_vs = np.cos(2 * np.pi / (sigmas.max() - sigmas.min()) * (sigmas - sigmas.min()))
    # - the observation count
    obs_vs = weight_obs * sigma_counts / sigma_counts.sum()
    values = distance_vs + obs_vs
    # Find the local maxima of this function starting from both ends.
    # argrelextrema returns the indices of values; look up the corresponding
    # sigma value in sigmas.
    values = np.concatenate([[-1], values, [-1]])
    return sigmas[argrelextrema(values, np.greater)[0][[0,-1]] - 1]

# Compute the standard model performance statistics based on a set of
# observations and model predictions
def run_stats(observed, modeled):
    fit, stats = np.polynomial.polynomial.Polynomial.fit(observed, modeled, 1, full=True)
    rmse = mean_squared_error(observed, modeled, squared=False)
    n = len(observed)
    r = np.corrcoef(modeled, fit(observed))[0,1]
    bias = modeled.mean() - observed.mean()
    return (fit, r, rmse, bias, n)

# Build a plot of observed vs modeled data annotated with the fit statistics
def plot_fit(ax, observed, modeled, title, unit=None):
    plot_margin = 0.05
    fit, r, rmse, bias, n = run_stats(observed, modeled)
    xrange = observed.max() - observed.min()
    xmin = observed.min() - plot_margin * xrange
    xmax = observed.max() + plot_margin * xrange
    yrange = modeled.max() - modeled.min()
    ymin = min(modeled.min() - plot_margin * yrange, xmin)
    ymax = max(modeled.max() + plot_margin * yrange, xmax)
    xbound = np.array((xmin, xmax))
    ax.plot(xbound, xbound, '--', color="gray", linewidth=.7)
    marker = "," if n > 10000 else "."
    ax.plot(observed, modeled, marker)
    ax.plot(xbound, fit(xbound))
    ax.grid()
    lbl_append = " ({0})".format(unit) if unit != None else ""
    ax.set(title="%s\n$R$ = %.2f RMSE = %.2f Bias = %.2f N=%d" %
            (title, r, rmse, bias, n),
          ybound=(ymin,ymax), xbound=xbound, xlabel="Observed" + lbl_append,
          ylabel="Model Predicted" + lbl_append)
