# validation_util.py
#
# Ben Roberts
#
# A collection of common functions for analyzing model skill against
# observations
#
# run_stats and plot_fit are copied almost entirely from code I wrote in
# summer 2020 as part of a research assistanceship funded by King County.

# Temporary workaround for recent geopandas
import os
os.environ['USE_PYGEOS'] = '0'

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
from sklearn.metrics import root_mean_squared_error
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from fvcom.grid import FvcomGrid
from fvcom.depth import DepthCoordinate
import db

# Dataframe that maps observation database parameter ID to NetCDF variable
# name
_parameter_map = pd.DataFrame({
    # FIXME handle chla: I think just take B1 and multiply by the assumed
    # chla to C ratio
    'param': ('temp','salt','o2','nh4','no23','ph','chla'),
    'output_var': ('temp','salinity','DOXG','NH4','NO3','pH',('B1','B2','B3')),
    # ICM State: temp and salinity are theoretically present but they're all 0's
    'icmstate_var': (0, 1, 26, 12, 13, -1, -1),
    'ecol_var': ('Var_18','Var_19','Var_10','Var_14','Var_15','Var_32',('Var_12','Var_13'))
}).set_index('param')

_ratios = {
    'chla': (37,50,1)
}

class Validator:
    """Abstract class for validation. Not meant to be instatiated directly"""

    def __init__(self, grid, depthcoord, start_date, end_date, engine=None):
        self.grid = grid
        self.depthcoord = depthcoord
        self.engine = db.connect() if engine is None else engine
        self.start_date = start_date.tz_localize('US/Pacific').tz_convert('GMT')
        self.end_date = end_date.tz_localize('US/Pacific').tz_convert('GMT')
        self.obsdfs = []

    def get_obsdata(self, params, exclude_stations=None, cv=None):
        obsdatas = []
        for param in params:
            query_sel = "SELECT id, datetime, depth, value, location_id, cast_id "
            query_from = "FROM obsdata.observations obs "
            query_from += "INNER JOIN obsdata.stations l ON obs.location_id = l.name "
            query_where = "WHERE obs.parameter_id=%s AND obs.datetime BETWEEN %s AND %s "
            query_params = [param, self.start_date, self.end_date]
            # Construct the polygon in PostGIS
            shp_poly = self.grid.to_polygon() if cv is None else cv.to_polygon()
            poly = "SRID=32610;POLYGON((" + ', '.join([f'{x} {y}' for x, y in shp_poly.exterior.coords]) + "))"
            query_where += f"AND ST_Within(l.geom, '{poly}') "
            # Exclude any holes in the CV
            if len(shp_poly.interiors) > 0:
                for ring in shp_poly.interiors:
                    poly = "SRID=32610;POLYGON((" + ', '.join([f'{x} {y}' for x, y in ring.coords]) + "))"
                    query_where += f"AND NOT ST_Within(l.geom, '{poly}') "

            if exclude_stations is not None and len(exclude_stations):
                query_where += "AND location_id NOT IN (" + ','.join(['%s' for s in exclude_stations]) + ") "
                query_params.extend(exclude_stations)
            query = query_sel + query_from + query_where + "ORDER BY datetime, depth"

            obsdata = pd.read_sql(query, con=self.engine, params=tuple(query_params), index_col='id')
            obsdata['datetime'] = pd.to_datetime(obsdata['datetime'],utc=True)
            # This is more efficient than querying it from the DB
            obsdata['parameter_id'] = param

            obsdatas.append(obsdata)

        all_station_ids = pd.concat([df['location_id'] for df in obsdatas]).drop_duplicates()
        stations = gpd.read_postgis(f"SELECT * FROM obsdata.stations", con=self.engine)
        stations = stations.loc[stations['name'].isin(all_station_ids)]

        model_points = self.grid.nodes_gdf()
        if 'depth' in model_points:
            del model_points['depth']
        station_nodes = gpd.tools.sjoin_nearest(stations, model_points).set_index('name').rename(columns={'index_right': 'node'})
        for i,obsdata in enumerate(obsdatas):
            obsdatas[i] = obsdata.merge(station_nodes, left_on='location_id', right_index=True)

        return (obsdatas, station_nodes)

    def _get_t_indices(self, datetimes):
        pass

    def get_model_match(self, param, t_slice, depth_slice, n_slice):
        pass

    def get_times(self, slc=slice(None)):
        pass

    def _get_ssh(self, t_slice, n_slice):
        pass

    def process_cast(self, cast_id, castdata):
        # Interpolate each cast
        profile = interp1d(castdata['depth'], castdata['value'],
                bounds_error=False)

        t_index = self._get_t_indices(castdata['datetime'].mean())
        n_index = castdata['node'].iloc[0] - 1
        match_data = self.get_model_match(castdata['parameter_id'].iloc[0],
                t_index, slice(None), n_index)
        match_depths = ((self.grid.ncoord[2,n_index] + self._get_ssh(t_index,n_index)) *
                self.depthcoord.zz[:-1] * -1)
        assert len(match_data) == len(match_depths), f"Invalid shape {match_data.shape} for match_data, should be {len(match_depths)}"

        # pull corresponding observed temperatures from the interpolated cast,
        # removing model results outside the interpolation range
        observed_data = profile(match_depths)

        df = pd.DataFrame({
            'location': castdata['location_id'].iloc[0],
            'node': castdata['node'].iloc[0],
            'datetime': castdata['datetime'].iloc[0],
            'depth': match_depths,
            'sigma': np.arange(1, self.depthcoord.kb),
            'model': match_data,
            'observed': observed_data
        }).dropna()
        df['cast_id'] = cast_id
        df['t'] = t_index
        return df[['location','node','cast_id','datetime','t','depth','sigma','observed','model']]

    def process_nocast(self, nocasts):
        if len(nocasts) == 0:
            return pd.DataFrame([], columns=['location','node','datetime','depth','observed','t','sigma','model'])

        param_id = nocasts['parameter_id'].iloc[0]

        siglay_ct = self.depthcoord.kb - 1

        df = pd.DataFrame({
            'location': nocasts['location_id'],
            'node': nocasts['node'],
            'datetime': nocasts['datetime'],
            'depth': nocasts['depth'],
            'observed': nocasts['value'],
            't': self._get_t_indices(nocasts['datetime'])
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
                model_layer_bounds = ((self.grid.ncoord[2,n-1] + self._get_ssh(t,n-1))
                    * self.depthcoord.z[:-1] * -1)
                sig_is = np.searchsorted(model_layer_bounds, group2['depth'])
                # The outer group is a copy and not a view of the underlying
                # DataFrame. Both need to be updated here so we can do the
                # groupby on sigma next
                siglays = np.where(sig_is < siglay_ct, sig_is, siglay_ct)
                df.loc[node_selector & (df['t'] == t), 'sigma'] = siglays
                group.loc[group['t'] == t, 'sigma'] = siglays
            for s,group2 in group.groupby('sigma'):
                df.loc[node_selector & (df['sigma'] == s), 'model'] = self.get_model_match(param_id, group2['t'], s-1, n-1)

        return df[['location','node','datetime','t','depth','sigma','observed','model']]

class ModelValidator(Validator):

    def __init__(self, start, model_output, engine=None, end_date=None):
        """Construct an instance of the validator.

        Parameters:
        - start: a value that can be passed to pandas.Timestamp that
          represents the zero time of the model
        - model_output: an optional NetCDF Dataset of the model results
        - engine: The SQL query engine for the observation database
        - end_date: An optional end date to query data for. If not specified,
          model_output must be given, and end date will be read as the final
          output time

        FIXME allow specifying a start date for the query that's later than model start
        """
        if not isinstance(start, pd.Timestamp):
            start_date = pd.Timestamp(start)
        else:
            start_date = start
        if end_date is None:
            end_date = start_date + pd.to_timedelta(model_output['time'][-1], 's')
        else:
            if not isinstance(end_date, pd.Timestamp):
                end_date = pd.Timestamp(end_date)
        model_output['zeta'].set_auto_mask(False)
        self.model_output = model_output

        grid = FvcomGrid.from_output(model_output)
        depthcoord = DepthCoordinate.from_output(model_output,
                grid=grid)
        super().__init__(grid, depthcoord, start_date, end_date)

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
        self.model_t_midpoints = self.start_date + pd.to_timedelta((model_output['time'][:-1] + model_output['time'][1:]) / 2, 's')

    def get_model_match(self, param_id, t_slice, depth_slice, n_slice):
        global _parameter_map
        # get the output parameter name
        param = _parameter_map.loc[param_id, 'output_var']
        if isinstance(param, tuple):
            # Drop any params that aren't in output
            included = set(param) & set(self.model_output.variables.keys())
            return np.sum([self.model_output[p][t_slice, depth_slice, n_slice] * r for p,r in zip(included, _ratios[param_id] if param_id in _ratios else np.zeros(len(param_id)) + 1)], axis=0)
        else:
            v = self.model_output[param][t_slice, depth_slice, n_slice]
            if param in _ratios:
                v *= _ratios[param]
            return v

    def get_times(self, slc=slice(None)):
        return self.start_date + pd.to_timedelta(self.model_output['time'][slc], 'S')

    def _get_t_indices(self, datetimes):
        return np.searchsorted(self.model_t_midpoints, datetimes)

    def _get_ssh(self, t_slice, n_slice):
        return self.model_output['zeta'][t_slice, n_slice]

class AbsStateValidator(Validator):
    def __init__(self, state, ts, span=5, engine=None):
        self.state = state
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        span_delta = pd.Timedelta(span, 'D')
        super().__init__(state.grid, state.dcoord, ts - span_delta,
                ts + span_delta, engine=engine)
        self.ts = self.start_date + (self.end_date - self.start_date) / 2

    def get_times(self, slc=slice(None)):
        return np.array([self.start_date])

    def _get_t_indices(self, datetimes):
        return np.zeros_like(datetimes)

    def get_obsdata(self, params, **kwargs):
        ret = super().get_obsdata(params, **kwargs)
        # Add a "tdiff" column with a precomputed time difference from the
        # state's time
        for r in ret[0]:
            r['tdiff'] = np.abs(r['datetime'] - self.ts)
        return ret

    def filter_best(self, obsdata):
        """Pick the observations closest to the state timestamp"""
        filtered_casts = []
        filtered_idxs = []
        for n,group in obsdata.groupby('node'):
            best_cast = None
            for cid,cgroup in group.groupby('cast_id'):
                if pd.isna(cid):
                    continue
                cast_time = group['datetime'].mean()
                if best_cast is None or np.abs(cast_time - self.ts) < np.abs(best_cast_time - self.ts):
                    best_cast = cid
                    best_cast_time = cast_time
            if best_cast is None:
                # No casts found. Look for non-cast data
                # Start by identifying the siglayer of every observation
                layer_bounds = ((self.grid.ncoord[2,n-1] + self._get_ssh(0,n-1))
                    * self.depthcoord.zz[:-1] * -1)
                sig_is = pd.Series(np.searchsorted(layer_bounds, group['depth']),
                        index=group.index)
                filtered_idxs.append(group.groupby(sig_is)['tdiff'].idxmin())
            else:
                filtered_casts.append(best_cast)
        if len(filtered_idxs) > 0:
            filtered_idxs = pd.concat(filtered_idxs, ignore_index=True)
        return obsdata.loc[np.isin(obsdata['cast_id'], filtered_casts) | np.isin(obsdata.index, filtered_idxs)]

class ICMStateValidator(AbsStateValidator):

    def get_model_match(self, param_id, t_slice, depth_slice, n_slice):
        # TODO handle pH separately as it must be computed from DIC, TALK and pCO2
        global _parameter_map
        # get the C2 constituent index
        param = _parameter_map.loc[param_id, 'icmstate_var']
        return self.state.c2[n_slice, depth_slice, param]

    def _get_ssh(self, t_slice, n_slice):
        # TODO
        return np.zeros(self.state.grid.m)[n_slice]

class FvcomStateValidator(AbsStateValidator):

    def get_model_match(self, param_id, t_slice, depth_slice, n_slice):
        assert param_id in ['temp','salt']
        arr = self.state.t1 if param_id == 'temp' else self.state.s1
        # Need to chop off the last depth index before applying the slice
        return arr.T[:,:-1][n_slice, depth_slice]

    def _get_ssh(self, t_slice, n_slice):
        # TODO
        return np.zeros(self.state.grid.m)[n_slice]

class EcolModelValidator(Validator):
    """For Ecology's downloadable model outputs in NetCDF format"""
    def __init__(self, start, model_output, grid, end_date=None):
        if not isinstance(start, pd.Timestamp):
            start_date = pd.Timestamp(start)
        else:
            start_date = start
        if end_date is None:
            end_date = start_date + pd.to_timedelta(365, 'D')
        else:
            if not isinstance(end_date, pd.Timestamp):
                end_date = pd.Timestamp(end_date)
        self.model_output = model_output

        depthcoord = DepthCoordinate.from_asym_sigma(11, grid, 1.5)
        super().__init__(grid, depthcoord, start_date, end_date)
        times = model_output.dimensions['Time'].size
        self.model_t_midpoints = self.start_date + pd.to_timedelta(np.linspace(0, 365, times)[:-1] + 1 / (2 * times), 'D')

    def _get_ijk(self, depth_slice, n_slice):
        if type(depth_slice) == slice:
            indices = depth_slice.indices(self.depthcoord.kb-1)
            ijk = slice(n_slice * (self.depthcoord.kb-1) + indices[0],
                    n_slice * (self.depthcoord.kb-1) + indices[1])
        else:
            ijk = n_slice * (self.depthcoord.kb-1) + depth_slice
        return ijk

    def get_model_match(self, param_id, t_slice, depth_slice, n_slice):
        global _parameter_map
        # get the output parameter name
        param = _parameter_map.loc[param_id, 'ecol_var']
        ijk = self._get_ijk(depth_slice, n_slice)
        if isinstance(param, tuple):
            return np.sum([self.model_output[p][t_slice, ijk] * r for p,r in zip(param, _ratios[param_id] if param_id in _ratios else np.zeros(len(param_id)) + 1)], axis=0)
        else:
            v = self.model_output[param][t_slice, ijk]
            if param in _ratios:
                v *= _ratios[param]
            return v

    def get_times(self, slc=slice(None)):
        ts = self.model_output.dimensions['Time'].size
        return self.start_date + pd.to_timedelta(np.linspace(0, 365, ts), 'D')

    def _get_t_indices(self, datetimes):
        return np.searchsorted(self.model_t_midpoints, datetimes)

    def _get_ssh(self, t_slice, n_slice):
        return self.model_output['Var_7'][t_slice, n_slice * (self.depthcoord.kb-1)].data

def tsplot_zs(validator, location_data, weight_obs=6.8):
    """Get max and min sigma plot values for a dataset.

    Find the optimal max and min sigma value to use in constructing a
    timeseries plot of data for a specific location.

    Arguments:
    validator -- a Validator object for looking up model data
    location_data -- a DataFrame containing observations from the same location
    weight_obs -- optional parameter to tune the selection algorithm. Larger
        values favor sigma layers with more observations over time at the
        expense of extremity in the water column (very shallow/deep).

    Returns:
        An size 2 list of the minimum and maximum z, one-indexed.
    """
    sigma_counts = location_data.groupby('sigma')['observed'].count()
    if len(sigma_counts) == 0:
        # No observations given, so just pick the most extreme values
        return [1, validator.depthcoord.kb - 1]
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

# Taken from hydroeval, as this isn't available on conda-forge
def nse(simulations, evaluation):
    """Nash-Sutcliffe Efficiency (NSE) as per `Nash and Sutcliffe, 1970
    <https://doi.org/10.1016/0022-1694(70)90255-6>`_.

    :Calculation Details:
        .. math::
           E_{\\text{NSE}} = 1 - \\frac{\\sum_{i=1}^{N}[e_{i}-s_{i}]^2}
           {\\sum_{i=1}^{N}[e_{i}-\\mu(e)]^2}

        where *N* is the length of the *simulations* and *evaluation*
        periods, *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, and *μ* is the arithmetic mean.

    """
    nse_ = 1 - (
            np.sum((evaluation - simulations) ** 2, axis=0, dtype=np.float64)
            / np.sum((evaluation - np.mean(evaluation)) ** 2, dtype=np.float64)
    )

    return nse_

# Compute the standard model performance statistics based on a set of
# observations and model predictions
def run_stats(observed, modeled):
    rmse = root_mean_squared_error(observed, modeled)
    n = len(observed)
    try:
        fit, stats = np.polynomial.polynomial.Polynomial.fit(observed, modeled, 1, full=True)
        r = np.corrcoef(modeled, fit(observed))[0,1]
    except np.linalg.LinAlgError:
        fit = None
        r = np.nan
    ns = nse(modeled.to_numpy(), observed.to_numpy())
    bias = modeled.mean() - observed.mean()
    return (fit, r, rmse, ns, bias, n)

# Build a plot of observed vs modeled data annotated with the fit statistics
def plot_fit(ax, observed, modeled, title, unit=None):
    plot_margin = 0.05
    fit, r, rmse, ns, bias, n = run_stats(observed, modeled)
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
    ax.set(title=f"{title}\n$R$={r:.2f} RMSE={rmse:.2f} NSE={ns:.2f} Bias={bias:.2f} N={n:d}",
          ybound=(ymin,ymax), xbound=xbound, xlabel="Observed" + lbl_append,
          ylabel="Model Predicted" + lbl_append)
