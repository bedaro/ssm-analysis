#!/usr/bin/env python3
"""
Plot bulk fluxes as a time series.
"""
from argparse import ArgumentParser
import os
from multiprocessing import Pool
import Lfun
import zfun

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pickle
from datetime import datetime, timedelta
# 2019.11.14 make monthly averages
import pandas as pd

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

def dir_path(string):
    if not os.path.isdir(string):
        raise NotADirectoryError(string)
    return string

parser = ArgumentParser(description='Plot bulk fluxes as a time series')
parser.add_argument('--loo', type=dir_path, help='LiveOcean Output dir')
parser.add_argument('-s', '--section', nargs='*',
        help='Section(s) to plot, or "all"')
parser.add_argument('item', nargs='?', help='The TEF fluxes to plot')
args = parser.parse_args()

if args.loo is None:
    # choose input and organize output
    Ldir = Lfun.Lstart()
    indir0 = Ldir['LOo'] + 'tef/'
else:
    indir0 = os.path.join(args.loo, 'tef/')
if args.item is None:
    # choose the tef extraction to process
    item = Lfun.choose_item(indir0)
else:
    item = args.item

indir0 = indir0 + item + '/'
indir = indir0 + 'bulk/'
sect_list_raw = os.listdir(indir)
sect_list_raw.sort()
sect_list = [item for item in sect_list_raw if ('.p' in item)]
if args.section is None:
    print(20*'=' + ' Processed Sections ' + 20*'=')
    print(*sect_list, sep=", ")
    print(61*'=')
# select which sections to process
    my_choice = input('-- Input section to plot (e.g. sog5, or Return to plot all): ')
    if len(my_choice)==0:
        # full list
        my_choice = sect_list
    else: # single item
        my_choice = [my_choice]
elif len(args.section) == 1 and args.section[0] == 'all':
    # full list
    my_choice = sect_list
else: # one or more items
    my_choice = args.section
if my_choice != sect_list:
    found_choices = []
    for c in my_choice:
        if (c + '.p') in sect_list:
            found_choices.append(c + '.p')
        else:
            raise FileNotFoundError(os.path.join(indir, c + '.p'))
    sect_list = found_choices

outdir = indir0 + 'bulk_plots/'
Lfun.make_dir(outdir)
save_fig = (len(sect_list) > 1)

sect_list = [item for item in sect_list]

def process_section(snp):
    
    sn = snp.replace('.p','')

    bulk = pickle.load(open(indir + snp, 'rb'))
    QQ = bulk['QQ']
    SS = bulk['SS']
    ot = bulk['ot']
    qnet = bulk['qnet_lp'] # net volume transport
    fnet = bulk['fnet_lp'] # net tidal energy flux
    ssh = bulk['ssh_lp'] # average SSH across the section, low-passed
    NT, NS = SS.shape

    # make vector and array times in days from start of the year
    dt = []
    for tt in ot:
        dt.append(Lfun.modtime_to_datetime(tt))
    td = []
    extract_file = indir0 + '/extractions/' + sn + '.nc'
    extract_nc = Dataset(extract_file)

    year = int(extract_nc.date_string0.split(".")[0])
    for tt in dt:
        #ttt = tt- datetime(dt[0].year,1,1)
        ttt = tt - datetime(year,1,1) # hardwire for 2016.12.15 start
        td.append(ttt.days + ttt.seconds/86400)
    td = np.array(td) # time in days from start of the year
    Time = td.reshape((NT,1)) * np.ones((1,NS)) # matrix version

    dir_str = "Up-Estuary"
    # some information about direction
    #x0, x1, y0, y1, landward = sect_df.loc[sn,:]    
    #if (x0==x1) and (y0!=y1):
    #    sdir = 'NS'
    #    if landward == 1:
    #        dir_str = 'Eastward'
    #    elif landward == -1:
    #        dir_str = 'Westward'
    #    a = [y0, y1]; a.sort()
    #    y0 = a[0]; y1 = a[1]
    #elif (x0!=x1) and (y0==y1):
    #    sdir = 'EW'
    #    if landward == 1:
    #        dir_str = 'Northward'
    #    elif landward == -1:
    #        dir_str = 'Southward'
    #    a = [x0, x1]; a.sort()
    #    x0 = a[0]; x1 = a[1]
        
    # separate out positive and negative transports
    QQp = QQ.copy()
    QQp[QQ<=0] = np.nan
    QQm = QQ.copy()
    QQm[QQ>=0] = np.nan
    
    # form two-layer versions of Q and S
    Qin = np.nansum(QQp, axis=1)
    QSin = np.nansum(QQp*SS, axis=1)
    Sin = QSin/Qin
    Qout = np.nansum(QQm, axis=1)
    QSout = np.nansum(QQm*SS, axis=1)
    Sout = QSout/Qout
    # and find net transport to compare with qnet (should be identical)
    Qnet = np.nansum(QQ, axis=1)
    # RESULT: it is identical
    
    td_list = []
    for t in td:
        td_list.append(datetime(year,1,1,0,0,0) + timedelta(days=t))
    tef_df = pd.DataFrame(index=td_list, columns=['Qin','Qout','Sin','Sout'])
    tef_df.loc[:,'Qin']=Qin
    tef_df.loc[:,'Qout']=Qout
    tef_df.loc[:,'Sin']=Sin
    tef_df.loc[:,'Sout']=Sout
    tef_df.to_csv(outdir + sn + '.csv')
    tef_mean_df = tef_df.resample('1M').mean()
    # the above puts timestamps at the end of the month
    # so here we set it to the middle of each month becasue it is more
    # consistent with the averaging
    tef_mean_df.index -= timedelta(days=15)
    tef_mean_df.loc[:,'yd'] = tef_mean_df.index.dayofyear

    # PLOTTING
    # See https://stackoverflow.com/a/65910539/413862 for how num and
    # clear prevent memory leaks
    fig = plt.figure(num=1 if save_fig else None, figsize=(21,9))
    
    alpha = .5

    # Salinity vs. Time (size and color by Transport)
    ax = fig.add_subplot(2,3,1)
    Qscale = np.nanmean(np.abs(QQ))
    qf = 25
    ax.scatter(Time, SS, s=qf*np.abs(QQp/Qscale), c='r', alpha=alpha)
    ax.scatter(Time, SS, s=qf*np.abs(QQm/Qscale), c='b', alpha=alpha)
    # add two-layer versions
    if False:
        ax.plot(td, Sin, '-k', td, Sout, '--k')
    else:
        tef_mean_df.plot(x='yd', y = 'Sin', style='-ok', ax=ax, legend=False)
        tef_mean_df.plot(x='yd', y = 'Sout', style='--ok', ax=ax, legend=False)
    ax.text(0.05, 0.1, 'Positive is ' + dir_str, transform=ax.transAxes, fontweight='bold')
    ax.set_xlim(0,366)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_ylabel('Salinity')
    # legend
    ax.scatter(.95, .2, s=qf, c='r', transform=ax.transAxes, alpha=alpha)
    ax.scatter(.95, .1, s=qf, c='b', transform=ax.transAxes, alpha=alpha)
    ax.text(.94, .2, 'Positive Q %d (m3/s)' % int(Qscale), color='r', fontweight='bold',
        horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.text(.94, .1, 'Negative Q %d (m3/s)' % int(Qscale), color='b', fontweight='bold',
        horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.set_title(indir0.split('/')[-2])
    
    # # Tidal energy flux vs. Time as second y-axis
    # ax = ax.twinx()
    # ax.plot(td, fnet/1e9, '-g', linewidth=2)
    # ax.set_ylabel('Energy Flux (GW)', color='g', alpha=alpha)
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(0,366)
    
    # Tranport vs. Time
    ax = fig.add_subplot(2,3,4)
    ax.scatter(Time, QQp/1e3, s=qf*np.abs(QQp/Qscale), c='r', alpha=alpha)
    ax.scatter(Time, -QQm/1e3, s=qf*np.abs(QQm/Qscale), c='b', alpha=alpha)
    # add two-layer versions
    if False:
        ax.plot(td, Qin/1e3, '-k', td, -Qout/1e3, '--k')
    else:
        this_yd = tef_mean_df.loc[:,'yd'].values
        this_qin = tef_mean_df.loc[:,'Qin'].values/1e3
        this_qout = -tef_mean_df.loc[:,'Qout'].values/1e3
        # tef_mean_df.plot(x='yd', y = 'Qin', style='-ok', ax=ax, legend=False)
        # tef_mean_df.plot(x='yd', y = 'Qout', style='--ok', ax=ax, legend=False)
        ax.plot(this_yd, this_qin, '-ok')
        ax.plot(this_yd, this_qout, '--ok')
    ax.set_xlim(0,366)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.set_xlabel('Days from 1/1/' + str(year))
    ax.set_ylabel('|Q| 1000 m3/s')
    
    # Tidal energy flux vs. Time as second y-axis
    ax = fig.add_subplot(3,3,2)
    ax.plot(td, fnet/1e9, '-g', linewidth=2)
    ax.set_ylabel('Energy Flux (GW)')
    #ax.set_ylim(bottom=0)
    ax.set_xlim(0,366)
    
    # Surface height
    ax = fig.add_subplot(3,3,5)
    ax.plot(td, ssh, '-b', linewidth=2)
    ax.set_xlim(0,366)
    ax.grid(True)
    ax.set_ylabel('SSH (m)')
    
    # Volume flux
    ax = fig.add_subplot(3,3,8)
    ax.plot(td, qnet/1e3, '-c', linewidth=2)
    ax.plot(td, Qnet/1e3, '--r', linewidth=2)
    ax.set_xlim(0,366)
    ax.grid(True)
    ax.set_xlabel('Days from 1/1/' + str(year))
    ax.set_ylabel('Qnet 1000 m3/s')
    
    if save_fig:
        plt.savefig(outdir + sn + '.png')
        fig.clear()
    else:
        plt.show()

tasks = len(os.sched_getaffinity(0))
with Pool(tasks) as p:
    p.map(process_section, sect_list)
