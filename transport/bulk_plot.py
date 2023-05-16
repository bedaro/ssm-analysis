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
import matplotlib.patches as patches
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

    # make vector and array times in dates
    td = pd.Timestamp('1/1/1970') + pd.to_timedelta(ot, 's')
    Time = np.broadcast_to(td.to_numpy(), (NS, NT)).T

    dir_str = "Up-Estuary"

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

    tef_df = pd.DataFrame(index=td, columns=['Qin','Qout','Sin','Sout'])
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
    fig = plt.figure(num=1 if save_fig else None, figsize=(14,9))
    
    alpha = .5

    # Salinity vs. Time (size and color by Transport)
    ax = fig.add_subplot(2,2,1)
    Qscale = np.nanmean(np.abs(QQ))
    qf = 25
    ax.scatter(Time, SS, s=qf*np.abs(QQp/Qscale), c='tab:red', alpha=alpha)
    ax.scatter(Time, SS, s=qf*np.abs(QQm/Qscale), c='tab:blue', alpha=alpha)
    # add two-layer versions
    if False:
        ax.plot(td, Sin, '-k', td, Sout, '--k')
    else:
        tef_mean_df['Sin'].plot(style='-ok', ax=ax, legend=False)
        tef_mean_df['Sout'].plot(style='--ok', ax=ax, legend=False)
    ax.text(0.05, 0.1, 'Positive is ' + dir_str, transform=ax.transAxes, fontweight='bold')
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_ylabel('Salinity')
    # legend
    ax.scatter(.95, .2, s=qf, c='tab:red', transform=ax.transAxes, alpha=alpha)
    ax.scatter(.95, .1, s=qf, c='tab:blue', transform=ax.transAxes, alpha=alpha)
    ax.text(.94, .2, 'Positive Q %d (m3/s)' % int(Qscale), color='tab:red', fontweight='bold',
        horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.text(.94, .1, 'Negative Q %d (m3/s)' % int(Qscale), color='tab:blue', fontweight='bold',
        horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.set_title(indir0.split('/')[-2])
    
    # # Tidal energy flux vs. Time as second y-axis
    # ax = ax.twinx()
    # ax.plot(td, fnet/1e9, '-g', linewidth=2)
    # ax.set_ylabel('Energy Flux (GW)', color='g', alpha=alpha)
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(0,366)
    
    # Tranport vs. Time
    ax = fig.add_subplot(2,2,3)
    ax.scatter(Time, QQp/1e3, s=qf*np.abs(QQp/Qscale), c='tab:red', alpha=alpha)
    ax.scatter(Time, -QQm/1e3, s=qf*np.abs(QQm/Qscale), c='tab:blue', alpha=alpha)
    # add two-layer versions
    if False:
        ax.plot(td, Qin/1e3, '-k', td, -Qout/1e3, '--k')
    else:
        this_yd = tef_mean_df.loc[:,'yd'].values
        this_qin = tef_mean_df.loc[:,'Qin']/1e3
        this_qout = -tef_mean_df.loc[:,'Qout']/1e3
        this_qin.plot(style='-ok', ax=ax)
        this_qout.plot(style='--ok', ax=ax)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.set_ylabel('|Q| 1000 m3/s')
    
    # Tidal energy flux vs. Time as second y-axis
    ax = fig.add_subplot(3,2,2)
    ax.plot(td, fnet/1e9, '-', color='tab:green', linewidth=2)
    ax.set_ylabel('Energy Flux (GW)')
    #ax.set_ylim(bottom=0)
    
    # Surface height
    ax = fig.add_subplot(3,2,4)
    ax.plot(td, ssh, '-', color='tab:blue', linewidth=2)
    ax.grid(True)
    ax.set_ylabel('SSH (m)')

    # Volume flux
    ax = fig.add_subplot(3,2,6)
    ax.plot(td, qnet/1e3, '-', color='tab:blue', linewidth=2)
    ax.plot(td, Qnet/1e3, '--', color='tab:red', linewidth=2)
    ax.grid(True)
    ax.set_ylabel('Qnet 1000 m3/s')

    if save_fig:
        plt.savefig(outdir + sn + '.png')
        fig.clear()
    else:
        plt.show()

tasks = len(os.sched_getaffinity(0))
with Pool(tasks) as p:
    p.map(process_section, sect_list)
