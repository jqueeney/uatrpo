"""Helper functions and command line parser for plot.py."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import argparse

sns.set()
sns.set_context('paper')

plot_parser = argparse.ArgumentParser()

plot_parser.add_argument('--import_path',help='import path',
    type=str,default='./logs')
plot_parser.add_argument('--import_files',nargs='+',
    help='list of simulation files',type=str)
plot_parser.add_argument('--labels',nargs='+',help='list of labels',type=str)
plot_parser.add_argument('--save_path',help='save path',
    type=str,default='./figs')
plot_parser.add_argument('--save_name',
    help='file name to use when saving plot',type=str,default='userplot')
plot_parser.add_argument('--metric',
    help='metric to plot',type=str,default='J_tot')
plot_parser.add_argument('--window',
    help='number of steps for plot smoothing',type=float,default=1e5)
plot_parser.add_argument('--timesteps',help='number of steps to plot',
    type=float,default=1e6)
plot_parser.add_argument('--interval',help='how often to plot data',
    type=float,default=5e3)
plot_parser.add_argument('--se_val',
    help='standard error multiplier for plot shading',type=float,default=0.5)
plot_parser.add_argument('--figsize',nargs='+',help='figure size',type=int)

def create_plotparser():
    return plot_parser

def aggregate_sim(results,x,window,metric):
    """Computes running averages for all trials."""
    sim = len(results)
    data_all = np.zeros((sim,len(x)))
    for idx in range(sim):
        log = results[idx]['train']
        samples = np.cumsum(log['steps'])
        x_filter = np.argmax(np.expand_dims(samples,1) 
            >= np.expand_dims(x,0),0)

        try:
            data_total = np.squeeze(log[metric])
        except:
            available = ', '.join(list(log.keys()))
            raise ValueError(
                '%s is not a recognized metric. Available metrics include: %s'%(
                    metric,available))

        if window > 1:
            data_totsmooth = np.convolve(np.squeeze(data_total),
                np.ones(window),'full')[:-(window-1)]        
            len_totsmooth = np.convolve(np.ones_like(data_total),
                np.ones(window),'full')[:-(window-1)]     

            data_ave = data_totsmooth / len_totsmooth   
        else:
            data_ave = data_total

        data_all[idx,:] = data_ave[x_filter]
    
    return data_all 

def open_and_aggregate(import_path,import_file,x,window,metric):
    """Returns aggregated data from raw filename."""

    if import_file is None:
        results = None
    else:
        with open(os.path.join(import_path,import_file),'rb') as f:
            data = pickle.load(f)
        
        M = data[0]['param']['runner_kwargs']['M']
        B = data[0]['param']['runner_kwargs']['B']
        n = data[0]['param']['runner_kwargs']['n']
        if M > 1:
            b_size = n
        else:
            b_size = B * n
        
        window_batch = int(window / b_size)
        
        results = aggregate_sim(data,x,window_batch,metric)
    
    return results

def plot_setup(import_path,import_files,x,window,metric):
    """Returns all aggregated data for plotting."""

    results_list = []
    for import_file in import_files:
        results = open_and_aggregate(import_path,import_file,
            x,window,metric)
        results_list.append(results)
    
    return results_list

def create_plot(x,results_list,se_val,labels,figsize,save_path,save_name):
    """Creates and saves plot."""

    if figsize is None:
        pass
    elif len(figsize) >= 2:
        figsize = tuple(figsize[:2])
    elif len(figsize) == 1:
        figsize = (figsize[0],figsize[0])


    fig, ax = plt.subplots(figsize=figsize)

    for file_idx in range(len(results_list)):

        data_active = results_list[file_idx]
        try:
            label_active = labels[file_idx]
        except:
            label_active = 'File %d'%file_idx

        data_mean = np.mean(data_active,axis=0)
        if data_active.shape[0] > 1:
            data_std = np.std(data_active,axis=0,ddof=1)
            data_se = data_std / np.sqrt(data_active.shape[0])
        else:
            data_se = np.zeros_like(data_mean)

        ax.plot(x/1e6,data_mean,color='C%d'%file_idx,label=label_active)
        ax.fill_between(x/1e6,
            data_mean-se_val*data_se,
            data_mean+se_val*data_se,
            alpha=0.2,color='C%d'%file_idx)

    ax.set_xlabel('Steps (M)')
    ax.legend()

    # Save plot
    save_date = datetime.today().strftime('%m%d%y_%H%M%S')
    save_file = '%s_%s'%(save_name,save_date)
    os.makedirs(save_path,exist_ok=True)
    save_filefull = os.path.join(save_path,save_file)

    filename = save_filefull+'.pdf'
    fig.savefig(filename,bbox_inches='tight',dpi=300)