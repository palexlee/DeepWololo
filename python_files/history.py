import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class History(object):
    """Hold the training history"""
    def __init__(self):
        self.hist = pd.DataFrame(columns=['train loss', 'train acc', 'val loss', 'val acc'])
        
    formatters = {
            'train loss': "{:0.8f}".format, 
            'train acc': "{:0.3f}".format,
            'val loss': "{:0.8f}".format, 
            'val acc': "{:0.3f}".format}
    
    def add(self, new_epoch):
        self.hist.loc[len(self.hist)] = new_epoch
        
    def get_last(self):
        return self.hist.tail(1)

    def get_best(self, n=1):
        return self.hist.sort_values('val loss').head(n)
    
    def get_best_val_acc(self):
        return self.hist.sort_values('val acc', ascending=False).head(1)['val acc'].values[0]

    def get_best_epochs_nb(self, n=1):
        return self.hist.sort_values('val loss').head(n).index.tolist()
    
    def get_hist(self):
        return self.hist
    
    def plot(self, title, avg_w_size=20): 
        colors = ['C0', 'C1']
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
        fig.suptitle(title)
        self.hist[['train loss', 'val loss']].ewm(span=avg_w_size).mean().plot(ax=ax1, color=colors)
        self.hist[['train loss', 'val loss']].plot(ax=ax1, alpha=0.4, color=colors, legend=False)
        self.hist[['train acc', 'val acc']].ewm(span=avg_w_size).mean().plot(ax=ax2, color=colors)
        self.hist[['train acc', 'val acc']].plot(ax=ax2, alpha=0.4, color=colors, legend=False)
        ax1.set_ylabel('categorical cross entropy')
        ax1.set_xlabel('epochs')
        ax1.set_yscale('log')
        ax1.grid(color='0.8', linewidth=0.5, ls='--')
        ax2.set_ylabel('accuracy [% correct]')
        ax2.set_xlabel('epochs')
        ax2.grid(color='0.8', linewidth=0.5, ls='--')
