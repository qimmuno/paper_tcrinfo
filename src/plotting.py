import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import pyrepseq as prs
import numpy as np
import string

def get_relevancy_grid(background_dataframe, specific_dataframe, features):
    relevancy_grid = np.zeros((len(features), len(features)))
    for i, feature_1 in enumerate(features):
        for j, feature_2 in enumerate(features[i:], start=i): 
            if feature_1 == feature_2:
                relevancy_grid[i, j] = prs.renyi2_entropy(background_dataframe, feature_1) - prs.renyi2_entropy(specific_dataframe, feature_1, "Epitope")
            else:
                relevancy_grid[i, j] = prs.renyi2_entropy(background_dataframe, [feature_1, feature_2]) - prs.renyi2_entropy(specific_dataframe, [feature_1, feature_2], "Epitope")
            if i != j:
                relevancy_grid[j, i] = relevancy_grid[i, j]  
    return relevancy_grid

def get_cross_relevancy_grid(background_dataframe, specific_dataframe, features_1, features_2):
    
    relevancy_grid = np.zeros((len(features_1), len(features_2)))
    for i, feature_1 in enumerate(features_1):
        for j, feature_2 in enumerate(features_2):
            relevancy_grid[i, j] = prs.renyi2_entropy(background_dataframe, feature_1) + prs.renyi2_entropy(background_dataframe, feature_2) - prs.renyi2_entropy(specific_dataframe, [feature_1, feature_2], "Epitope")
            
    return relevancy_grid

def plot_grid(data_array, labels_index, labels_columns, save_file_name, save_path = '../../manuscript/figures/', facecolor = 'xkcd:black', *args, **kwargs):
    df = pd.DataFrame(data_array, index=labels_index, columns=labels_columns)
    fig, ax = plt.subplots(figsize = data_array.shape)
    sns.heatmap(df, square='true', cmap='Blues', ax=ax, annot=True, *args, **kwargs)
    ax.set_facecolor(facecolor)
    plt.savefig(save_path+f'{save_file_name}',bbox_inches='tight', dpi=500);
    
    
def alpha_beta_entropy_by_epitope_plot(feature_alpha, feature_beta, specific_dataframe, background_dataframe, epitope_meta, save_file_name, labels, markers, save_path = '../../manuscript/figures/', independent_back = True):
    
    #Background statistics 
    background_alpha = prs.renyi2_entropy(background_dataframe, feature_alpha)
    background_beta = prs.renyi2_entropy(background_dataframe, feature_beta)
    
    if independent_back:
        background_full_sequence = background_alpha + background_beta
    else:
        background_full_sequence = prs.renyi2_entropy(background_dataframe, [feature_alpha, feature_beta])    
    
    error_background_alpha = prs.stdrenyi2_entropy(background_dataframe, feature_alpha)
    error_background_beta = prs.stdrenyi2_entropy(background_dataframe, feature_beta)
    
    if independent_back:
        error_background_full_sequence = np.sqrt(error_background_alpha**2 + error_background_beta**2)
    else:
        error_background_full_sequence = prs.stdrenyi2_entropy(background_dataframe, [feature_alpha, feature_beta]) 
        
    #Specific statistics
    specific_alpha = specific_dataframe.groupby('Epitope').apply(lambda x: prs.renyi2_entropy(x, feature_alpha))
    specific_beta = specific_dataframe.groupby('Epitope').apply(lambda x: prs.renyi2_entropy(x, feature_beta))
    specific_full_sequence = specific_dataframe.groupby('Epitope').apply(lambda x: prs.renyi2_entropy(x, [feature_alpha, feature_beta]))
        
    error_specific_alpha = specific_dataframe.groupby('Epitope').apply(lambda x: prs.stdrenyi2_entropy(x, feature_alpha))
    error_specific_beta = specific_dataframe.groupby('Epitope').apply(lambda x: prs.stdrenyi2_entropy(x, feature_beta))
    error_specific_full_sequence = specific_dataframe.groupby('Epitope').apply(lambda x: prs.stdrenyi2_entropy(x, [feature_alpha, feature_beta]))
        
    #General plot settings
    fig, ax = plt.subplots(1, 4, figsize = (12,5), dpi=100, sharey=True, sharex=True, layout='constrained')
    for n, a in enumerate(ax):  
                a.text(-0.1, 1.1, f"\\bf {string.ascii_lowercase[n]})", transform=a.transAxes,
                size=20)
    
    ax[0].set_xlim([-0.5+1,len(epitope_meta)-0.5+1])
    ax[0].set_title(f'{labels[0]}')
    ax[0].set_ylabel("$H_2[P(X)] - H_2[P(X|\pi)]$ \n [bits]")
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax[0].axhline(background_alpha - prs.renyi2_entropy(specific_dataframe, feature_alpha, "Epitope"), c= 'C0', alpha=0.4, linestyle="--")
    
    ax[1].set_title(f'{labels[1]}')
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax[1].axhline(background_beta - prs.renyi2_entropy(specific_dataframe, feature_beta, "Epitope"), c= 'C1', alpha=0.4, linestyle="--")
    
    ax[2].set_title(f'{labels[0]} $\cap$ {labels[1]}')
    ax[2].xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax[2].axhline(background_full_sequence - prs.renyi2_entropy(specific_dataframe, [feature_alpha, feature_beta], "Epitope"), c= 'C2', alpha=0.4, linestyle="--")
    
    ax[3].set_title(f'{labels[0]} $\perp \!\!\! \perp$ {labels[1]}')
    ax[3].xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax[3].axhline(background_alpha+background_beta - prs.renyi2_entropy(specific_dataframe, feature_alpha, "Epitope") - prs.renyi2_entropy(specific_dataframe, feature_beta, "Epitope")
                   , c= 'C3', alpha=0.4, linestyle="--")
    
    ax[0].set_ylim([np.amin(np.concatenate([background_alpha-specific_alpha - 4*np.sqrt(error_background_alpha**2+error_specific_alpha**2),
                             background_beta-specific_beta - 4*np.sqrt(error_background_beta**2+error_specific_beta**2),
                             background_full_sequence-specific_full_sequence - 4*np.sqrt(error_background_full_sequence**2+error_specific_full_sequence**2)])),
                    np.amax(np.concatenate([background_alpha-specific_alpha + 4*np.sqrt(error_background_alpha**2+error_specific_alpha**2),
                             background_beta-specific_beta + 4*np.sqrt(error_background_beta**2+error_specific_beta**2),
                             background_full_sequence-specific_full_sequence + 4*np.sqrt(error_background_full_sequence**2+error_specific_full_sequence**2)]))
                    ])
    ax[0].scatter(10, -1000, marker='o', c='black', label="Minervina")
    ax[0].scatter(10, -1000,  marker='^', c='black', label="Dash")
    ax[0].legend()
    
    #Plot epitope data
    for epitope in epitope_meta['Epitope']:
        id_code = epitope_meta[epitope_meta['Epitope'] == epitope]['id_code'].iloc[0]
        data_set = epitope_meta[epitope_meta['Epitope'] == epitope]['set'].iloc[0]
        
        ax[0].errorbar(id_code, (background_alpha-specific_alpha)[epitope], yerr=np.sqrt(error_background_alpha**2+error_specific_alpha**2)[epitope], fmt=markers[data_set], c='C0')
        ax[1].errorbar(id_code, (background_beta-specific_beta)[epitope], yerr=np.sqrt(error_background_beta**2+error_specific_beta**2)[epitope],fmt=markers[data_set], c='C1')
        ax[2].errorbar(id_code, (background_full_sequence-specific_full_sequence)[epitope], yerr=np.sqrt(error_background_full_sequence**2+error_specific_full_sequence**2)[epitope], fmt=markers[data_set], c='C2')
        ax[3].errorbar(id_code, (background_alpha+background_beta-specific_alpha-specific_beta)[epitope], yerr=np.sqrt(error_background_alpha**2+error_background_beta**2+error_specific_alpha**2+error_specific_beta**2)[epitope], fmt=markers[data_set], c='C3')
        
    fig.supxlabel("Epitope ID")
    plt.savefig(save_path+f'{save_file_name}',bbox_inches='tight', dpi=500);