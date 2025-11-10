import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nilearn import datasets, plotting

sns.set(context='paper', style='white')

RECEPTORS = [
    'AR', 'ESR1', 'ESR2', 'GPER',
    'GABRA1','GRIN3A','GRIN2D'
]

allen_brain_all = '/Users/katherine.b/Dropbox/Projects/kangaroo/microPuberty/all_plus_extra_estrogen.csv'

er_all = pd.read_csv(allen_brain_all, 
                     #index_col=[0,1], 
                     header=0)
er_all.index = pd.MultiIndex.from_frame((er_all[['donor_id', 'structure_id']]))
er_all = er_all.sort_index()

er_all['Brain Region'] = er_all['structure_abbreviation'].replace(
    {
        'AMY': 'Amyg',
        'MFC': 'rACC/mPFC',
        'CBC': 'Crbllm',
        'DFC': 'dlPFC',
        'HIP': 'Hipp',
        'ITC': 'ITG, BA20',
        'MD': 'MDThal',
        'STC': 'pSTG',
        'IPC': "IPL",
        'A1C': "Aud",
        'S1C': "S1",
        'M1C': "M1",
        'V1C': 'V1',
        'VFC': 'vlPFC',
        'STR': "Striatum"
    }
)


for i in er_all.index:
    num, unit = er_all.loc[i]['donor_age'].split(' ')
    num = int(num)
    if unit == 'pcw':
        er_all.at[i, 'Donor Age'] = 0 - (num / 52)
        er_all.at[i, 'Life Stage'] = 'prenatal'
    elif unit == 'mos':
        er_all.at[i, 'Donor Age'] = num / 12
        er_all.at[i, 'Life Stage'] = 'infant'
    elif unit == 'yrs':
        er_all.at[i, 'Donor Age'] = num
        if num < 9:
            er_all.at[i, 'Life Stage'] = 'child'
        elif num < 11:
            er_all.at[i, 'Life Stage'] = 'pre puberty'
        elif num < 14:
            er_all.at[i, 'Life Stage'] = 'puberty'
        elif num < 20:
            er_all.at[i, 'Life Stage'] = 'post puberty'
        elif num < 30:
            er_all.at[i, 'Life Stage'] = 'young adulth'
        elif num < 55:
            er_all.at[i, 'Life Stage'] = 'adult'
        else:
            er_all.at[i, 'Life Stage'] = 'older adult'

teen_regions = list(er_all[er_all['Donor Age'].between(8,14, inclusive='both')]['structure_abbreviation'].unique())
puberty_brains = er_all[er_all['Donor Age'].between(8,15, inclusive='both')]
adult_brains = er_all[er_all['Donor Age'].between(24,60, inclusive='both')]

region_order = [
    'vlPFC',
    'rACC/mPFC',
    'dlPFC',
    'OFC',
    'Amyg',
    'Hipp',
    "Striatum",
    'MDThal',
    'ITG, BA20',
    'pSTG',
    "IPL",
    "Aud",
    "S1",
    "M1",
    'V1',
    'Crbllm',
]

tol_rainbow = ["#E8ECFB", "#D9CCE3", "#D1BBD7", "#CAACCB", "#BA8DB4",
                "#AE76A3", "#AA6F9E", "#994F88", "#882E72", "#1965B0",
                "#437DBF", "#5289C7", "#6195CF", "#7BAFDE", "#4EB265",
                "#90C987", "#CAE0AB", "#F7F056", "#F7CB45", "#F6C141",
                "#F4A736", "#F1932D", "#EE8026", "#E8601C", "#E65518",
                "#DC050C", "#A5170E", "#72190E", "#42150A"]
tol_indices = [3, 5, 7, 9, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28]
tol_indices = [int(i - 1) for i in tol_indices]
tol_list = [tol_rainbow[i] for i in tol_indices]
tol_pal = sns.color_palette(tol_list)
sns.palplot(tol_pal)


fig,ax = plt.subplots(
    nrows=7, 
    ncols=2, 
    figsize=(5,9), 
    sharex='col',
    sharey='row',
    #layout='tight'
)
for receptor in RECEPTORS:
    temp = RECEPTORS.index(receptor)
    plt.tight_layout(h_pad=1)
    temp_ax = ax[temp,0]
    sns.boxplot(
        puberty_brains,
        x='Brain Region',
        y=receptor,
        order=region_order,
        ax=temp_ax,
        hue='Brain Region',
        palette=tol_pal,
        hue_order=region_order
    )
    temp_ax.axhline(
        puberty_brains[receptor].median(), 
        linestyle='--', 
        color='#888888'
    )
    temp_ax.set_xlabel('Brain Region')
    if temp == 0:
        temp_ax.set_title('Early Adolescence', loc='left')
    for label in temp_ax.get_xticklabels(which='major'):
        label.set(
            rotation=90, 
            #horizontalalignment='right'
        )
    #### adults #############
    temp_ax = ax[temp,1]
    if temp == 0:
        temp_ax.set_title('Adulthood', loc='left')
    sns.boxplot(
        adult_brains,
        x='Brain Region',
        y=receptor,
        order=region_order,
        ax=temp_ax,
        hue='Brain Region',
        palette=tol_pal,
        hue_order=region_order
    )
    temp_ax.axhline(
        adult_brains[receptor].median(), 
        linestyle='--', 
        color='#888888'
    )
    for label in temp_ax.get_xticklabels(which='major'):
        label.set(
            rotation=90, 
            #horizontalalignment='right'
        )
    sns.despine()


fig.savefig(
    'receptors_by_region.png',
    facecolor='#FFFFFF',
    dpi=400,
    bbox_inches='tight'
)