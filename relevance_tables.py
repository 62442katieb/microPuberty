#!/usr/bin/env python
# coding: utf-8

#import enlighten
import json
import ggseg

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nilearn import datasets, plotting, surface

from os.path import join, exists

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/microPuberty"
DAT_DIR = "data"
OUT_DIR = "output"
FIG_DIR = "figures"


SEXES = [
    'F',
    'M'
]

HORMONES = {
    'F': [
        'filtered_dhea',
        'filtered_ert',
        'filtered_hse'
    ],
    'M': [
        'filtered_dhea',
        'filtered_ert',
    ]
}


RSIs = [
    'rni', 
    'rnd'
]


WAVES = [
    0,
    2
]

hemis = ['left', 'right']

importance_mean = pd.DataFrame(dtype=float)
importance_sdev = pd.DataFrame(dtype=float)
importance_pcnt = pd.DataFrame(dtype=float)

for rsi in RSIs:
    for sex in SEXES:
        for hormone in HORMONES[sex]:
            for wave in WAVES:
                try:
                    # load in relevance scores
                    name = f"{sex}_{wave}-{rsi}_{hormone}"
                    dat = pd.read_pickle(
                        join(PROJ_DIR, OUT_DIR, f'{sex}-{wave}-{hormone}-{rsi}-region_by_score.pkl')
                    )
                    print(rsi, sex, hormone, wave)
                    temp_mean = dat.filter(like='filtered').mean(axis=1)
                    temp_mean.name = name
                    temp_sdev = dat.filter(like='filtered').std(axis=1)
                    temp_sdev.name = name
                    temp_pcnt = (1 - (dat.filter(like='filtered').isna().sum(axis=1) / len(dat.filter(like='filtered').columns))) * 100
                    temp_pcnt.name = name
                    importance_mean = pd.concat(
                        [
                            importance_mean,
                            temp_mean
                        ],
                        axis=1
                    )
                    importance_sdev = pd.concat(
                        [
                            importance_sdev,
                            temp_sdev
                        ],
                        axis=1
                    )
                    importance_pcnt = pd.concat(
                        [
                            importance_pcnt,
                            temp_pcnt
                        ],
                        axis=1
                    )
                except:
                    pass
importance_mean.to_csv(join(PROJ_DIR, OUT_DIR, 'importance_means.csv'))

rni_mean = importance_mean.filter(
    like='rsirni', axis=0
).dropna(
    how='all', axis=1
).dropna(
    how='all', axis=0
)
rni_mean.to_csv(
    join(PROJ_DIR, OUT_DIR, "rni_mean_importance.csv")
)

rnd_mean = importance_mean.filter(
    like='rsirnd', axis=0
).dropna(
    how='all', axis=1
).dropna(
    how='all', axis=0
)
rnd_mean.to_csv(
    join(PROJ_DIR, OUT_DIR, "rnd_mean_importance.csv")
)

rni_sdev = importance_sdev.filter(
    like='rsirni', axis=0
).dropna(
    how='all', axis=1
)
rni_sdev.to_csv(
    join(PROJ_DIR, OUT_DIR, "rni_sdev_importance.csv")
)

rnd_sdev = importance_sdev.filter(
    like='rsirnd', axis=0
).dropna(
    how='all', axis=1
)
rnd_sdev.to_csv(
    join(PROJ_DIR, OUT_DIR, "rnd_sdev_importance.csv")
)

rni_pcnt = importance_pcnt.filter(
    like='rsirni', axis=0
).dropna(
    how='all', axis=1
).dropna(
    how='all', axis=0
)
rnd_pcnt = importance_pcnt.filter(
    like='rsirnd', axis=0
).dropna(
    how='all', axis=1
).dropna(
    how='all', axis=0
)
# now make paper-ready tables?
rni_table_for_paper = pd.DataFrame(dtype=str)
for i in rni_mean.index:
    for j in rni_mean.columns:
        try:
            mean = rni_mean.loc[i][j]
            sdev = rni_sdev.loc[i][j]
            pcnt = rni_pcnt.loc[i][j]
            if mean > 0:
                rni_table_for_paper.at[i,j] = f"{np.round(mean, 3)} ± {np.round(sdev, 3)} ({np.round(pcnt, 0)}%)"
        except Exception as e:
            print(i,j,e)
rni_table_for_paper.to_csv(
    join(PROJ_DIR, OUT_DIR, 'rni_importance_table.csv')
)

# now make paper-ready tables?
rnd_table_for_paper = pd.DataFrame(dtype=str)
for i in rnd_mean.index:
    for j in rnd_mean.columns:
        try:
            mean = rnd_mean.loc[i][j]
            sdev = rnd_sdev.loc[i][j]
            pcnt = rnd_pcnt.loc[i][j]
            if mean > 0:
                rni_table_for_paper.at[i,j] = f"{np.round(mean, 3)} ± {np.round(sdev, 3)} ({np.round(pcnt, 0)}%)"
        except Exception as e:
            print(i,j,e)
rnd_table_for_paper.to_csv(
    join(PROJ_DIR, OUT_DIR, 'rnd_importance_table.csv')
)