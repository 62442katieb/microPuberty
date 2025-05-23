#!/usr/bin/env python
# coding: utf-8

#import enlighten
import json

import numpy as np
import pandas as pd

from os.path import join, exists
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit, RepeatedKFold, cross_val_score
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

#PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/microPuberty"
PROJ_DIR = "./"
DAT_DIR = "data"
OUT_DIR = "output"
FIG_DIR = "figures"


SEXES = [
    #'F',
    'M'
]

WAVES = [
    0,
    #2
]

hormones = {
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

MODEL_VARS = [
  'birth_weight',
  'devhx_3_p',
  'dmri_meanmotion',
  'devhx_9_alcohol',
  'devhx_9_tobacco',
  'physical_activity1_y',
  'sds_p_ss_total',
  'cna_p_ss_sum',
  'interview_age',
  'waist_height',
  'since_midnight',
  'collection_duration',
  'time_to_freeze',
  'hormone_sal_caff_y',
  'hormone_sal_active'
]

CATEGORICAL = [
    'highest_parent_educ_tp',
    'household_income_4bins_tp',
    'mri_info_manufacturer',
    'race_ethnicity_c_bl',
    'site_id_l'
]

n_splits = 1000
#scoring = ['neg_log_loss', 'roc_auc_ovo_weighted', 'neg_hinge_loss']
outer_cv = GroupShuffleSplit(test_size=3, n_splits=n_splits)
inner_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=7)

algo = MultiTaskElasticNetCV(
    l1_ratio=[.5, .7, .9],
    fit_intercept=True,
    cv=inner_cv,
    max_iter=1000,
    n_jobs=6
)

# now we get into the XAI part of the adventure
# use vars of interest to predict cluster belonging
# and then assess importance of each feature
rsi = [
    'rni', 
    'rnd'
]
progress = {}
for sex in SEXES:
    print(sex)
    for wave in WAVES:
        print(wave)
        df = pd.read_pickle(
            join(PROJ_DIR, DAT_DIR, f'{sex}-{wave}_clean.pkl')
        )
        for measure in rsi:
            features = MODEL_VARS + list(df.filter(like=f'dmri_rsi{measure}').columns)
            temp_df = df.copy()
            for var in CATEGORICAL:
                dummies = pd.get_dummies(df[var], drop_first=True).replace({True: 1, False: 0})
                features += list(dummies.columns)
                temp_df = pd.concat(
                    [
                        temp_df.drop(var, axis=1),
                        dummies
                    ],
                    axis=1
                )
            #manager = enlighten.get_manager()
            #tocks = manager.counter(total=len(hormones[sex]) * n_splits, desc='Progress', unit='models')
            for hormone in hormones[sex]:
                print(hormone)
                temp_dat = temp_df.copy()
                vars_of_interest = features + [hormone, 'pds_p_average']
                temp_dat = temp_dat[vars_of_interest].dropna()
                
                Y = temp_dat[[hormone, 'pds_p_average']]
                X = temp_dat[features]
                # Nested CV with parameter optimization
                #######################################
                # TRAINING SUBSET
                groups = df.loc[temp_dat.index]['site_id_l']
                importance_df = pd.DataFrame()
                brain_regions = pd.DataFrame(
                    index=list(df.filter(like='dmri_rsi').columns),
                    columns=pd.MultiIndex.from_product(
                        [
                            [hormone, 'pds'],
                            range(10)
                        ]
                    )
                )
                model_stats = pd.DataFrame(
                    index=range(n_splits),
                    columns=['l1:l2', 'score']
                )
                for i, (train_index, test_index) in enumerate(outer_cv.split(X, Y, groups)):
                    #if not i % 10:
                        #print(f"Iteration {i}:\t{str(datetime.now())}")
                    Y_train = Y.iloc[train_index]
                    Y_test = Y.iloc[test_index]

                    X_train = X.iloc[train_index]
                    X_test = X.iloc[test_index]
                    
                    algo.fit(X_train, Y_train)
                    #print(algo.l1_ratio_)
                    model_stats = model_stats.copy()
                    model_stats.at[i, 'l1:l2'] = algo.l1_ratio_
                    
                    score = algo.score(X_test, Y_test)
                    model_stats.at[i, 'score'] = score

                    coefficients = pd.DataFrame(
                        algo.coef_.T, 
                        index=features, 
                        columns=[hormone, 'pds']
                    )
                    brain_regions = brain_regions.copy()
                    for col in coefficients.columns:
                        nonzero = coefficients[coefficients[col] != 0].filter(like='dmri_rsi', axis=0).index
                        for region in nonzero:
                            brain_regions.at[region, (col,i)] = score

                    #######################################
                    # TESTING SUBSET
                    r = permutation_importance(
                        algo, 
                        X_test, 
                        Y_test, 
                        n_repeats=50, 
                        random_state=0, 
                        n_jobs=6
                        #scoring=scorer
                    )

                    sorted_importances_idx = r.importances_mean.argsort()
                    
                    importance_temp = pd.DataFrame(
                        r.importances[sorted_importances_idx].T,
                        columns=X_test.columns[sorted_importances_idx],
                    ) / score
                    importance_df = pd.concat(
                        [
                            importance_df,
                            importance_temp
                        ],
                        axis=0
                    )
                    #tocks.update()
                    if not i % 10:
                        progress[f'{sex}, {wave}, {hormone.split("_")[1]}, {measure}, {i}'] = str(datetime.now())
                        with open(join(PROJ_DIR, OUT_DIR, 'progress.json'), 'w') as fp:
                            json.dump(progress, fp)
                importance_df.to_pickle(
                    join(PROJ_DIR, OUT_DIR, f'{sex}-{wave}-{hormone}-{measure}-feature_importance.pkl')
                )
                brain_regions.to_pickle(
                    join(PROJ_DIR, OUT_DIR, f'{sex}-{wave}-{hormone}-{measure}-region_by_score.pkl')
                )
                model_stats.to_pickle(
                    join(PROJ_DIR, OUT_DIR, f'{sex}-{wave}-{hormone}-{measure}-model_stats.pkl')
            )

