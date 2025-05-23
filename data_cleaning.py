#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import abcdWrangler as abcdw

from os.path import join
from datetime import datetime
from sklearn.neighbors import LocalOutlierFactor

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/microPuberty"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

DEMO_VARS = [
    "ehi_y_ss_scoreb",
    'race_ethnicity_c_bl',
    'household_income_4bins_bl',
    "highest_parent_educ_bl",
    "site_id_l",
    'rel_family_id',
]

MODEL_VARS = [
  'birth_weight',
  'highest_parent_educ_tp',
  'devhx_3_p',
  'dmri_meanmotion',
  'household_income_4bins_tp',
  'mri_info_manufacturer',
  'devhx_9_alcohol',
  'devhx_9_tobacco',
  'physical_activity1_y',
  'pds_p_ss_category_2',
  'sds_p_ss_total',
  'cna_p_ss_sum',
  'race_ethnicity_c_bl',
  'interview_age',
  'waist_height',
  'demo_sex_v2_bl',
]

CATEGORICAL = [
  'highest_parent_educ_tp',
  'household_income_4bins_tp',
  'mri_info_deviceserialnumber',
  'devhx_8_alcohol',
  'devhx_8_tobacco',
  'pds_p_ss_category_2',
  'race_ethnicity_c_bl',
  'demo_sex_v2_bl',
  'mri_info_manufacturer',
]

SEXES = [
    'F',
    'M'
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

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "dset.pkl"))
df = df.rename({'baseline_year_1_arm_1': '0_year_follow_up_y_arm_1'}, axis=0)
df['interview_date'] = pd.to_datetime(df['interview_date'], format="%m/%d/%Y")

##################################################################################

# need to calculate time since midnight, collection duration, and time to freeze
temp =  pd.to_datetime(df['hormone_sal_start_y'], format='%H:%M') - datetime(year=1900, month=1, day=1, hour=0, minute=0)
temp = [i.total_seconds() / 60 / 60 for i in temp]
df['since_midnight'] = temp
# drop participants who had samples collected outside of 7am-7pm range
poor_hormone_precision = []
poor_hormone_precision += list(df[df['since_midnight'] < 7].index)
poor_hormone_precision += list(df[df['since_midnight'] > 20].index)

temp = pd.to_datetime(df['hormone_sal_end_y'], format='%H:%M') - pd.to_datetime(df['hormone_sal_start_y'], format='%H:%M')
temp = [i.total_seconds() / 60 for i in temp]
df['collection_duration'] = temp
# drop participants with negative collection duration
poor_hormone_precision += list(df[df['collection_duration'] < 0].index)
poor_hormone_precision += list(df[df['collection_duration'] > 15].index)

temp = pd.to_datetime(df['hormone_sal_freezer_y'], format='%H:%M') - pd.to_datetime(df['hormone_sal_end_y'], format='%H:%M')
temp = [i.total_seconds() / 60 for i in temp]
df['time_to_freeze'] = temp
poor_hormone_precision += list(df[df['time_to_freeze'] < 0].index)
poor_hormone_precision += list(df[df['time_to_freeze'] > 10000].index)

temp = df.filter(like='filtered').drop(poor_hormone_precision, axis=0)

df = pd.concat(
    [
        df.drop(df.filter(like='filtered', axis=1).columns, axis=1),
        temp
    ],
    axis=1
)

df = df.replace([999, -np.inf, np.inf], np.nan)

YEARS = [
    '0_year_follow_up_y_arm_1', 
    '2_year_follow_up_y_arm_1', 
]

f_df = df[df['demo_sex_v2_bl'] == "Female"]
m_df = df[df['demo_sex_v2_bl'] == "Male"]

dfs = {
    'F': f_df,
    'M': m_df
}

# Steps
# remove site 15
# assess dmri quality
# remove outliers
# complete case data

for sex in SEXES:
    table = pd.DataFrame(
        index=[
        "N"
        ], 
        columns=pd.MultiIndex.from_product([['ABCD Study'], YEARS])
    )
    temp_df = dfs[sex]

    col_to_df = {
        'ABCD Study': temp_df.index.get_level_values(0).unique(),
    }


    sample_size = pd.DataFrame(
        columns=[
            'keep', 
            'drop'
        ],
        index=[
            'ABCD Study',
            'Not site15',
        ]
    )

    all_ppts = temp_df.index.get_level_values(0).unique()
    for year in YEARS:
        lil_temp = temp_df.xs(year, level=1)
        table.at['N', ('ABCD Study', year)] = len(lil_temp.index)
    
        for col in MODEL_VARS + hormones[sex]:
            table.at[f'{col}-missing', ('ABCD Study', year)] = lil_temp[col].isna().sum()
            if col in CATEGORICAL:
                counts = lil_temp[col].value_counts()
                for level in counts.index:
                    table.at[f'{col}-{level}',('ABCD Study', year)] = counts[level]
            else:
                table.at[f'{col}-mean',('ABCD Study', year)] = lil_temp[col].mean()
                table.at[f'{col}-sdev',('ABCD Study', year)] = lil_temp[col].std()

    temp_df = temp_df[temp_df['site_id_l'] != 'site15']
    not_site15 = temp_df.index.get_level_values(0).unique()
    col_to_df['Not site15'] = not_site15
    for year in YEARS:
        lil_temp = temp_df.loc[not_site15].xs(year, level=1)
        table.at['N', ('Not site15', year)] = len(lil_temp.index)
    
        for col in MODEL_VARS + hormones[sex]:
            table.at[f'{col}-missing', ('Not site15', year)] = lil_temp[col].isna().sum()
            if col in CATEGORICAL:
                counts = lil_temp[col].value_counts()
                for level in counts.index:
                    table.at[f'{col}-{level}',('Not site15', year)] = counts[level]
            else:
                table.at[f'{col}-mean',('Not site15', year)] = lil_temp[col].mean()
                table.at[f'{col}-sdev',('Not site15', year)] = lil_temp[col].std()

    sample_size.at['ABCD Study', 'keep'] = len(all_ppts)
    sample_size.at['ABCD Study', 'drop'] = 0
    sample_size.at['Not site15', 'keep'] = len(not_site15)
    sample_size.at['Not site15', 'drop'] = len(all_ppts) - len(not_site15)

    good_dmri = abcdw.dmri_qc(temp_df, motion_thresh=2)
    temp_temp = temp_df.filter(like='dmri_rsi', axis=1).loc[good_dmri]
    for year in YEARS:
        good_dmri = abcdw.dmri_qc(temp_df, motion_thresh=2)
        ano = year.split("_")[0]
        temp2 = temp_df.xs(year, level=1)
        
        good_dmri = [i[0] for i in good_dmri if i[1] == year]
        col_to_df[f'dMRI QC {ano}'] = good_dmri
        table.at['N', (f'dMRI QC {ano}', year)] = len(good_dmri)
    
        for col in MODEL_VARS + hormones[sex]:
            table.at[f'{col}-missing', (f'dMRI QC {ano}', year)] = temp2[col].isna().sum()
            if col in CATEGORICAL:
                counts = temp2[col].value_counts()
                for level in counts.index:
                    table.at[f'{col}-{level}',(f'dMRI QC {ano}', year)] = counts[level]
            else:
                table.at[f'{col}-mean',(f'dMRI QC {ano}', year)] = temp2[col].mean()
                table.at[f'{col}-sdev',(f'dMRI QC {ano}', year)] = temp2[col].std()

        # imaging quality control at baselien
        sample_size.at[f'dMRI QC {ano}', 'keep'] = len(good_dmri)
        sample_size.at[f'dMRI QC {ano}', 'drop'] = len(not_site15) - len(good_dmri)
        if '2' in year:
            pre_covid = temp2[temp2["interview_date"] < '2020-03-01'].index.unique()
            pre_covid = list(set(pre_covid) & set(good_dmri))
            sample_size.at['pre-covid', 'keep'] = len(pre_covid)
            sample_size.at['pre-covid', 'drop'] = len(good_dmri) - len(pre_covid)
            col_to_df['pre-covid'] = pre_covid
            good_dmri = list(set(good_dmri) & set(pre_covid))
            table.at['N', ('pre-covid', year)] = len(good_dmri)
            lil_temp = temp2.loc[good_dmri]
            for col in MODEL_VARS + hormones[sex]:
                table.at[f'{col}-missing', ('pre-covid', year)] = lil_temp[col].isna().sum()
                if col in CATEGORICAL:
                    counts = lil_temp[col].value_counts()
                    for level in counts.index:
                        table.at[f'{col}-{level}',('pre-covid', year)] = counts[level]
                else:
                    table.at[f'{col}-mean',('pre-covid', year)] = lil_temp[col].mean()
                    table.at[f'{col}-sdev',('pre-covid', year)] = lil_temp[col].std()


        vars_of_interest = list(temp2.filter(like='dmri_rsi').columns) + hormones[sex]
        temp = temp2[vars_of_interest].loc[good_dmri].dropna()
        clf = LocalOutlierFactor(
            n_neighbors=25
        )
        outliers = pd.Series(
            clf.fit_predict(temp),
            index=temp.index
        )

        remove = outliers[outliers < 0].index
        inliers = list(set(good_dmri) - set(remove))
        col_to_df[f'outlier screening {ano}'] = inliers
        sample_size.at[f'outlier screening {ano}', 'keep'] = len(inliers)
        sample_size.at[f'outlier screening {ano}', 'drop'] = len(remove)
        table.at['N', (f'outlier screening {ano}', year)] = len(inliers)
        lil_temp = temp2.loc[inliers]
        for col in MODEL_VARS + hormones[sex]:
            table.at[f'{col}-missing', (f'outlier screening {ano}', year)] = lil_temp[col].isna().sum()
            if col in CATEGORICAL:
                counts = lil_temp[col].value_counts()
                for level in counts.index:
                    table.at[f'{col}-{level}',(f'outlier screening {ano}', year)] = counts[level]
            else:
                table.at[f'{col}-mean',(f'outlier screening {ano}', year)] = lil_temp[col].mean()
                table.at[f'{col}-sdev',(f'outlier screening {ano}', year)] = lil_temp[col].std()


        complete = temp2.loc[inliers][MODEL_VARS + hormones[sex]].dropna().index
        col_to_df[f'complete {ano}'] = complete
        sample_size.at[f'complete {ano}', 'keep'] = len(complete)
        sample_size.at[f'complete {ano}', 'drop'] = len(good_dmri) - len(complete)
        table.at['N', (f'complete {ano}', year)] = len(complete)
        lil_temp = temp2.loc[complete]
        for col in MODEL_VARS + hormones[sex]:
            table.at[f'{col}-missing', (f'complete {ano}', year)] = lil_temp[col].isna().sum()
            if col in CATEGORICAL:
                counts = lil_temp[col].value_counts()
                for level in counts.index:
                    table.at[f'{col}-{level}',(f'complete {ano}', year)] = counts[level]
            else:
                table.at[f'{col}-mean',(f'complete {ano}', year)] = lil_temp[col].mean()
                table.at[f'{col}-sdev',(f'complete {ano}', year)] = lil_temp[col].std()

        lil_temp.to_pickle(
            join(PROJ_DIR, DATA_DIR, f'{sex}-{ano}_clean.pkl')
        )
    table.to_csv(
        join(PROJ_DIR, OUTP_DIR, f'{sex}-demographics.csv')
    )
    sample_size.to_csv(
        join(PROJ_DIR, OUTP_DIR, f'{sex}-sample_size_qc.csv')
    )

    
    ppts = pd.DataFrame(
        index=all_ppts,
        columns=list(col_to_df.keys())
    )

    for ppt in all_ppts:
        for key in col_to_df.keys():
            if ppt in col_to_df[key]:
                ppts.at[ppt, key] = 1
    ppts.to_pickle(join(PROJ_DIR, OUTP_DIR, f'{sex}-ppts_qc.pkl'))

    temp_df = pd.concat(
        [
            temp_df.drop(temp_temp.columns, axis=1),
            temp_temp
        ],
        axis=1
    )

    missingness = pd.DataFrame(
        index=MODEL_VARS,
        columns=YEARS,
        dtype=int
    )
    for year in YEARS:
        temp = temp_df.xs(year, level=1)
        for var in MODEL_VARS + hormones[sex]:
            missingness.at[var, year] = temp[var].isna().sum() / len(temp.index)
    missingness.to_csv(
        join(PROJ_DIR, DATA_DIR, f'{sex}-missingness_model_vals.csv')
    )
