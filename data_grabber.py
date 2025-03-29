import pyreadr
import enlighten

import numpy as np
import pandas as pd
import seaborn as sns
import abcdWrangler as abcdw
import matplotlib.pyplot as plt

from os.path import join, isdir
from os import makedirs

from sklearn.neighbors import LocalOutlierFactor

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/microPuberty"
DAT_DIR = "data"
OUT_DIR = "output"
FIG_DIR = "figures"

# if the folder you want to save your dataset in doesn't exist, this will create it for you
if not isdir(DAT_DIR):
    makedirs(DAT_DIR)

ABCD_DIR = "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.1/abcd-data-release-5.1"

VARS = [
    "interview_date",
    "interview_age",

    "anthroheightcalc",
    "anthroweightcalc",
    # waist circumference
    # might use to calc w/h ratio?
    "anthro_waist_cm",
    ## DAG variables
    # sleep quality -- this is sleep disturbance scale
    # will prob use Devyn's sleep quality code instead, 
    'sds_p_ss_total',
    'physical_activity1_y',
    # avg cortical thickness?
    "pds_p_ss_female_category_2", 
    "pds_p_ss_male_category_2", 
    'site_id_l',
    'site_id_l',
    'pds_1_p',
    'pds_2_p',
    'pds_3_p',

    'pds_f4_p',
    'pds_f5b_p',
    'pds_f6_p',
    'pds_m4_p',
    'pds_m5_p',
]

BASELINE_VARS = [
    # maternal age
    "devhx_3_p",
    # birth weight
    "birth_weight_lbs",
    "birth_weight_oz",
    "devhx_8_tobacco",
    "devhx_8_cigs_per_day",
    "devhx_8_alcohol",
    "devhx_8_alchohol_avg",
    "devhx_9_tobacco",
    "devhx_9_cigs_per_day",
    "devhx_9_alcohol",
    "devhx_9_alchohol_avg",
    'ehi_y_ss_scoreb'
]

DEMO_VARS = [
    'race_ethnicity_c',
    'household_income_4bins_tp',
    'rel_family_id',
    "demo_sex_v2_bl",
    'highest_parent_educ_tp',
]

MRI_VARS = [

    # head motion -- since we don't have FD during t1, 
    # we'll average rsfMRI and dMRI
    'dmri_meanmotion',
    'mri_info_deviceserialnumber',
    'mri_info_manufacturer',
    "imgincl_dmri_include",
    "mrif_score",
    # microstructure time
    'dmri_rsirni_fib_fxrh', #	  right fornix
    'dmri_rsirni_fib_fxlh', #	  left fornix
    'dmri_rsirni_fib_cgcrh', #	  right cingulate cingulum
    'dmri_rsirni_fib_cgclh', #	  left cingulate cingulum
    'dmri_rsirni_fib_cghrh', #	  right parahpcm cingulum
    'dmri_rsirni_fib_cghlh', #	  left parahpcm cingulum
    'dmri_rsirni_fib_cstrh', #	  right corticospinal/pyramidal
    'dmri_rsirni_fib_cstlh', #	  left corticospinal/pyramidal
    'dmri_rsirni_fib_atrrh', #	  right anterior thalamic radiations
    'dmri_rsirni_fib_atrlh', #	  left right anterior thalamic radiations
    'dmri_rsirni_fib_uncrh', #	  right uncinate fasciculus
    'dmri_rsirni_fib_unclh', #	  left uncinate
    'dmri_rsirni_fib_ilfrh', #	  right inferior longitudinal fasciculus
    'dmri_rsirni_fib_ilflh', #	  left inferior longitudinal fasciculus
    'dmri_rsirni_fib_iforh', #	  right inferior fronto-occipital fasciculus
    'dmri_rsirni_fib_ifolh', #	  left inferior fronto-occipital fasciculus
    'dmri_rsirni_fib_fmaj', #	  forceps major
    'dmri_rsirni_fib_fmin', #	  forceps minor
    'dmri_rsirni_fib_cc', #	  corpus callosum
    'dmri_rsirni_fib_slfrh', #	  right superior longitudinal fasciculus
    'dmri_rsirni_fib_slflh', #	  left superior longitudinal fasciculus
    'dmri_rsirni_fib_tslfrh', #	  right temporal superior longitudinal fasiculus
    'dmri_rsirni_fib_tslflh', #	  left temporal superior longitudinal fasiculus
    'dmri_rsirni_fib_pslfrh', #	  right parietal superior longitudinal fasiculus
    'dmri_rsirni_fib_pslflh', #	  left parietal superior longitudinal fasiculus
    'dmri_rsirni_fib_scsrh', #	  right superior corticostriate
    'dmri_rsirni_fib_scslh', #	  left superior corticostriate
    'dmri_rsirni_fib_fscsrh', #	  right superior corticostriate-frontal cortex only
    'dmri_rsirni_fib_fscslh', #	  left superior corticostriate-frontal cortex only
    'dmri_rsirni_fib_pscsrh', #	  right superior corticostriate-parietal cortex only
    'dmri_rsirni_fib_pscslh', #	  left superior corticostriate-parietal cortex only
    'dmri_rsirni_fib_sifcrh', #	  right striatal inferior frontal cortex
    'dmri_rsirni_fib_sifclh', #	  left striatal inferior frontal cortex
    'dmri_rsirni_fib_ifsfcrh', #	  right inferior frontal superior frontal cortex
    'dmri_rsirni_fib_ifsfclh', #	  left inferior frontal superior frontal cortex
    'dmri_rsirni_scs_cbwmlh', #	  left-cerebellum-white-matter
    'dmri_rsirni_scs_cbclh', #	  left-cerebellum-cortex
    'dmri_rsirni_scs_tplh', #	  left-thalamus-proper
    'dmri_rsirni_scs_cdlh', #	  left-caudate
    'dmri_rsirni_scs_ptlh', #	  left-putamen
    'dmri_rsirni_scs_pllh', #	  left-pallidum
    'dmri_rsirni_scs_bs', #	  brain-stem
    'dmri_rsirni_scs_hclh', #	  left-hippocampus
    'dmri_rsirni_scs_aglh', #	  left-amygdala
    'dmri_rsirni_scs_ablh', #	  left-accumbens-area
    'dmri_rsirni_scs_vdclh', #	  left-ventraldc
    'dmri_rsirni_scs_cbwmrh', #	  right-cerebellum-white-matter
    'dmri_rsirni_scs_cbcrh', #	  right-cerebellum-cortex
    'dmri_rsirni_scs_tprh', #	  right-thalamus-proper
    'dmri_rsirni_scs_cdrh', #	  right-caudate
    'dmri_rsirni_scs_ptrh', #	  right-putamen
    'dmri_rsirni_scs_plrh', #	  right-pallidum
    'dmri_rsirni_scs_hcrh', #	  right-hippocampus
    'dmri_rsirni_scs_agrh', #	  right-amygdala
    'dmri_rsirni_scs_abrh', #	  right-accumbens-area
    'dmri_rsirni_scs_vdcrh', #	  right-ventraldc
    'dmri_rsirnigm_cdx_gsfmlh', #	  left hemisphere fronto-marginal gyrus and sulcus
    'dmri_rsirnigm_cdx_gsoilh', #	  left hemisphere inferior occipital gyrus and sulcus
    'dmri_rsirnigm_cdx_gspclh', #	  left hemisphere paracentral lobule and sulcus
    'dmri_rsirnigm_cdx_gssclh', #	  left hemisphere subcentral gyrus and sulci
    'dmri_rsirnigm_cdx_gstflh', #	  left hemisphere transverse frontopolar gyri and sulci
    'dmri_rsirnigm_cdx_gscalh', #	  left hemisphere anterior part of the cingulate gyrus and sulcus
    'dmri_rsirnigm_cdx_gscmalh', #	  left hemisphere middle-anterior part of the cingulate gyrus and sulcus
    'dmri_rsirnigm_cdx_gscmplh', #	  left hemisphere middle-posterior part of the cingulate gyrus and sulcus
    'dmri_rsirnigm_cdx_gcpdlh', #	  left hemisphere posterior-dorsal part of the cingulate gyrus
    'dmri_rsirnigm_cdx_gcpvlh', #	  left hemisphere posterior-ventral part of the cingulate gyrus
    'dmri_rsirnigm_cdx_gcnlh', #	  left hemisphere cuneus
    'dmri_rsirnigm_cdx_gfiolh', #	  left hemisphere opercular part of the inferior frontal gyrus
    'dmri_rsirnigm_cdx_gfioblh', #	  left hemisphere orbital part of the inferior frontal gyrus
    'dmri_rsirnigm_cdx_gfitlh', #	  left hemisphere triangular part of the inferior frontal gyrus
    'dmri_rsirnigm_cdx_gfmlh', #	  left hemisphere middle frontal gyrus
    'dmri_rsirnigm_cdx_gfslh', #	  left hemisphere superior frontal gyrus
    'dmri_rsirnigm_cdx_gilscilh', #	  left hemisphere long insular gyrus and central sulcus of the insula
    'dmri_rsirnigm_cdx_gislh', #	  left hemisphere short insular gyri
    'dmri_rsirnigm_cdx_gomlh', #	  left hemisphere middle occipital gyrus
    'dmri_rsirnigm_cdx_goslh', #	  left hemisphere superior occipital gyrus
    'dmri_rsirnigm_cdx_gotlflh', #	  left hemisphere lateral occipito-temporal gyrus
    'dmri_rsirnigm_cdx_gotmllh', #	  left hemisphere lingual gyrus
    'dmri_rsirnigm_cdx_gotmplh', #	  left hemisphere parahippocampal gyrus
    'dmri_rsirnigm_cdx_golh', #	  left hemisphere orbital gyri
    'dmri_rsirnigm_cdx_gpialh', #	  left hemisphere angular gyrus
    'dmri_rsirnigm_cdx_gpislh', #  left hemisphere supramarginal gyrus
    'dmri_rsirnigm_cdx_gpslh', #  left hemisphere superior parietal lobule
    'dmri_rsirnigm_cdx_gpclh', #  left hemisphere postcentral gyrus
    'dmri_rsirnigm_cdx_gprctlh', #  left hemisphere precentral gyrus
    'dmri_rsirnigm_cdx_gprcnlh', #  left hemisphere precuneus
    'dmri_rsirnigm_cdx_grlh', #  left hemisphere gyrus rectus
    'dmri_rsirnigm_cdx_gslh', #  left hemisphere subcallosal gyrus
    'dmri_rsirnigm_cdx_gtsgttlh', #  left hemisphere anterior transverse temporal gyrus
    'dmri_rsirnigm_cdx_gtsllh', #  left hemisphere lateral aspect of the superior temporal gyrus
    'dmri_rsirnigm_cdx_gtspplh', #  left hemisphere planum polare of the superior temporal gyrus
    'dmri_rsirnigm_cdx_gtsptlh', #  left hemisphere planum temporale
    'dmri_rsirnigm_cdx_gtilh', #  left hemisphere inferior temporal gyrus
    'dmri_rsirnigm_cdx_gtmlh', #  left hemisphere middle temporal gyrus
    'dmri_rsirnigm_cdx_lfahlh', #  left hemisphere horizontal ramus of the anterior segment of the lateral sulcus
    'dmri_rsirnigm_cdx_lfavlh', #  left hemisphere vertical ramus of the anterior segment of the lateral sulcus
    'dmri_rsirnigm_cdx_lfplh', #  left hemisphere posterior ramus of the lateral sulcus
    'dmri_rsirnigm_cdx_polh', #  left hemisphere occipital pole
    'dmri_rsirnigm_cdx_ptlh', #  left hemisphere temporal pole
    'dmri_rsirnigm_cdx_scclh', #  left hemisphere calcarine sulcus
    'dmri_rsirnigm_cdx_sclh', #  left hemisphere central sulcus
    'dmri_rsirnigm_cdx_scmlh', #  left hemisphere marginal branch of the cingulate sulcus
    'dmri_rsirnigm_cdx_scialh', #  left hemisphere anterior segment of the circular sulcus of the insula
    'dmri_rsirnigm_cdx_sciilh', #  left hemisphere inferior segment of the circular sulcus of the insula
    'dmri_rsirnigm_cdx_scislh', #  left hemisphere superior segment of the circular sulcus of the insula
    'dmri_rsirnigm_cdx_sctalh', #  left hemisphere anterior transverse collateral sulcus
    'dmri_rsirnigm_cdx_sctplh', #  left hemisphere posterior transverse collateral sulcus
    'dmri_rsirnigm_cdx_sfilh', #  left hemisphere inferior frontal sulcus
    'dmri_rsirnigm_cdx_sfmlh', #  left hemisphere middle frontal sulcus
    'dmri_rsirnigm_cdx_sfslh', #  left hemisphere superior frontal sulcus
    'dmri_rsirnigm_cdx_sipjlh', #  left hemisphere sulcus intermedius primus
    'dmri_rsirnigm_cdx_siptlh', #  left hemisphere intraparietal sulcus and transverse parietal sulci
    'dmri_rsirnigm_cdx_somllh', #  left hemisphere middle occipital sulcus and lunatus sulcus
    'dmri_rsirnigm_cdx_sostlh', #  left hemisphere superior occipital sulcus and transverse occipital sulcus
    'dmri_rsirnigm_cdx_soalh', #  left hemisphere anterior occipital sulcus and preoccipital notch
    'dmri_rsirnigm_cdx_sotllh', #  left hemisphere lateral occipito-temporal sulcus
    'dmri_rsirnigm_cdx_sotmllh', #  left hemisphere medial occipito-temporal sulcus and lingual sulcus
    'dmri_rsirnigm_cdx_sollh', #  left hemisphere lateral orbital sulcus
    'dmri_rsirnigm_cdx_somolh', #  left hemisphere medial orbital sulcus
    'dmri_rsirnigm_cdx_sohslh', #  left hemisphere orbital sulci
    'dmri_rsirnigm_cdx_spolh', #  left hemisphere parieto-occipital sulcus
    'dmri_rsirnigm_cdx_spclh', #  left hemisphere pericallosal sulcus
    'dmri_rsirnigm_cdx_spctlh', #  left hemisphere postcentral sulcus
    'dmri_rsirnigm_cdx_spriplh', #  left hemisphere inferior part of the precentral sulcus
    'dmri_rsirnigm_cdx_sprsplh', #  left hemisphere superior part of the precentral sulcus
    'dmri_rsirnigm_cdx_ssolh', #  left hemisphere suborbital sulcus
    'dmri_rsirnigm_cdx_ssplh', #  left hemisphere subparietal sulcus
    'dmri_rsirnigm_cdx_stilh', #  left hemisphere inferior temporal sulcus
    'dmri_rsirnigm_cdx_stslh', #  left hemisphere superior temporal sulcus
    'dmri_rsirnigm_cdx_sttlh', #  left hemisphere transverse temporal sulcus
    'dmri_rsirnigm_cdx_gsfmrh', #  right hemisphere fronto-marginal gyrus and sulcus
    'dmri_rsirnigm_cdx_gsoirh', #  right hemisphere inferior occipital gyrus and sulcus
    'dmri_rsirnigm_cdx_gspcrh', #  right hemisphere paracentral lobule and sulcus
    'dmri_rsirnigm_cdx_gsscrh', #  right hemisphere subcentral gyrus and sulci
    'dmri_rsirnigm_cdx_gstfrh', #  right hemisphere transverse frontopolar gyri and sulci
    'dmri_rsirnigm_cdx_gscarh', #  right hemisphere anterior part of the cingulate gyrus and sulcus
    'dmri_rsirnigm_cdx_gscmarh', #  right hemisphere middle-anterior part of the cingulate gyrus and sulcus
    'dmri_rsirnigm_cdx_gscmprh', #  right hemisphere middle-posterior part of the cingulate gyrus and sulcus
    'dmri_rsirnigm_cdx_gcpdrh', #  right hemisphere posterior-dorsal part of the cingulate gyrus
    'dmri_rsirnigm_cdx_gcpvrh', #  right hemisphere posterior-ventral part of the cingulate gyrus
    'dmri_rsirnigm_cdx_gcnrh', #  right hemisphere cuneus
    'dmri_rsirnigm_cdx_gfiorh', #  right hemisphere opercular part of the inferior frontal gyrus
    'dmri_rsirnigm_cdx_gfiobrh', #  right hemisphere orbital part of the inferior frontal gyrus
    'dmri_rsirnigm_cdx_gfitrh', #  right hemisphere triangular part of the inferior frontal gyrus
    'dmri_rsirnigm_cdx_gfmrh', #  right hemisphere middle frontal gyrus
    'dmri_rsirnigm_cdx_gfsrh', #  right hemisphere superior frontal gyrus
    'dmri_rsirnigm_cdx_gilscirh', #  right hemisphere long insular gyrus and central sulcus of the insula
    'dmri_rsirnigm_cdx_gisrh', #  right hemisphere short insular gyri
    'dmri_rsirnigm_cdx_gomrh', #  right hemisphere middle occipital gyrus
    'dmri_rsirnigm_cdx_gosrh', #  right hemisphere superior occipital gyrus
    'dmri_rsirnigm_cdx_gotlfrh', #  right hemisphere lateral occipito-temporal gyrus
    'dmri_rsirnigm_cdx_gotmlrh', #  right hemisphere lingual gyrus
    'dmri_rsirnigm_cdx_gotmprh', #  right hemisphere parahippocampal gyrus
    'dmri_rsirnigm_cdx_gorh', #  right hemisphere orbital gyri
    'dmri_rsirnigm_cdx_gpiarh', #  right hemisphere angular gyrus
    'dmri_rsirnigm_cdx_gpisrh', #  right hemisphere supramarginal gyrus
    'dmri_rsirnigm_cdx_gpsrh', #  right hemisphere superior parietal lobule
    'dmri_rsirnigm_cdx_gpcrh', #  right hemisphere postcentral gyrus
    'dmri_rsirnigm_cdx_gprctrh', #  right hemisphere precentral gyrus
    'dmri_rsirnigm_cdx_gprcnrh', #  right hemisphere precuneus
    'dmri_rsirnigm_cdx_grrh', #  right hemisphere gyrus rectus
    'dmri_rsirnigm_cdx_gsrh', #  right hemisphere subcallosal gyrus
    'dmri_rsirnigm_cdx_gtsgttrh', #  right hemisphere anterior transverse temporal gyrus
    'dmri_rsirnigm_cdx_gtslrh', #  right hemisphere lateral aspect of the superior temporal gyrus
    'dmri_rsirnigm_cdx_gtspprh', #  right hemisphere planum polare of the superior temporal gyrus
    'dmri_rsirnigm_cdx_gtsptrh', #  right hemisphere planum temporale
    'dmri_rsirnigm_cdx_gtirh', #  right hemisphere inferior temporal gyrus
    'dmri_rsirnigm_cdx_gtmrh', #  right hemisphere middle temporal gyrus
    'dmri_rsirnigm_cdx_lfahrh', #  right hemisphere horizontal ramus of the anterior segment of the lateral sulcus
    'dmri_rsirnigm_cdx_lfavrh', #  right hemisphere vertical ramus of the anterior segment of the lateral sulcus
    'dmri_rsirnigm_cdx_lfprh', #  right hemisphere posterior ramus of the lateral sulcus
    'dmri_rsirnigm_cdx_porh', #  right hemisphere occipital pole
    'dmri_rsirnigm_cdx_ptrh', #  right hemisphere temporal pole
    'dmri_rsirnigm_cdx_sccrh', #  right hemisphere calcarine sulcus
    'dmri_rsirnigm_cdx_scrh', #  right hemisphere central sulcus
    'dmri_rsirnigm_cdx_scmrh', #  right hemisphere marginal branch of the cingulate sulcus
    'dmri_rsirnigm_cdx_sciarh', #  right hemisphere anterior segment of the circular sulcus of the insula
    'dmri_rsirnigm_cdx_sciirh', #  right hemisphere inferior segment of the circular sulcus of the insula
    'dmri_rsirnigm_cdx_scisrh', #  right hemisphere superior segment of the circular sulcus of the insula
    'dmri_rsirnigm_cdx_sctarh', #  right hemisphere anterior transverse collateral sulcus
    'dmri_rsirnigm_cdx_sctprh', #  right hemisphere posterior transverse collateral sulcus
    'dmri_rsirnigm_cdx_sfirh', #  right hemisphere inferior frontal sulcus
    'dmri_rsirnigm_cdx_sfmrh', #  right hemisphere middle frontal sulcus
    'dmri_rsirnigm_cdx_sfsrh', #  right hemisphere superior frontal sulcus
    'dmri_rsirnigm_cdx_sipjrh', #  right hemisphere sulcus intermedius primus
    'dmri_rsirnigm_cdx_siptrh', #  right hemisphere intraparietal sulcus and transverse parietal sulci
    'dmri_rsirnigm_cdx_somlrh', #  right hemisphere middle occipital sulcus and lunatus sulcus
    'dmri_rsirnigm_cdx_sostrh', #  right hemisphere superior occipital sulcus and transverse occipital sulcus
    'dmri_rsirnigm_cdx_soarh', #  right hemisphere anterior occipital sulcus and preoccipital notch
    'dmri_rsirnigm_cdx_sotlrh', #  right hemisphere lateral occipito-temporal sulcus
    'dmri_rsirnigm_cdx_sotmlrh', #  right hemisphere medial occipito-temporal sulcus and lingual sulcus
    'dmri_rsirnigm_cdx_solrh', #  right hemisphere lateral orbital sulcus
    'dmri_rsirnigm_cdx_somorh', #  right hemisphere medial orbital sulcus
    'dmri_rsirnigm_cdx_sohsrh', #  right hemisphere orbital sulci
    'dmri_rsirnigm_cdx_sporh', #  right hemisphere parieto-occipital sulcus
    'dmri_rsirnigm_cdx_spcrh', #  right hemisphere pericallosal sulcus
    'dmri_rsirnigm_cdx_spctrh', #  right hemisphere postcentral sulcus
    'dmri_rsirnigm_cdx_spriprh', #  right hemisphere inferior part of the precentral sulcus
    'dmri_rsirnigm_cdx_sprsprh', #  right hemisphere superior part of the precentral sulcus
    'dmri_rsirnigm_cdx_ssorh', #  right hemisphere suborbital sulcus
    'dmri_rsirnigm_cdx_ssprh', #  right hemisphere subparietal sulcus
    'dmri_rsirnigm_cdx_stirh', #  right hemisphere inferior temporal sulcus
    'dmri_rsirnigm_cdx_stsrh', #  right hemisphere superior temporal sulcus
    'dmri_rsirnigm_cdx_sttrh', #  right hemisphere transverse temporal sulcus
    'dmri_rsirnd_fib_fxrh', #  right fornix
    'dmri_rsirnd_fib_fxlh', #  left fornix
    'dmri_rsirnd_fib_cgcrh', #  right cingulate cingulum
    'dmri_rsirnd_fib_cgclh', #  left cingulate cingulum
    'dmri_rsirnd_fib_cghrh', #  right parahpcm cingulum
    'dmri_rsirnd_fib_cghlh', #  left parahpcm cingulum
    'dmri_rsirnd_fib_cstrh', #  right corticospinal/pyramidal
    'dmri_rsirnd_fib_cstlh', #  left corticospinal/pyramidal
    'dmri_rsirnd_fib_atrrh', #  right anterior thalamic radiations
    'dmri_rsirnd_fib_atrlh', #  left right anterior thalamic radiations
    'dmri_rsirnd_fib_uncrh', #  right uncinate fasciculus
    'dmri_rsirnd_fib_unclh', #  left uncinate
    'dmri_rsirnd_fib_ilfrh', #  right inferior longitudinal fasciculus
    'dmri_rsirnd_fib_ilflh', #  left inferior longitudinal fasciculus
    'dmri_rsirnd_fib_iforh', #  right inferior fronto-occipital fasciculus
    'dmri_rsirnd_fib_ifolh', #  left inferior fronto-occipital fasciculus
    'dmri_rsirnd_fib_fmaj', #  forceps major
    'dmri_rsirnd_fib_fmin', #  forceps minor
    'dmri_rsirnd_fib_cc', #  corpus callosum
    'dmri_rsirnd_fib_slfrh', #  right superior longitudinal fasciculus
    'dmri_rsirnd_fib_slflh', #  left superior longitudinal fasciculus
    'dmri_rsirnd_fib_tslfrh', #  right temporal superior longitudinal fasiculus
    'dmri_rsirnd_fib_tslflh', #  left temporal superior longitudinal fasiculus
    'dmri_rsirnd_fib_pslfrh', #  right parietal superior longitudinal fasiculus
    'dmri_rsirnd_fib_pslflh', #  left parietal superior longitudinal fasiculus
    'dmri_rsirnd_fib_scsrh', #  right superior corticostriate
    'dmri_rsirnd_fib_scslh', #  left superior corticostriate
    'dmri_rsirnd_fib_fscsrh', #  right superior corticostriate-frontal cortex only
    'dmri_rsirnd_fib_fscslh', #  left superior corticostriate-frontal cortex only
    'dmri_rsirnd_fib_pscsrh', #  right superior corticostriate-parietal cortex only
    'dmri_rsirnd_fib_pscslh', #  left superior corticostriate-parietal cortex only
    'dmri_rsirnd_fib_sifcrh', #  right striatal inferior frontal cortex
    'dmri_rsirnd_fib_sifclh', #  left striatal inferior frontal cortex
    'dmri_rsirnd_fib_ifsfcrh', #  right inferior frontal superior frontal cortex
    'dmri_rsirnd_fib_ifsfclh', #  left inferior frontal superior frontal cortex
    'dmri_rsirnd_scs_lvlh', # left-lateral-ventricle
    'dmri_rsirnd_scs_cbwmlh', # left-cerebellum-white-matter
    'dmri_rsirnd_scs_cbclh', # left-cerebellum-cortex
    'dmri_rsirnd_scs_tplh', # left-thalamus-proper
    'dmri_rsirnd_scs_cdlh', # left-caudate
    'dmri_rsirnd_scs_ptlh', # left-putamen
    'dmri_rsirnd_scs_pllh', # left-pallidum
    'dmri_rsirnd_scs_bs', # brain-stem
    'dmri_rsirnd_scs_hclh', # left-hippocampus
    'dmri_rsirnd_scs_aglh', # left-amygdala
    'dmri_rsirnd_scs_ablh', # left-accumbens-area
    'dmri_rsirnd_scs_vdclh', # left-ventraldc
    'dmri_rsirnd_scs_cbwmrh', # right-cerebellum-white-matter
    'dmri_rsirnd_scs_cbcrh', # right-cerebellum-cortex
    'dmri_rsirnd_scs_tprh', # right-thalamus-proper
    'dmri_rsirnd_scs_cdrh', # right-caudate
    'dmri_rsirnd_scs_ptrh', # right-putamen
    'dmri_rsirnd_scs_plrh', # right-pallidum
    'dmri_rsirnd_scs_hcrh', # right-hippocampus
    'dmri_rsirnd_scs_agrh', # right-amygdala
    'dmri_rsirnd_scs_abrh', # right-accumbens-area
    'dmri_rsirnd_scs_vdcrh', # right-ventraldc
    'dmri_rsirndgm_cdx_gsfmlh', #  left hemisphere fronto-marginal gyrus and sulcus
    'dmri_rsirndgm_cdx_gsoilh', #  left hemisphere inferior occipital gyrus and sulcus
    'dmri_rsirndgm_cdx_gspclh', #  left hemisphere paracentral lobule and sulcus
    'dmri_rsirndgm_cdx_gssclh', #  left hemisphere subcentral gyrus and sulci
    'dmri_rsirndgm_cdx_gstflh', #  left hemisphere transverse frontopolar gyri and sulci
    'dmri_rsirndgm_cdx_gscalh', #  left hemisphere anterior part of the cingulate gyrus and sulcus
    'dmri_rsirndgm_cdx_gscmalh', #  left hemisphere middle-anterior part of the cingulate gyrus and sulcus
    'dmri_rsirndgm_cdx_gscmplh', #  left hemisphere middle-posterior part of the cingulate gyrus and sulcus
    'dmri_rsirndgm_cdx_gcpdlh', #  left hemisphere posterior-dorsal part of the cingulate gyrus
    'dmri_rsirndgm_cdx_gcpvlh', #  left hemisphere posterior-ventral part of the cingulate gyrus
    'dmri_rsirndgm_cdx_gcnlh', #  left hemisphere cuneus
    'dmri_rsirndgm_cdx_gfiolh', #  left hemisphere opercular part of the inferior frontal gyrus
    'dmri_rsirndgm_cdx_gfioblh', #  left hemisphere orbital part of the inferior frontal gyrus
    'dmri_rsirndgm_cdx_gfitlh', #  left hemisphere triangular part of the inferior frontal gyrus
    'dmri_rsirndgm_cdx_gfmlh', #  left hemisphere middle frontal gyrus
    'dmri_rsirndgm_cdx_gfslh', #  left hemisphere superior frontal gyrus
    'dmri_rsirndgm_cdx_gilscilh', #  left hemisphere long insular gyrus and central sulcus of the insula
    'dmri_rsirndgm_cdx_gislh', #  left hemisphere short insular gyri
    'dmri_rsirndgm_cdx_gomlh', #  left hemisphere middle occipital gyrus
    'dmri_rsirndgm_cdx_goslh', #  left hemisphere superior occipital gyrus
    'dmri_rsirndgm_cdx_gotlflh', #  left hemisphere lateral occipito-temporal gyrus
    'dmri_rsirndgm_cdx_gotmllh', #  left hemisphere lingual gyrus
    'dmri_rsirndgm_cdx_gotmplh', #  left hemisphere parahippocampal gyrus
    'dmri_rsirndgm_cdx_golh', #  left hemisphere orbital gyri
    'dmri_rsirndgm_cdx_gpialh', #  left hemisphere angular gyrus
    'dmri_rsirndgm_cdx_gpislh', #  left hemisphere supramarginal gyrus
    'dmri_rsirndgm_cdx_gpslh', #  left hemisphere superior parietal lobule
    'dmri_rsirndgm_cdx_gpclh', #  left hemisphere postcentral gyrus
    'dmri_rsirndgm_cdx_gprctlh', #  left hemisphere precentral gyrus
    'dmri_rsirndgm_cdx_gprcnlh', #  left hemisphere precuneus
    'dmri_rsirndgm_cdx_grlh', #  left hemisphere gyrus rectus
    'dmri_rsirndgm_cdx_gslh', #  left hemisphere subcallosal gyrus
    'dmri_rsirndgm_cdx_gtsgttlh', #  left hemisphere anterior transverse temporal gyrus
    'dmri_rsirndgm_cdx_gtsllh', #  left hemisphere lateral aspect of the superior temporal gyrus
    'dmri_rsirndgm_cdx_gtspplh', #  left hemisphere planum polare of the superior temporal gyrus
    'dmri_rsirndgm_cdx_gtsptlh', #  left hemisphere planum temporale
    'dmri_rsirndgm_cdx_gtilh', #  left hemisphere inferior temporal gyrus
    'dmri_rsirndgm_cdx_gtmlh', #  left hemisphere middle temporal gyrus
    'dmri_rsirndgm_cdx_lfahlh', #  left hemisphere horizontal ramus of the anterior segment of the lateral sulcus
    'dmri_rsirndgm_cdx_lfavlh', #  left hemisphere vertical ramus of the anterior segment of the lateral sulcus
    'dmri_rsirndgm_cdx_lfplh', #  left hemisphere posterior ramus of the lateral sulcus
    'dmri_rsirndgm_cdx_polh', #  left hemisphere occipital pole
    'dmri_rsirndgm_cdx_ptlh', #  left hemisphere temporal pole
    'dmri_rsirndgm_cdx_scclh', #  left hemisphere calcarine sulcus
    'dmri_rsirndgm_cdx_sclh', #  left hemisphere central sulcus
    'dmri_rsirndgm_cdx_scmlh', #  left hemisphere marginal branch of the cingulate sulcus
    'dmri_rsirndgm_cdx_scialh', #  left hemisphere anterior segment of the circular sulcus of the insula
    'dmri_rsirndgm_cdx_sciilh', #  left hemisphere inferior segment of the circular sulcus of the insula
    'dmri_rsirndgm_cdx_scislh', #  left hemisphere superior segment of the circular sulcus of the insula
    'dmri_rsirndgm_cdx_sctalh', #  left hemisphere anterior transverse collateral sulcus
    'dmri_rsirndgm_cdx_sctplh', #  left hemisphere posterior transverse collateral sulcus
    'dmri_rsirndgm_cdx_sfilh', #  left hemisphere inferior frontal sulcus
    'dmri_rsirndgm_cdx_sfmlh', #  left hemisphere middle frontal sulcus
    'dmri_rsirndgm_cdx_sfslh', #  left hemisphere superior frontal sulcus
    'dmri_rsirndgm_cdx_sipjlh', #  left hemisphere sulcus intermedius primus
    'dmri_rsirndgm_cdx_siptlh', #  left hemisphere intraparietal sulcus and transverse parietal sulci
    'dmri_rsirndgm_cdx_somllh', #  left hemisphere middle occipital sulcus and lunatus sulcus
    'dmri_rsirndgm_cdx_sostlh', #  left hemisphere superior occipital sulcus and transverse occipital sulcus
    'dmri_rsirndgm_cdx_soalh', #  left hemisphere anterior occipital sulcus and preoccipital notch
    'dmri_rsirndgm_cdx_sotllh', #  left hemisphere lateral occipito-temporal sulcus
    'dmri_rsirndgm_cdx_sotmllh', #  left hemisphere medial occipito-temporal sulcus and lingual sulcus
    'dmri_rsirndgm_cdx_sollh', #  left hemisphere lateral orbital sulcus
    'dmri_rsirndgm_cdx_somolh', #  left hemisphere medial orbital sulcus
    'dmri_rsirndgm_cdx_sohslh', #  left hemisphere orbital sulci
    'dmri_rsirndgm_cdx_spolh', #  left hemisphere parieto-occipital sulcus
    'dmri_rsirndgm_cdx_spclh', #  left hemisphere pericallosal sulcus
    'dmri_rsirndgm_cdx_spctlh', #  left hemisphere postcentral sulcus
    'dmri_rsirndgm_cdx_spriplh', #  left hemisphere inferior part of the precentral sulcus
    'dmri_rsirndgm_cdx_sprsplh', #  left hemisphere superior part of the precentral sulcus
    'dmri_rsirndgm_cdx_ssolh', #  left hemisphere suborbital sulcus
    'dmri_rsirndgm_cdx_ssplh', #  left hemisphere subparietal sulcus
    'dmri_rsirndgm_cdx_stilh', #  left hemisphere inferior temporal sulcus
    'dmri_rsirndgm_cdx_stslh', #  left hemisphere superior temporal sulcus
    'dmri_rsirndgm_cdx_sttlh', #  left hemisphere transverse temporal sulcus
    'dmri_rsirndgm_cdx_gsfmrh', #  right hemisphere fronto-marginal gyrus and sulcus
    'dmri_rsirndgm_cdx_gsoirh', #  right hemisphere inferior occipital gyrus and sulcus
    'dmri_rsirndgm_cdx_gspcrh', #  right hemisphere paracentral lobule and sulcus
    'dmri_rsirndgm_cdx_gsscrh', #  right hemisphere subcentral gyrus and sulci
    'dmri_rsirndgm_cdx_gstfrh', #  right hemisphere transverse frontopolar gyri and sulci
    'dmri_rsirndgm_cdx_gscarh', #  right hemisphere anterior part of the cingulate gyrus and sulcus
    'dmri_rsirndgm_cdx_gscmarh', #  right hemisphere middle-anterior part of the cingulate gyrus and sulcus
    'dmri_rsirndgm_cdx_gscmprh', #  right hemisphere middle-posterior part of the cingulate gyrus and sulcus
    'dmri_rsirndgm_cdx_gcpdrh', #  right hemisphere posterior-dorsal part of the cingulate gyrus
    'dmri_rsirndgm_cdx_gcpvrh', #  right hemisphere posterior-ventral part of the cingulate gyrus
    'dmri_rsirndgm_cdx_gcnrh', #  right hemisphere cuneus
    'dmri_rsirndgm_cdx_gfiorh', #  right hemisphere opercular part of the inferior frontal gyrus
    'dmri_rsirndgm_cdx_gfiobrh', #  right hemisphere orbital part of the inferior frontal gyrus
    'dmri_rsirndgm_cdx_gfitrh', #  right hemisphere triangular part of the inferior frontal gyrus
    'dmri_rsirndgm_cdx_gfmrh', #  right hemisphere middle frontal gyrus
    'dmri_rsirndgm_cdx_gfsrh', #  right hemisphere superior frontal gyrus
    'dmri_rsirndgm_cdx_gilscirh', #  right hemisphere long insular gyrus and central sulcus of the insula
    'dmri_rsirndgm_cdx_gisrh', #  right hemisphere short insular gyri
    'dmri_rsirndgm_cdx_gomrh', #  right hemisphere middle occipital gyrus
    'dmri_rsirndgm_cdx_gosrh', #  right hemisphere superior occipital gyrus
    'dmri_rsirndgm_cdx_gotlfrh', #  right hemisphere lateral occipito-temporal gyrus
    'dmri_rsirndgm_cdx_gotmlrh', #  right hemisphere lingual gyrus
    'dmri_rsirndgm_cdx_gotmprh', #  right hemisphere parahippocampal gyrus
    'dmri_rsirndgm_cdx_gorh', #  right hemisphere orbital gyri
    'dmri_rsirndgm_cdx_gpiarh', #  right hemisphere angular gyrus
    'dmri_rsirndgm_cdx_gpisrh', #  right hemisphere supramarginal gyrus
    'dmri_rsirndgm_cdx_gpsrh', #  right hemisphere superior parietal lobule
    'dmri_rsirndgm_cdx_gpcrh', #  right hemisphere postcentral gyrus
    'dmri_rsirndgm_cdx_gprctrh', #  right hemisphere precentral gyrus
    'dmri_rsirndgm_cdx_gprcnrh', #  right hemisphere precuneus
    'dmri_rsirndgm_cdx_grrh', #  right hemisphere gyrus rectus
    'dmri_rsirndgm_cdx_gsrh', #  right hemisphere subcallosal gyrus
    'dmri_rsirndgm_cdx_gtsgttrh', #  right hemisphere anterior transverse temporal gyrus
    'dmri_rsirndgm_cdx_gtslrh', #  right hemisphere lateral aspect of the superior temporal gyrus
    'dmri_rsirndgm_cdx_gtspprh', #  right hemisphere planum polare of the superior temporal gyrus
    'dmri_rsirndgm_cdx_gtsptrh', #  right hemisphere planum temporale
    'dmri_rsirndgm_cdx_gtirh', #  right hemisphere inferior temporal gyrus
    'dmri_rsirndgm_cdx_gtmrh', #  right hemisphere middle temporal gyrus
    'dmri_rsirndgm_cdx_lfahrh', #  right hemisphere horizontal ramus of the anterior segment of the lateral sulcus
    'dmri_rsirndgm_cdx_lfavrh', #  right hemisphere vertical ramus of the anterior segment of the lateral sulcus
    'dmri_rsirndgm_cdx_lfprh', #  right hemisphere posterior ramus of the lateral sulcus
    'dmri_rsirndgm_cdx_porh', #  right hemisphere occipital pole
    'dmri_rsirndgm_cdx_ptrh', #  right hemisphere temporal pole
    'dmri_rsirndgm_cdx_sccrh', #  right hemisphere calcarine sulcus
    'dmri_rsirndgm_cdx_scrh', #  right hemisphere central sulcus
    'dmri_rsirndgm_cdx_scmrh', #  right hemisphere marginal branch of the cingulate sulcus
    'dmri_rsirndgm_cdx_sciarh', #  right hemisphere anterior segment of the circular sulcus of the insula
    'dmri_rsirndgm_cdx_sciirh', #  right hemisphere inferior segment of the circular sulcus of the insula
    'dmri_rsirndgm_cdx_scisrh', #  right hemisphere superior segment of the circular sulcus of the insula
    'dmri_rsirndgm_cdx_sctarh', #  right hemisphere anterior transverse collateral sulcus
    'dmri_rsirndgm_cdx_sctprh', #  right hemisphere posterior transverse collateral sulcus
    'dmri_rsirndgm_cdx_sfirh', #  right hemisphere inferior frontal sulcus
    'dmri_rsirndgm_cdx_sfmrh', #  right hemisphere middle frontal sulcus
    'dmri_rsirndgm_cdx_sfsrh', #  right hemisphere superior frontal sulcus
    'dmri_rsirndgm_cdx_sipjrh', #  right hemisphere sulcus intermedius primus
    'dmri_rsirndgm_cdx_siptrh', #  right hemisphere intraparietal sulcus and transverse parietal sulci
    'dmri_rsirndgm_cdx_somlrh', #  right hemisphere middle occipital sulcus and lunatus sulcus
    'dmri_rsirndgm_cdx_sostrh', #  right hemisphere superior occipital sulcus and transverse occipital sulcus
    'dmri_rsirndgm_cdx_soarh', #  right hemisphere anterior occipital sulcus and preoccipital notch
    'dmri_rsirndgm_cdx_sotlrh', #  right hemisphere lateral occipito-temporal sulcus
    'dmri_rsirndgm_cdx_sotmlrh', #  right hemisphere medial occipito-temporal sulcus and lingual sulcus
    'dmri_rsirndgm_cdx_solrh', #  right hemisphere lateral orbital sulcus
    'dmri_rsirndgm_cdx_somorh', #  right hemisphere medial orbital sulcus
    'dmri_rsirndgm_cdx_sohsrh', #  right hemisphere orbital sulci
    'dmri_rsirndgm_cdx_sporh', #  right hemisphere parieto-occipital sulcus
    'dmri_rsirndgm_cdx_spcrh', #  right hemisphere pericallosal sulcus
    'dmri_rsirndgm_cdx_spctrh', #  right hemisphere postcentral sulcus
    'dmri_rsirndgm_cdx_spriprh', #  right hemisphere inferior part of the precentral sulcus
    'dmri_rsirndgm_cdx_sprsprh', #  right hemisphere superior part of the precentral sulcus
    'dmri_rsirndgm_cdx_ssorh', #  right hemisphere suborbital sulcus
    'dmri_rsirndgm_cdx_ssprh', #  right hemisphere subparietal sulcus
    'dmri_rsirndgm_cdx_stirh', #  right hemisphere inferior temporal sulcus
    'dmri_rsirndgm_cdx_stsrh', #  right hemisphere superior temporal sulcus
    'dmri_rsirndgm_cdx_sttrh', #  right hemisphere transverse temporal sulcus
]


# MIND diet score
nutrition = abcdw.data_grabber(ABCD_DIR, ['cna_p_ss_sum'], '1_year_follow_up_y_arm_1')
# it seems like they only collected the nutrition questionnaire at year-1 so eek
nutrition0 = nutrition.copy()
nutrition.index = pd.MultiIndex.from_product([nutrition.index, ['baseline_year_1_arm_1']])
nutrition0.index = pd.MultiIndex.from_product([nutrition0.index, ['2_year_follow_up_y_arm_1']])

nutrition_both = pd.concat([nutrition, nutrition0], axis=0)

dat = abcdw.data_grabber(ABCD_DIR, BASELINE_VARS, 'baseline_year_1_arm_1')
dat['birth_weight'] = dat["birth_weight_lbs"] + (dat["birth_weight_oz"] / 16)
dat0 = dat.copy()

dat.index = pd.MultiIndex.from_product([dat.index, ['baseline_year_1_arm_1']])
dat0.index = pd.MultiIndex.from_product([dat0.index, ['2_year_follow_up_y_arm_1']])

dat_both = pd.concat([dat, dat0], axis=0)

dat2 = pd.read_pickle('/Volumes/projects_herting/LABDOCS/Personnel/Katie/puberty_profiles/data/filtered_hormones.pkl')
dat2 = dat2.drop(['1_year_follow_up_y_arm_1', '3_year_follow_up_y_arm_1', '4_year_follow_up_y_arm_1'], level=1)

dat3 = abcdw.data_grabber(
    ABCD_DIR, 
    VARS, 
    [
        'baseline_year_1_arm_1', 
        '2_year_follow_up_y_arm_1', 
        #'4_year_follow_up_y_arm_1',
    ],
    multiindex=True,
)

dat4 = abcdw.data_grabber(
    ABCD_DIR, 
    MRI_VARS, 
    [
        'baseline_year_1_arm_1', 
        '2_year_follow_up_y_arm_1', 
        #'4_year_follow_up_y_arm_1',
    ],
    multiindex=True,
)


dat3['waist_height'] = dat3["anthro_waist_cm"] / dat3["anthroheightcalc"]

demo_df = pyreadr.read_r(
    '/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.1/01_Demographics/ABCD_5.1_demographics_complete.RDS'
)
demo_df = demo_df[None]
demo_df0 = demo_df[demo_df['eventname'] == 'baseline_year_1_arm_1'].set_index(['src_subject_id', 'eventname'])
demo_df2 = demo_df[demo_df['eventname'] == '2_year_follow_up_y_arm_1'].set_index(['src_subject_id', 'eventname'])

demo_df = pd.concat([demo_df0, demo_df2], axis=0)

big_dat = pd.concat(
    [
        dat_both, # baseline vars
        dat2, # hormones
        dat3, # 2tpt vars
        demo_df[DEMO_VARS], 
        dat4, # brain
        nutrition_both, # nutrition
    ], 
    axis=1
)

fppts = big_dat[big_dat['demo_sex_v2_bl'] == 'Female'].index
mppts = big_dat[big_dat['demo_sex_v2_bl'] == 'Male'].index

f_pds = [
    'pds_1_p', 
    'pds_2_p', 
    'pds_3_p', 
    'pds_f5b_p', 
    'pds_f4_p',
]

m_pds = [
    'pds_1_p',
    'pds_2_p',
    'pds_3_p',
    'pds_m4_p',
    'pds_m5_p',
]

big_dat['pds_p_ss_category_2'] = big_dat['pds_p_ss_female_category_2'].fillna(0) + big_dat['pds_p_ss_male_category_2'].fillna(0)
big_dat['pds_p_ss_category_2'].replace({0: np.nan}, inplace=True)

f_pds_avg = big_dat.loc[fppts][f_pds].mean(axis=1)
f_pds_avg.name = 'pds_p_average'
m_pds_avg = big_dat.loc[mppts][m_pds].mean(axis=1)
m_pds_avg.name = 'pds_p_average'


big_dat = pd.concat([big_dat, pd.concat([f_pds_avg, m_pds_avg], axis=0)], axis=1)

big_dat.to_pickle(join(PROJ_DIR, DAT_DIR, 'dset.pkl'))
