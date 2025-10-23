

#!/usr/bin/env python
# coding: utf-8

#import enlighten
import json
import ggseg

import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib as mpl
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

hormones = {
    'F': [
        'filtered_dhea',
        'filtered_ert',
        'filtered_hse'
    ],
    #'M': [
    #    'filtered_dhea',
    #    'filtered_ert',
    #]
}

fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
destrieux = datasets.fetch_atlas_surf_destrieux()
subcortic = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm")
subcort_array = subcortic.maps.get_fdata()


rni = [
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
    'dmri_rsirnigm_cdx_sttrh', #  right hemisphere transverse temporal sulcu
]
rnd = [
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

mapping = pd.DataFrame(
    columns=[
        'labels', 'rnd_left', 'rnd_right', 'rni_left', 'rni_right'],
)

mapping['labels'] = destrieux.labels

for i in mapping.index[1:]:
    mapping.at[i,'rnd_left'] = rnd[i-1]
    mapping.at[i,'rni_left'] = rni[i-1]
mapping['rnd_right'] = [f'{str(i)[:-2]}rh' for i in mapping['rnd_left']]
mapping['rni_right'] = [f'{str(i)[:-2]}rh' for i in mapping['rni_left']]

subcort={
    #'dmri_rsirni_scs_cbwmlh': 'Left Cerebellum White Matter', #	  left-cerebellum-white-matter
    #'dmri_rsirni_scs_cbclh': 'Left Cerebellum Cortex', #	  left-cerebellum-cortex
    'dmri_rsirni_scs_tplh': 'Left Thalamus', #	  left-thalamus-proper
    'dmri_rsirni_scs_cdlh': 'Left Caudate', #	  left-caudate
    'dmri_rsirni_scs_ptlh': 'Left Putamen', #	  left putamen
    'dmri_rsirni_scs_pllh': 'Left Pallidum', #	  left pallidum
    'dmri_rsirni_scs_bs': 'Brain-Stem', #	  brain-stem
    'dmri_rsirni_scs_hclh': 'Left Hippocampus', #	  left hippocampus
    'dmri_rsirni_scs_aglh': 'Left Amygdala', #	  left amygdala
    'dmri_rsirni_scs_ablh': 'Left Accumbens', #	  left accumbens area
    #'dmri_rsirni_scs_vdclh': 'Left-VentralDC', #	  left-ventraldc
    #'dmri_rsirni_scs_cbwmrh': 'Right Cerebellum White Matter', #	  right-cerebellum-white-matter
    #'dmri_rsirni_scs_cbcrh': 'Right Cerebellum Cortex', #	  right-cerebellum-cortex
    'dmri_rsirni_scs_tprh': 'Right Thalamus', #	  right-thalamus-proper
    'dmri_rsirni_scs_cdrh': 'Left Caudate', #	  right-caudate
    'dmri_rsirni_scs_ptrh': 'Right Putamen', #	  right-putamen
    'dmri_rsirni_scs_plrh': 'Right Pallidum', #	  right-pallidum
    'dmri_rsirni_scs_hcrh': 'Right Hippocampus', #	  right-hippocampus
    'dmri_rsirni_scs_agrh': 'Right Amygdala', #	  right-amygdala
    'dmri_rsirni_scs_abrh': "Right Accumbens", #	  right-accumbens-area
    #'dmri_rsirni_scs_vdcrh': 'Right VentralDC', #	  right-ventraldc
}

fib_to_jhu = {
    #'dmri_rsirni_fib_fxrh':, #	  right fornix
    #'dmri_rsirni_fib_fxlh', #	  left fornix
    'dmri_rsirni_fib_atrlh': 'lh_atr', #	  left right anterior thalamic radiations
    'dmri_rsirni_fib_atrrh': 'rh_atr', #	  right anterior thalamic radiations
    'dmri_rsirni_fib_cstlh': 'lh_cst', #	  left corticospinal/pyramidal
    'dmri_rsirni_fib_cstrh': 'rh_cst', #	  right corticospinal/pyramidal
    'dmri_rsirni_fib_cgclh': 'lh_ccg', #	  left cingulate cingulum
    'dmri_rsirni_fib_cgcrh': 'rh_ccg', #	  right cingulate cingulum
    'dmri_rsirni_fib_cghlh': 'lh_cab', #	  left parahpcm cingulum
    'dmri_rsirni_fib_cghrh': 'rh_cab', #	  right parahpcm cingulum
    'dmri_rsirni_fib_fmaj': 'fmaj', #	  forceps major
    'dmri_rsirni_fib_fmin': 'fmin', #	  forceps minor
    'dmri_rsirni_fib_ifolh': 'lh_fof', #	  left inferior fronto-occipital fasciculus
    'dmri_rsirni_fib_iforh': 'rh_fof', #	  right inferior fronto-occipital fasciculus
    
    'dmri_rsirni_fib_ilflh': 'lh_ilf', #	  left inferior longitudinal fasciculus
    'dmri_rsirni_fib_ilfrh': 'rh_ilf', #	  right inferior longitudinal fasciculus
    
    #'dmri_rsirni_fib_cc': , #	  corpus callosum
    'dmri_rsirni_fib_slflh': 'lh_slf', #	  left superior longitudinal fasciculus
    'dmri_rsirni_fib_slfrh': 'rh_slf', #	  right superior longitudinal fasciculus
    'dmri_rsirni_fib_unclh': 'lh_cnf', #	  left uncinate
    'dmri_rsirni_fib_uncrh': 'rh_unf', #	  right uncinate fasciculus
    'dmri_rsirni_fib_tslflh': 'lh_slft', #	  left temporal superior longitudinal fasiculus
    'dmri_rsirni_fib_tslfrh': 'rh_slft', #	  right temporal superior longitudinal fasiculus
    #'dmri_rsirni_fib_pslfrh', #	  right parietal superior longitudinal fasiculus
    #'dmri_rsirni_fib_pslflh', #	  left parietal superior longitudinal fasiculus
    #'dmri_rsirni_fib_scsrh', #	  right superior corticostriate
    #'dmri_rsirni_fib_scslh', #	  left superior corticostriate
    #'dmri_rsirni_fib_fscsrh', #	  right superior corticostriate-frontal cortex only
    #'dmri_rsirni_fib_fscslh', #	  left superior corticostriate-frontal cortex only
    #'dmri_rsirni_fib_pscsrh', #	  right superior corticostriate-parietal cortex only
    #'dmri_rsirni_fib_pscslh', #	  left superior corticostriate-parietal cortex only
    #'dmri_rsirni_fib_sifcrh', #	  right striatal inferior frontal cortex
    #'dmri_rsirni_fib_sifclh', #	  left striatal inferior frontal cortex
    #'dmri_rsirni_fib_ifsfcrh', #	  right inferior frontal superior frontal cortex
    #'dmri_rsirni_fib_ifsfclh',
}

RSIs = [
    'rni', 
    'rnd'
]


SEXES = {
    'F': 'Female youth',
    'M': 'Male youth'
}

WAVES = [
    0,
    2
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


hormone_dict = {
    'dhea': {
        #'values': dhea_vals,
        'name': 'DHEA',
        'cmap': sns.color_palette('YlOrRd', as_cmap=True) #sns.cubehelix_palette(
            #start=2, 
            #rot=0, 
            #dark=.2, 
            #light=.75, 
            #as_cmap=True
        #)
    },
    'ert': {
        #'values': ert_vals,
        'name': 'T',
        'cmap': sns.color_palette('BuGn', as_cmap=True) #sns.cubehelix_palette(
            #start=.5, 
            #rot=-.5, 
            #dark=.2, 
            #light=.75, 
            #as_cmap=True
        #)
    },
    'hse': {
        #'values': hse_vals,
        'name': 'E2',
        'cmap': sns.color_palette('RdPu', as_cmap=True) #sns.cubehelix_palette(
            #dark=.2, 
            #light=.75, 
            #as_cmap=True
        #)
    },
}

ages = [
    '9-11 years',
    '',
    '11-13 years'
]

'''
filter regions that are only important in a few models
'''
hemis = ['left', 'right']

for rsi in RSIs:
    for sex in SEXES.keys():
        for hormone in HORMONES[sex]:
            for wave in WAVES:
                try:
                    # load in relevance scores
                    cmap = hormone_dict[hormone.split('_')[1]]['cmap']
                    abbrev = hormone_dict[hormone.split('_')[1]]['name']
                    vmax = 0.14
                    threshold = 0.001
                    label = f'{SEXES[sex]}, {ages[wave]}: {abbrev}-related {rsi.upper()}'
                    dat = pd.read_pickle(
                        join(PROJ_DIR, OUT_DIR, f'{sex}-{wave}-{hormone}-{rsi}-region_by_score.pkl')
                    )
                    print('Loaded ', join(PROJ_DIR, OUT_DIR, f'{sex}-{wave}-{hormone}-{rsi}-region_by_score.pkl'))
                    dat.drop('dmri_rsirnd_scs_lvlh', axis=0)
                    #dat_hormone = dat[hormone].T.describe().T[['mean', 'std', '25%', '50%']]
                    #dat_puberty = dat['pds'].T.describe().T[['mean', 'std', '25%', '50%']]
                    dat_hormone = pd.Series(
                        index=dat.index,
                        name=f'{sex} {ages[wave]} {hormone} {rsi}'
                    )
                    for i in dat_hormone.index:
                        nans = dat.loc[i][hormone].isna().sum()
                        if nans / len(dat.loc[i][hormone]) < 0.5:
                            dat_hormone.loc[i] = dat[hormone].loc[i].dropna().mean()
                        else:
                            dat_hormone.loc[i] = 0

                    rni_subcort = pd.DataFrame(index=list(subcort.keys()))
                    for i in rni_subcort.index:
                        rni_subcort.at[i, 'label'] = subcort[i]
                        rni_subcort.at[i, hormone]  = dat_hormone.loc[i]
                    rni_subcort.to_csv(join(PROJ_DIR, OUT_DIR, f'{sex}-{wave}-{hormone}-{rsi}-subcort.csv'))
                    if rni_subcort.sum(numeric_only=True)[hormone] > 0:
                        rnd_f_0 = np.zeros_like(subcort_array)
                        for var in subcort.keys():
                            region = subcort[var]
                            index = subcortic.labels.index(region)
                            rnd_f_0 = np.where(subcort_array == index, rni_subcort.loc[var][hormone], rnd_f_0)
                        subcort_nifti = nib.Nifti1Image(rnd_f_0, subcortic.maps.affine, subcortic.maps.header)
                        
                        xyz = plotting.plot_roi(
                            subcort_nifti, cmap=cmap, draw_cross=False,
                            threshold=threshold, vmax=vmax, black_bg=False, 
                            title=label, colorbar=False, 
                            output_file=join(
                                PROJ_DIR, 
                                FIG_DIR, 
                                f'{sex}-{wave}-{hormone.split("_")[1]}-{rsi}-subcort.png'
                            )
                        )
                    else:
                        pass

                    wm_dhea = pd.DataFrame(index=list(fib_to_jhu.keys()))
                    for i in wm_dhea.index:
                        wm_dhea.at[i, 'jhu'] = fib_to_jhu[i]
                        wm_dhea.at[i, hormone] = dat_hormone.loc[i]
                    if wm_dhea.sum(numeric_only=True)[hormone] > 0:
                        rnd_f_0 = np.zeros_like(subcort_array)
                        for var in subcort.keys():
                            region = subcort[var]
                            index = subcortic.labels.index(region)
                            rnd_f_0 = np.where(subcort_array == index, rni_subcort.loc[var][hormone], rnd_f_0)
                        subcort_nifti = nib.Nifti1Image(rnd_f_0, subcortic.maps.affine, subcortic.maps.header)
                        
                        xyz = plotting.plot_roi(
                            subcort_nifti, cmap=cmap, draw_cross=False,
                            threshold=threshold, vmax=vmax, black_bg=False, 
                            title=label, colorbar=False, 
                            output_file=join(
                                PROJ_DIR, 
                                FIG_DIR, 
                                f'{sex}-{wave}-{hormone.split("_")[1]}-{rsi}-subcort.png'
                            )
                        )
                    else:
                        pass # will fix this when I learn how to plot white matter

                    wm_dhea.replace(0, np.nan).to_csv(join(PROJ_DIR, OUT_DIR, f'{sex}-{wave}-{hormone}-{rsi}-wm_jhu.csv'))

                    rsi_df = mapping.copy()

                    rsi_left = pd.Series(index=range(destrieux['map_left'].shape[0]))
                    rsi_left.loc[np.where(destrieux['map_left'] == 0)] = 0
                    rsi_right = pd.Series(index=range(destrieux['map_right'].shape[0]))
                    rsi_right.loc[np.where(destrieux['map_right'] == 0)] = 0

                    for i in rsi_df.index[1:]:
                        # left first
                        #print('hi!')
                        abcd_name = rsi_df.loc[i][f'{rsi}_left']
                        mean_imp = dat_hormone.loc[abcd_name]
                        rsi_df.at[i,f'{hormone}_left'] = mean_imp
                        indices = np.where(destrieux['map_left'] == i)
                        rsi_left.loc[indices] = mean_imp

                        abcd_name = rsi_df.loc[i][f'{rsi}_right']
                        mean_imp = dat_hormone.loc[abcd_name]
                        rsi_df.at[i,f'{hormone}_left'] = mean_imp
                        indices = np.where(destrieux['map_left'] == i)
                        rsi_right.loc[indices] = mean_imp
                    print('check point')
                    vals = {
                        'left': rsi_left.values,
                        'right': rsi_right.values,
                    }
                    if sum(rsi_left.values) + sum(rsi_right.values) > 0:
                    
                        fig,ax = plt.subplots(
                            nrows=2, ncols=2, figsize=(6,4),
                            subplot_kw={"projection": "3d"}
                        )
                        fig.suptitle(label)
                        plt.tight_layout(w_pad=-10, h_pad=-2)
                        g = plotting.plot_surf_stat_map(
                            fsaverage[f'pial_left'], 
                            #destrieux['map_left'], 
                            vals['left'],
                            hemi='left', 
                            cmap=cmap,
                            bg_map=fsaverage[f'sulc_left'],
                            bg_on_data=True,
                            threshold=0.01,
                            avg_method='median',
                            colorbar=False,
                            vmax=vmax,
                            #engine='plotly',
                            figure=fig,
                            axes=ax[0,0]
                        )
                        h = plotting.plot_surf_stat_map(
                            fsaverage[f'pial_left'], 
                            #destrieux['map_left'], 
                            vals['left'],
                            hemi='right', 
                            cmap=cmap,
                            bg_map=fsaverage[f'sulc_left'],
                            bg_on_data=True,
                            threshold=0.01,
                            avg_method='median',
                            colorbar=False,
                            vmax=vmax,
                            #engine='plotly',
                            figure=fig,
                            axes=ax[1,0]
                        )
                        k = plotting.plot_surf_stat_map(
                            fsaverage[f'pial_right'], 
                            #destrieux['map_left'], 
                            vals['right'],
                            hemi='right', 
                            cmap=cmap,
                            bg_map=fsaverage[f'sulc_right'],
                            bg_on_data=True,
                            threshold=0.01,
                            colorbar=False,
                            avg_method='median',
                            vmax=vmax,
                            #engine='plotly',
                            figure=fig,
                            axes=ax[0,1]
                        )
                        l = plotting.plot_surf_stat_map(
                            fsaverage[f'pial_right'], 
                            #destrieux['map_left'], 
                            vals['right'],
                            hemi='left', 
                            cmap=cmap,
                            bg_map=fsaverage[f'sulc_right'],
                            bg_on_data=True,
                            threshold=0.01,
                            avg_method='median',
                            colorbar=False,
                            vmax=vmax,
                            #engine='plotly',
                            figure=fig,
                            axes=ax[1,1]
                        )
                        
                        print('saving...')
                        fig.savefig(
                            join(
                                PROJ_DIR, 
                                FIG_DIR, 
                                f'{sex}-{wave}-{hormone.split("_")[1]}-{rsi}-cortex.png'
                            ),
                            dpi=400,
                            facecolor='#FFFFFF',
                            bbox_inches='tight'
                        )
                    else:
                        pass
                    fig, ax = plt.subplots(figsize=(4, .5), layout='constrained')
                    norm = mpl.colors.Normalize(vmin=0.01, vmax=vmax)
                    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                                cax=ax, orientation='horizontal', label='$R^2$')
                    fig.savefig(
                        join(
                            PROJ_DIR, 
                            FIG_DIR, 
                            f'{sex}-{wave}-{hormone.split("_")[1]}-{rsi}-cortex_cbar.png'
                        ),
                        dpi=400,
                        facecolor='#FFFFFF',
                        bbox_inches='tight'
                    )
                except Exception as e:
                    print(e)

