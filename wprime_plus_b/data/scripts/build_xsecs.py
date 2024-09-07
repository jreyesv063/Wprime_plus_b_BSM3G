import json

"""
https://github.com/jennetd/hbb-coffea/blob/master/xsec-json.ipynb
"""

xs = {}

# Branching ratios
BR_THadronic = 0.665
BR_TLeptonic = 1 - BR_THadronic

# TTbar
xs["TTTo2L2Nu"] = 6.871e+02 * BR_TLeptonic**2
xs["TTToHadronic"] = 6.871e+02 * BR_THadronic**2
xs["TTToSemiLeptonic"] = 6.871e+02 * 2 * BR_TLeptonic * BR_THadronic

# Single Top
xs["ST_s-channel_4f_leptonDecays"] = 3.549e+00 * BR_TLeptonic
    
# W+jets W(lv)
xs["WJetsToLNu_HT-70To100"] = 1.270e+03
xs["WJetsToLNu_HT-100To200"] = 1.252e+03 
xs["WJetsToLNu_HT-200To400"] = 3.365e+02 
xs["WJetsToLNu_HT-400To600"] = 4.512e+01 
xs["WJetsToLNu_HT-600To800"] = 1.099e+01 
xs["WJetsToLNu_HT-800To1200"] = 4.938e+00 
xs["WJetsToLNu_HT-1200To2500"] = 1.155e+00 
xs["WJetsToLNu_HT-2500ToInf"] = 2.625e-02 
        
# DY+jets
xs["DYJetsToLL_HT-70to100"] = 1.399e+02
xs["DYJetsToLL_HT-100to200"] = 1.401e+02
xs["DYJetsToLL_HT-200to400"] = 3.835e+01
xs["DYJetsToLL_HT-400to600"] = 5.217e+00
xs["DYJetsToLL_HT-600to800"] = 1.267e+00
xs["DYJetsToLL_HT-800to1200"] = 5.682e-01
xs["DYJetsToLL_HT-1200to2500"] = 1.332e-01
xs["DYJetsToLL_HT-2500toInf"] = 2.978e-03

xs["DYJetsToLL_Pt-50To100"] = 3.941e+02
xs["DYJetsToLL_Pt-100To250"] = 9.442e+01
xs["DYJetsToLL_Pt-250To400"] = 3.651e+00
xs["DYJetsToLL_Pt-400To650"] = 4.986e-01
xs["DYJetsToLL_Pt-650ToInf"] = 4.678e-02

# VV 
xs["WW"] = 7.583e+01
xs["WZ"] = 2.756e+01
xs["ZZ"] = 1.214e+01

if __name__ == "__main__":
    with open("../xsec.json", "w") as outfile:
        json.dump(xs, outfile, indent=4)