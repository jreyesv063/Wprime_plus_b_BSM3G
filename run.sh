#!/bin/bash

# Before running this script, be sure to grant execution permissions with the following command:
# chmod +x run.sh
# Script is used with the command ./run.sh


echo "########################################"
echo "######  Starting the analysis code #####"
echo "########################################"


# Obtener el directorio del script actual
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Certificado GRID
echo $GRID_PASSWORD | voms-proxy-init --voms cms


# Moverse al directorio del conjunto de archivos
cd wprime_plus_b/fileset/

# Obtener el shell de Singularity
singularity shell -B /afs -B /eos -B /cvmfs /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest-py3.10 << EOF

# Ejecutar el script 'make_fileset_lxplus.py' dentro de Singularity
#python make_fileset_lxplus.py


# Salir del shell de Singularity
exit

EOF


# Volver al directorio donde se encuentra el script
cd "$SCRIPT_DIR"

# Declarar variables
processor="top_tagger"     # ttbar ; ztoll: top_tagger; signal; wjets; qcd_abcd
channel=""        # wjets -> {1j1l, 1l0b};  ztoll-> {ll, ll_ISR}; qcd_abcd -> {1l0b; 1l0b_A; 1l0b_B; 1l0b_C; 1l0b_D}; ttbar -> {2b1l, 1b1e1mu, 1b1l}


lepton_flavor="mu"
year="2017" # 2016APV; 2016; 2017; 2018
nfiles="-1"
executor="futures"
output_type="array" # hist/array
nsample="1" # Importante: Dejar nsample="" si no se quiere un nsample especifico, en caso de querer uno especifico nsample="3"
output_folder="2017_top_tagger" 


samples=(
 "TTToSemiLeptonic"
  "TTTo2L2Nu"   
  "TTToHadronic"
  "DYJetsToLL_M-10to50"
  "DYJetsToLL_M-50_HT-70to100"
  "DYJetsToLL_M-50_HT-100to200"
  "DYJetsToLL_M-50_ext"
  "DYJetsToLL_M-50_HT-200to400"
  "DYJetsToLL_M-50_HT-400to600"
  "DYJetsToLL_M-50_HT-600to800"
  "DYJetsToLL_M-50_HT-800to1200"
  "DYJetsToLL_M-50_inclusive"
  "DYJetsToLL_M-50_HT-1200to2500"
  "DYJetsToLL_M-50_HT-2500toInf"
# # #  "SingleMuon"  
   "MET" 
#    "Tau"                       
# # # #    "SingleElectron"
  "ST_s-channel_4f_leptonDecays"
  "ST_t-channel_antitop_5f_InclusiveDecays"
  "ST_t-channel_top_5f_InclusiveDecays"
  "ST_tW_antitop_5f_inclusiveDecays"
  "ST_tW_top_5f_inclusiveDecays"
  "WJetsToLNu_HT-70To100"
  "WJetsToLNu_HT-100To200"
  "WJetsToLNu_HT-200To400"
  "WJetsToLNu_inclusive"
  "WJetsToLNu_HT-400To600"
  "WJetsToLNu_HT-600To800"
  "WJetsToLNu_ext"  
  "WJetsToLNu_HT-800To1200"
  "WJetsToLNu_HT-1200To2500"
  "WJetsToLNu_HT-2500ToInf"
  "WW"
  "WZ"
 "ZZ"                             
#  "SignalTau_600GeV"
#    "DYJetsToLL_M-50_CH3"
)



if [ $processor == "ttbar" ] || [ $processor == "wjets" ] || [ $processor == "ztoll" ] || [ $processor == "qcd_abcd" ]; then
    for sample in "${samples[@]}"; do 
      python3 submit_lxplus.py --processor "$processor" --channel "$channel" --lepton_flavor "$lepton_flavor" --sample "$sample" --year "$year" --nfiles "$nfiles" --executor "$executor" --output_type "$output_type" --nsample "$nsample"
      sleep 90 #  Wait for 90 seconds before sending the next sample
    done

elif [ $processor == "top_tagger" ] || [ $processor == "signal" ]; then
    for sample in "${samples[@]}"; do 
      python3 submit_lxplus.py --processor "$processor" --lepton_flavor "$lepton_flavor" --sample "$sample" --year "$year" --nfiles "$nfiles" --executor "$executor" --output_type "$output_type" --nsample "$nsample"
      sleep 90 #  Wait for 90 seconds before sending the next sample
    done
fi
    


echo "########################################################################################################################"
echo "###  The jobs have been sent, don't forget to run lxplus_test.sh file once they finish to complete the missing ones  ###"
echo "########################################################################################################################"
