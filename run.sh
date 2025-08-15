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
#echo $GRID_PASSWORD | voms-proxy-init --voms cms


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
processor="ztoll"     # top_tagger; signal; qcd_hadronic; wplusjets; ztoll:  wjets; btag_eff
channel="ll"           # top_tagger -> {}; signal -> {}; qcd_hadronic -> {cr_b, cr_c, cr_d}; wplusjets -> {wjets, cr_b, cr_c, cr_d}, wjets -> {1j1l*, 1l0b}; ztoll-> {ll, ll+c}


lepton_flavor="mu"
year="2017" # 2016APV; 2016; 2017; 2018
nfiles="-1"
executor="futures"
output_type="array" # hist/array
nsample="" # Importante: Dejar nsample="" si no se quiere un nsample especifico, en caso de querer uno especifico nsample="3"
output_folder="/eos/user/j/jreyesve/WINDOWS/Desktop/final/$processor/$year"
run_systematics="true" # Cambiar a "true" para activar sistemáticos a nivel de objeto (solo cuando sea necesario, ya que puede aumentar el tiempo de ejecución considerablemente)



samples=(
    "TTToSemiLeptonic"
    "TTTo2L2Nu"
    "TTToHadronic"
    "DYJetsToLL_nlo_M-10to50"
    "DYJetsToLL_nlo_M-50"
    "SingleMuon"
#    "MET"
#   "Tau"
#   "SingleElectron"
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
    "QCD_HT50to100"
    "QCD_HT100to200"
    "QCD_HT200to300"
    "QCD_HT300to500"
    "QCD_HT500to700"
    "QCD_HT700to1000"
    "QCD_HT1000to1500"
    "QCD_HT1500to2000"
    "QCD_HT2000toInf"
    "GluGluHToWWToLNuQQ" 
    "VBFHToWWTo2L2Nu"
    "VBFHToWWToLNuQQ"
#   "SignalTau_600GeV"
#   "SignalTau_1TeV"
#   "SignalTau_2TeV"
#   "SignalTau_3TeV"
#  "DYJetsToLL_M-50_CH3"
#  "DYJetsToLL_M-10to50"
#  "DYJetsToLL_M-50_HT-70to100"
#  "DYJetsToLL_M-50_HT-100to200"
#  "DYJetsToLL_M-50_ext"
#  "DYJetsToLL_M-50_HT-200to400"
#  "DYJetsToLL_M-50_HT-400to600"
#  "DYJetsToLL_M-50_HT-600to800"
#  "DYJetsToLL_M-50_HT-800to1200"
#  "DYJetsToLL_M-50_inclusive"
#  "DYJetsToLL_M-50_HT-1200to2500"
#  "DYJetsToLL_M-50_HT-2500toInf"
)



# Ejecutar build_filesets una sola vez antes del bucle
python3 -c "
from utils import update_nsplit, build_filesets;



print('::::: Ejecutando update_nsplit para el año $year :::::');
update_nsplit('$year');

print('::::: Creando particiones de las muestras con build_filesets() para $year ::::::');

args = {
    'processor': '$processor',
    'lepton_flavor': '$lepton_flavor',
    'year': '$year',
    'run_systematics': '$run_systematics',
    'sample': 'TTToSemiLeptonic',                     # Este es un ejemplo, se debe revisar en el caso de región de señal
    'facility': 'lxplus'
};
build_filesets(args);
"



if [ $processor == "ttbar" ] || [ $processor == "wjets" ] || [ $processor == "ztoll" ] || [ $processor == "qcd_abcd" ] || [ $processor == "wplusjets" ] || [ $processor == "qcd_hadronic" ]; then
    # Definir la ruta donde se creará mover_archivos.sh
    dir_to_create="wprime_plus_b/outs/$processor/$channel/$lepton_flavor/$year"

    for sample in "${samples[@]}"; do
      python3 submit_lxplus.py --processor "$processor" --channel "$channel" --lepton_flavor "$lepton_flavor" --sample "$sample" --year "$year" --nfiles "$nfiles" --executor "$executor" --output_type "$output_type" --nsample "$nsample" --run_systematics "$run_systematics"  --output_folder "$output_folder"
      sleep 60 #  Wait for 60 seconds before sending the next sample  
    done

elif [ $processor == "top_tagger" ] || [ $processor == "signal" ] || [ $processor == "btag_eff" ]; then
    
    # Definir la ruta donde se creará mover_archivos.sh
    dir_to_create="wprime_plus_b/outs/$processor/$lepton_flavor/$year"


    for sample in "${samples[@]}"; do
      python3 submit_lxplus.py --processor "$processor" --lepton_flavor "$lepton_flavor" --sample "$sample" --year "$year" --nfiles "$nfiles" --executor "$executor" --output_type "$output_type" --nsample "$nsample" --run_systematics "$run_systematics"  --output_folder "$output_folder"
      sleep 60 #  Wait for 60 seconds before sending the next sample
    done
fi


echo $ANALYSIS_PATH/$output_folder

# sleep 600
# ./run_v1.sh


echo "###### Year: $year; processor $processor is over, now waiting for the jobs to finish #####"

echo "#######################################################################################################################################"
echo "##################################### When all jobs have been sent ####################################################################"
echo "### Use find_XRoot_sites_with_error.sh to find the sites with errors in the condor logs, and comment them in make_fileset_lxplus.py ###"
echo "##################### Use checkfiles.sh file to sent the complete list of the missing jobs  ###########################################"
echo "#######################################################################################################################################"

