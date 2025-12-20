#!/bin/bash

# Before running this script, make sure to grant execution permissions:
# chmod +x checkfiles.sh
# Run the script with:
# ./checkfiles.sh

echo "########################################"
echo "###########  Checking files ############"
echo "########################################"


###################################
##### Variables to be modified ####
###################################

# Flag to decide whether to create the fileset
# IMPORTANT: set this to true if servers were commented out
create_fileset=true   # Set to false if you do not want to recreate the fileset

# Directory where this bash script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Dataset selection configuration
consider_MET=true
consider_SingleMuon=false
consider_SingleElectron=false
consider_Tau=false

consider_higgs=true
consider_wj=true
consider_inclusive_wj=true
consider_inclusive_ext_wj=true
consider_inclusive_dy_nlo=true
consider_tt=true
consider_st=true
consider_vv=true
consider_qcd=false

consider_signal_tau=false
consider_signal_ele=false
consider_signal_mu=false

consider_inclusive_dy=false
consider_inclusive_ext_dy=false
consider_inclusive_ch3=false
consider_dy=false

# Path to the run.sh file
run_file="run.sh"


###################################
###################################


# Extract variables from run.sh
processor=$(grep -o 'processor=".*"' "$run_file" | cut -d'"' -f2)
channel=$(grep -o 'channel=".*"' "$run_file" | cut -d'"' -f2)
lepton_flavor=$(grep -o 'lepton_flavor=".*"' "$run_file" | cut -d'"' -f2)
year=$(grep -o 'year=".*"' "$run_file" | cut -d'"' -f2)
nfiles=$(grep -o 'nfiles=".*"' "$run_file" | cut -d'"' -f2)
executor=$(grep -o 'executor=".*"' "$run_file" | cut -d'"' -f2)
output_type=$(grep -o 'output_type=".*"' "$run_file" | cut -d'"' -f2)
run_systematics=$(grep -o 'run_systematics=".*"' "$run_file" | cut -d'"' -f2)

# Resolve environment variables inside output_folder
output_folder_raw=$(grep -o 'output_folder=".*"' "$run_file" | cut -d'"' -f2)
output_folder=$(eval echo "$output_folder_raw")


# Select the YAML configuration file
# Update nsplit only if systematics are enabled
if [ "$run_systematics" == "true" ]; then
    echo "Updating nsplit for year $year"
    python3 -c "from utils import update_nsplit; update_nsplit('$year')"
    yaml_file="datasets_configs_systematics.yaml"
else
    yaml_file="datasets_configs.yaml"
fi


# Move to the dataset configuration directory
cd wprime_plus_b/configs/dataset


# Create an empty associative array
declare -A mapa


# Read the YAML file and fill the map
while IFS= read -r nombre_archivo && IFS= read -r divisiones; do

    # Extract dataset name and number of splits
    nombre_archivo=$(echo "$nombre_archivo" | sed 's/:$//')   # Remove trailing colon
    divisiones=$(echo "$divisiones" | awk '{print $2}')
    mapa["$nombre_archivo"]=$divisiones

done < "$yaml_file"


# ================================
# Filter datasets based on flags
# ================================

# Data samples
if ! $consider_SingleElectron; then unset mapa["SingleElectron"]; fi
if ! $consider_Tau;           then unset mapa["Tau"];           fi
if ! $consider_MET;           then unset mapa["MET"];           fi
if ! $consider_SingleMuon;    then unset mapa["SingleMuon"];    fi

# Higgs samples
if ! $consider_higgs; then
    unset mapa["VBFHToWWTo2L2Nu"]
    unset mapa["VBFHToWWToLNuQQ"]
    unset mapa["GluGluHToWWToLNuQQ"]
fi

# DY HT-binned samples
if ! $consider_dy; then
    unset mapa["DYJetsToLL_M-50_HT-70to100"]
    unset mapa["DYJetsToLL_M-50_HT-100to200"]
    unset mapa["DYJetsToLL_M-50_HT-200to400"]
    unset mapa["DYJetsToLL_M-50_HT-400to600"]
    unset mapa["DYJetsToLL_M-50_HT-600to800"]
    unset mapa["DYJetsToLL_M-50_HT-800to1200"]
    unset mapa["DYJetsToLL_M-50_HT-1200to2500"]
    unset mapa["DYJetsToLL_M-50_HT-2500toInf"]
fi

# Single top samples
if ! $consider_st; then
    unset mapa["ST_s-channel_4f_leptonDecays"]
    unset mapa["ST_t-channel_antitop_5f_InclusiveDecays"]
    unset mapa["ST_t-channel_top_5f_InclusiveDecays"]
    unset mapa["ST_tW_antitop_5f_inclusiveDecays"]
    unset mapa["ST_tW_top_5f_inclusiveDecays"]
fi

# Diboson samples
if ! $consider_vv; then
    unset mapa["WW"]
    unset mapa["WZ"]
    unset mapa["ZZ"]
fi

# W+jets samples
if ! $consider_wj; then
    unset mapa["WJetsToLNu_HT-70To100"]
    unset mapa["WJetsToLNu_HT-100To200"]
    unset mapa["WJetsToLNu_HT-200To400"]
    unset mapa["WJetsToLNu_HT-400To600"]
    unset mapa["WJetsToLNu_HT-600To800"]
    unset mapa["WJetsToLNu_HT-800To1200"]
    unset mapa["WJetsToLNu_HT-1200To2500"]
    unset mapa["WJetsToLNu_HT-2500ToInf"]
fi

# ttbar samples
if ! $consider_tt; then
    unset mapa["TTToSemiLeptonic"]
    unset mapa["TTTo2L2Nu"]
    unset mapa["TTToHadronic"]
fi

# Signal samples
if ! $consider_signal_tau; then
    unset mapa["SignalTau_300GeV"]
    unset mapa["SignalTau_400GeV"]
    unset mapa["SignalTau_600GeV"]
    unset mapa["SignalTau_750GeV"]
    unset mapa["SignalTau_1000GeV"]
    unset mapa["SignalTau_1500GeV"]
    unset mapa["SignalTau_2000GeV"]
    unset mapa["SignalTau_3000GeV"]
fi

if ! $consider_signal_ele; then
    unset mapa["SignalElectron_600GeV"]
    unset mapa["SignalElectron_1TeV"]
    unset mapa["SignalElectron_2TeV"]
fi

if ! $consider_signal_mu; then
    unset mapa["SignalMuon_600GeV"]
    unset mapa["SignalMuon_1TeV"]
    unset mapa["SignalMuon_2TeV"]
fi

# Inclusive DY samples
if ! $consider_inclusive_dy; then
    unset mapa["DYJetsToLL_M-50_inclusive"]
    unset mapa["DYJetsToLL_M-10to50"]
fi

# Inclusive W+jets samples
if ! $consider_inclusive_wj; then
    unset mapa["WJetsToLNu_inclusive"]
fi

# Extended inclusive samples
if ! $consider_inclusive_ext_dy; then unset mapa["DYJetsToLL_M-50_ext"]; fi
if ! $consider_inclusive_ext_wj; then unset mapa["WJetsToLNu_ext"];     fi
if ! $consider_inclusive_ch3;    then unset mapa["DYJetsToLL_M-50_CH3"]; fi

# NLO DY samples
if ! $consider_inclusive_dy_nlo; then
    unset mapa["DYJetsToLL_nlo_M-10to50"]
    unset mapa["DYJetsToLL_nlo_M-50"]
fi

# QCD samples
if ! $consider_qcd; then
    unset mapa["QCD_HT50to100"]
    unset mapa["QCD_HT100to200"]
    unset mapa["QCD_HT200to300"]
    unset mapa["QCD_HT300to500"]
    unset mapa["QCD_HT500to700"]
    unset mapa["QCD_HT700to1000"]
    unset mapa["QCD_HT1000to1500"]
    unset mapa["QCD_HT1500to2000"]
    unset mapa["QCD_HT2000toInf"]
fi


# ================================
# Check missing metadata files
# ================================

missing_files=()

for base_name in "${!mapa[@]}"; do
    n_splits="${mapa[$base_name]}"

    cd "$output_folder/metadata"

    if [ "$n_splits" -eq 1 ]; then
        file="${base_name}_metadata.json"
        [ ! -f "$file" ] && missing_files+=("$file")
    else
        for (( i=1; i<=n_splits; i++ )); do
            file="${base_name}_${i}_metadata.json"
            [ ! -f "$file" ] && missing_files+=("$file")
        done
    fi
done


# Uncomment this block to print the list of missing files
#: '
if [ ${#missing_files[@]} -gt 0 ]; then
    printf '%s\n' "${missing_files[@]}"
fi
#'


cd "$SCRIPT_DIR/wprime_plus_b/fileset/"

#######################################################
### Preparing the reprocessing of missing files     ###
#######################################################

# Activate GRID proxy
echo $GRID_PASSWORD | voms-proxy-init --voms cms --pwstdin

# Enter Singularity shell
singularity shell -B /afs -B /eos -B /cvmfs \
/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest-py3.10 << EOF

if [ "$create_fileset" = "true" ]; then
    echo "Running make_fileset_lxplus.py..."
    python make_fileset_lxplus.py --year "$year"
else
    echo "make_fileset_lxplus.py is disabled"
fi

# Exit Singularity shell
exit
EOF


# Return to the main directory
cd "$SCRIPT_DIR"


# Build filesets once before resubmitting jobs
python3 -c "
from utils import update_nsplit, build_filesets;

print('::::: Running update_nsplit for year $year :::::');
update_nsplit('$year');

print('::::: Creating sample partitions with build_filesets() for $year ::::::');

args = {
    'processor': '$processor',
    'lepton_flavor': '$lepton_flavor',
    'year': '$year',
    'run_systematics': '$run_systematics',
    'sample': 'TTToSemiLeptonic',  # Example sample; review for signal regions
    'facility': 'lxplus'
};
build_filesets(args);
"


# =====================================
# Resubmit jobs for missing files
# =====================================

for missing_file in "${missing_files[@]}"; do

    base="${missing_file%_metadata.json}"

    if [[ "$base" =~ _[0-9]+$ ]]; then
        nsample=$(echo "$base" | awk -F'_' '{print $NF}')
        sample_name="${base%_*}"
    else
        nsample=""
        sample_name="$base"
    fi

    extra_arg=""
    [ -n "$nsample" ] && extra_arg="--nsample $nsample"

    if [ "$processor" == "ttbar" ] || \
       [ "$processor" == "wjets" ] || \
       [ "$processor" == "ztoll" ] || \
       [ "$processor" == "zplusc" ] || \
       [ "$processor" == "qcd_abcd" ] || \
       [ "$processor" == "wplusjets" ] || \
       [ "$processor" == "qcd_hadronic" ] || \
       [ "$processor" == "qcd_hadronic_closure" ]; then

        command="python3 submit_lxplus.py \
            --processor $processor \
            --channel $channel \
            --lepton_flavor $lepton_flavor \
            --sample $sample_name \
            --year $year \
            --nfiles $nfiles \
            --executor $executor \
            --output_type $output_type \
            $extra_arg \
            --run_systematics $run_systematics \
            --output_folder $output_folder"

    elif [ "$processor" == "top_tagger" ] || \
         [ "$processor" == "signal" ] || \
         [ "$processor" == "btag_eff" ] || \
         [ "$processor" == "ctag_eff" ]; then

        command="python3 submit_lxplus.py \
            --processor $processor \
            --lepton_flavor $lepton_flavor \
            --sample $sample_name \
            --year $year \
            --nfiles $nfiles \
            --executor $executor \
            --output_type $output_type \
            $extra_arg \
            --run_systematics $run_systematics \
            --output_folder $output_folder"
    fi

    cd "$SCRIPT_DIR"
    eval "$command"
done


# ================================
# Final summary
# ================================

num_missing_files=${#missing_files[@]}

echo "########################################"
echo "############  Results  #################"
echo "Number of missing files: $num_missing_files"
echo "########################################"

