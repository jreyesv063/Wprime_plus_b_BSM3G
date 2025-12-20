#!/bin/bash

# Before running this script, make sure to grant execution permissions:
# chmod +x run.sh
# Run the script with:
# ./run.sh


echo "########################################"
echo "######  Starting the analysis code #####"
echo "########################################"


# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Initialize GRID certificate (CMS VOMS proxy)
echo $GRID_PASSWORD | voms-proxy-init --voms cms --pwstdin


# Move to the fileset directory
cd wprime_plus_b/fileset/

# Enter the Singularity shell with required bindings
singularity shell -B /afs -B /eos -B /cvmfs \
/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest-py3.10 << EOF

# Run the fileset creation script inside Singularity (if needed)
# python make_fileset_lxplus.py

# Exit the Singularity shell
exit
EOF


# Return to the directory where the script is located
cd "$SCRIPT_DIR"

# =========================
# Analysis configuration
# =========================

# Define processor and channel
processor="wplusjets"     # Options: top_tagger; signal; qcd_hadronic; qcd_hadronic_closure; wplusjets; ztoll; wjets; btag_eff; ctag_eff; zplusc
channel="wjets"           # Channel depends on processor:
                          # top_tagger -> {}
                          # signal -> {}
                          # qcd_hadronic -> {cr_b, cr_c, cr_d}
                          # qcd_hadronic_closure -> {signal_prime, cr_b, cr_c, cr_d}
                          # wplusjets -> {wjets, cr_b, cr_c, cr_d}
                          # wjets -> {1j1l*, 1l0b}
                          # ztoll -> {ll}
                          # zplusc -> {ll+c}

# Lepton flavor
lepton_flavor="tau"

# Data-taking year
year="2017"                  # Options: 2016APV; 2016; 2017; 2018

# Number of files to process (-1 means all files)
nfiles="1"

# Executor type
executor="futures"

# Output type: histogram or array
output_type="array"          # hist / array

# Sample index (leave empty to run all samples)
nsample="1"                  # IMPORTANT: Leave nsample="" unless a specific sample index is needed (e.g. nsample="3")

# Output directory
output_folder="/eos/user/j/jreyesve/WINDOWS/Desktop/2025/Diciembre/$processor/$channel/$year/"

# Enable object-level systematics (can significantly increase runtime)
run_systematics="true"       # Set to "true" only when needed

# Control whether new filesets should be created
create_new_filesets="true"  # Set to "true" to regenerate filesets
                             # (use only if servers were modified in make_fileset_lxplus.py)

# =========================
# Sample list
# =========================

samples=(
    # "TTToSemiLeptonic"
    # "TTTo2L2Nu"
    "TTToHadronic"
    # "DYJetsToLL_nlo_M-10to50"
    # "DYJetsToLL_nlo_M-50"
    # "SingleMuon"
    #"MET"
    # "Tau"
    # "SingleElectron"
    # "ST_s-channel_4f_leptonDecays"
    # "ST_t-channel_antitop_5f_InclusiveDecays"
    # "ST_t-channel_top_5f_InclusiveDecays"
    # "ST_tW_antitop_5f_inclusiveDecays"
    # "ST_tW_top_5f_inclusiveDecays"
    # "WJetsToLNu_HT-70To100"
    # "WJetsToLNu_HT-100To200"
    # "WJetsToLNu_HT-200To400"
    # "WJetsToLNu_inclusive"
    # "WJetsToLNu_HT-400To600"
    # "WJetsToLNu_HT-600To800"
    # "WJetsToLNu_ext"
    # "WJetsToLNu_HT-800To1200"
    # "WJetsToLNu_HT-1200To2500"
    # "WJetsToLNu_HT-2500ToInf"
    # "WW"
    # "WZ"
    # "ZZ"
    # "QCD_HT50to100"
    # "QCD_HT100to200"
    # "QCD_HT200to300"
    # "QCD_HT300to500"
    # "QCD_HT500to700"
    # "QCD_HT700to1000"
    # "QCD_HT1000to1500"
    # "QCD_HT1500to2000"
    # "QCD_HT2000toInf"
    # "GluGluHToWWToLNuQQ"
    # "VBFHToWWTo2L2Nu"
    # "VBFHToWWToLNuQQ"
    # "SignalTau_300GeV"
    # "SignalTau_400GeV"
    # "SignalTau_600GeV"
    # "SignalTau_750GeV"
    # "SignalTau_1000GeV"
    # "SignalTau_1500GeV"
    # "SignalTau_2000GeV"
    # "SignalTau_3000GeV"
)

# =========================
# Fileset creation (optional)
# =========================

# Run build_filesets only if create_new_filesets is set to true
if [ "$create_new_filesets" = "true" ]; then
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
    'facility': 'lxplus'
};
build_filesets(args);
"
else
    echo "create_new_filesets=false → skipping fileset creation"
fi

# =========================
# Job submission
# =========================

if [ "$processor" == "ttbar" ] || \
   [ "$processor" == "wjets" ] || \
   [ "$processor" == "ztoll" ] || \
   [ "$processor" == "zplusc" ] || \
   [ "$processor" == "qcd_abcd" ] || \
   [ "$processor" == "wplusjets" ] || \
   [ "$processor" == "qcd_hadronic" ] || \
   [ "$processor" == "qcd_hadronic_closure" ]; then

    # Output directory for this processor/channel/lepton/year
    dir_to_create="wprime_plus_b/outs/$processor/$channel/$lepton_flavor/$year"

    for sample in "${samples[@]}"; do
        python3 submit_lxplus.py \
            --processor "$processor" \
            --channel "$channel" \
            --lepton_flavor "$lepton_flavor" \
            --sample "$sample" \
            --year "$year" \
            --nfiles "$nfiles" \
            --executor "$executor" \
            --output_type "$output_type" \
            --nsample "$nsample" \
            --run_systematics "$run_systematics" \
            --output_folder "$output_folder"
        sleep 60  # Wait 60 seconds before submitting the next sample (optional)
    done

elif [ "$processor" == "top_tagger" ] || \
     [ "$processor" == "signal" ] || \
     [ "$processor" == "btag_eff" ] || \
     [ "$processor" == "ctag_eff" ]; then

    # Output directory for this processor/lepton/year
    dir_to_create="wprime_plus_b/outs/$processor/$lepton_flavor/$year"

    for sample in "${samples[@]}"; do
        python3 submit_lxplus.py \
            --processor "$processor" \
            --lepton_flavor "$lepton_flavor" \
            --sample "$sample" \
            --year "$year" \
            --nfiles "$nfiles" \
            --executor "$executor" \
            --output_type "$output_type" \
            --nsample "$nsample" \
            --run_systematics "$run_systematics" \
            --output_folder "$output_folder"
        sleep 60  # Wait 60 seconds before submitting the next sample (optional)
    done
fi


# Print final output path
echo "$ANALYSIS_PATH/$output_folder"



echo "###### Year: $year; processor $processor is over, now waiting for the jobs to finish #####"

echo "#######################################################################################################################################"
echo "##################################### When all jobs have been sent ####################################################################"
echo "### Use find_XRoot_sites_with_error.sh to find the sites with errors in the condor logs, and comment them in make_fileset_lxplus.py ###"
echo "##################### Use checkfiles.sh file to sent the complete list of the missing jobs  ###########################################"
echo "#######################################################################################################################################"

