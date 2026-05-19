#!/bin/bash

# =========================
# Global Configuration
# =========================
processor="signal"     # Options: top_tagger; signal; qcd_hadronic_closure; wplusjets; wjets; ztoll; zplusc  # ; btag_eff; ctag_eff;
lepton_flavor="tau"
run_era="run_2"  # Options: run_2, run_3
BASE_OUTPUT_DIR="/eos/user/j/jreyesve/WINDOWS/Desktop/2026/Mayo/CRs_shapes"
unblinded="false" # Only affects signal processor; set to "true" to include data in the signal region (use with caution)
qcd_cr_B_TF_estimation="true" # Set to "true" to run the QCD shape estimation (only for processor with ABCD methodology)
global_redirector="true" # Set to "true" to use the global redirector


# Define years based on the run_era
if [ "$run_era" == "run_2" ]; then
    years=("2016APV" "2016" "2017" "2018")
elif [ "$run_era" == "run_3" ]; then
    years=("2022pre" "2022post" "2023pre" "2023post" "2024")
else
    echo "[ERROR] Unknown run_era: $run_era"
    exit 1
fi


# Main loop over years
for year in "${years[@]}"; do
    echo "###################################################"
    echo "### Starting Analysis for Year: $year ($run_era) ###"
    echo "###################################################"

    # Get the directory where this script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    # Initialize GRID certificate
    echo "$GRID_PASSWORD" | voms-proxy-init --voms cms --pwstdin

    # --- Fileset logic ---
    cd "$SCRIPT_DIR/wprime_plus_b/fileset" || exit 1
    shopt -s nullglob
    files=(fileset_*.json)
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then
        echo "[INFO] No fileset JSON found. Creating fileset..."
        singularity shell -B /afs -B /eos -B /cvmfs \
        /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.30-py3.10 << EOF
PYTHONNOUSERSITE=1 python3 make_fileset_lxplus.py
exit
EOF
    fi

    cd "$SCRIPT_DIR"

    # =========================
    # Standard configuration
    # =========================
    nfiles="-1"
    executor="futures"
    output_type="array"
    nsample=""
    run_systematics="true"
    create_new_filesets="true"
    qcd_data_driven="true"

    # Update output folder dynamically for each year
    output_folder="${BASE_OUTPUT_DIR}/${processor}/${lepton_flavor}/${year}"

    # =========================
    # Load Samples from YAML
    # =========================
    yaml_file="wprime_plus_b/selection_criteria/$processor/samples_${lepton_flavor}.yaml"

    if [ -f "$yaml_file" ]; then
        samples_raw=$(python3 -c "
import yaml
try:
    with open('$yaml_file', 'r') as f:
        data = yaml.safe_load(f)
        era = '$run_era'
        if era in data and data[era]:
            print(' '.join(data[era]))
        else:
            print('PYTHON_ERROR: Era not found in YAML')
except Exception as e:
    print(f'PYTHON_ERROR: {e}')
")
        if [[ "$samples_raw" == *"PYTHON_ERROR"* ]]; then
            echo "[ERROR] $samples_raw"; exit 1
        fi
        samples=($samples_raw)
    else
        echo "[ERROR] YAML file not found: $yaml_file"; exit 1
    fi

    # =========================
    # Fileset creation
    # =========================
    if [ "$create_new_filesets" = "true" ]; then
        python3 -c "
from utils import update_nsplit, build_filesets
update_nsplit('$year')
args = {'processor': '$processor', 'lepton_flavor': '$lepton_flavor', 'year': '$year', 'run_systematics': '$run_systematics', 'facility': 'lxplus'}
build_filesets(args)
"
    fi

    # =========================
    # Job submission
    # =========================
    # Assuming standard submission for your processors
    for sample in "${samples[@]}"; do
        echo "[SUBMITTING] Sample: $sample Year: $year"
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
            --qcd_data_driven "$qcd_data_driven" \
            --unblinded "$unblinded" \
            --global_redirector "$global_redirector" \
            --output_folder "$output_folder" \
            --qcd_cr_B_TF_estimation "$qcd_cr_B_TF_estimation"
        
        sleep 60 # Wait 60 seconds before submitting the next sample
    done

    echo "[DONE] Year $year submitted. Output: $output_folder"

    # =========================
    # 15-minute Wait Logic
    # =========================
    # Check if this is the last year in the array to avoid waiting at the very end
    if [ "$year" != "${years[-1]}" ]; then
        echo "[WAIT] All samples for $year submitted. Waiting 15 minutes before next year..."
        sleep 900 # 15 minutes = 900 seconds
    fi

done

echo "################################################"
echo "### All years processed successfully!        ###"
echo "################################################"