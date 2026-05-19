#!/bin/bash

# Before running this script, be sure to grant execution permissions with the following command:
# chmod +x checkfiles.sh
# Script is used with the command ./checkfiles.sh

echo "########################################"
echo "########### Checking files  ############"
echo "########################################"


###################################
######  Variables to modify  ######
###################################
# Indicate whether you want to create the fileset or not: Important to set it to true if servers were commented out.
create_fileset=false #If you want to create the fileset set to true. If you don't want to create it, set it to false
create_partitions=true # If you want to create the dataset partitions set to true. If you don't want to create them, set it to false.

# Bash file directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Dataset to be used
considerar_MET=true
considerar_SingleMuon=false
considerar_SingleElectron=false
considerar_Tau=false


considerar_higgs=true
considerar_wj=true
considerar_inclusive_wj=true
considerar_inclusive_ext_wj=true
considerar_inclusive_dy_nlo=true
considerar_tt=true
considerar_st=true
considerar_vv=true
considerar_qcd=false


considerar_signal_tau=false
considerar_signal_ele=false
considerar_signal_mu=false


considerar_inclusive_dy=false
considerar_inclusive_ext_dy=false
considerar_inclusive_ch3=false
considerar_dy=false

# Path to the run.sh file you want to check
archivo_run="run.sh"


#####################################
#####################################
# Extract the variables from the run.sh file
processor=$(grep -o 'processor=".*"' "$archivo_run" | cut -d'"' -f2)
lepton_flavor=$(grep -o 'lepton_flavor=".*"' "$archivo_run" | cut -d'"' -f2)
year=$(grep -o 'year=".*"' "$archivo_run" | cut -d'"' -f2)
executor=$(grep -o 'executor=".*"' "$archivo_run" | cut -d'"' -f2)
output_type=$(grep -o 'output_type=".*"' "$archivo_run" | cut -d'"' -f2)
run_systematics=$(grep -o 'run_systematics=".*"' "$archivo_run" | cut -d'"' -f2)
qcd_data_driven=$(grep -o 'qcd_data_driven=".*"' "$archivo_run" | cut -d'"' -f2)
unblinded=$(grep -o 'unblinded=".*"' "$archivo_run" | cut -d'"' -f2)
global_redirector=$(grep -o 'global_redirector=".*"' "$archivo_run" | cut -d'"' -f2)

output_folder_raw=$(grep -o 'output_folder=".*"' "$archivo_run" | cut -d'"' -f2)
output_folder=$(eval echo "$output_folder_raw")


# Select the YAML file and update it only if run_systematics is true
if [ "$run_systematics" == "true" ]; then
    # Call the Python function only if systematic errors occur
    echo "Updating nsplit for year $year"
    python3 -c "from utils import update_nsplit; update_nsplit('$year')"
    yaml_file="datasets_configs_systematics.yaml"
else
    yaml_file="datasets_configs.yaml"
fi


# Change to the directory where the file is located
cd wprime_plus_b/configs/dataset


# Create an empty map
declare -A mapa


# Read the file
while IFS= read -r nombre_archivo && IFS= read -r divisiones; do

    # Extract the file name and number of divisions
    nombre_archivo=$(echo "$nombre_archivo" | sed 's/:$//')  # Remove the colon at the end of the file name
    divisiones=$(echo "$divisiones" | awk '{print $2}')
    mapa["$nombre_archivo"]=$divisiones


done < "$yaml_file"


# Identify only the appropriate data sets
if ! $considerar_SingleElectron; then
    unset mapa["SingleElectron"]
fi

if ! $considerar_Tau; then
    unset mapa["Tau"]
fi

if ! $considerar_MET; then
    unset mapa["MET"]
fi

if ! $considerar_SingleMuon; then
    unset mapa["SingleMuon"]
fi

if ! $considerar_higgs; then
    unset mapa["VBFHToWWTo2L2Nu"]
    unset mapa["VBFHToWWToLNuQQ"]
    unset mapa["GluGluHToWWToLNuQQ"]
fi
if ! $considerar_dy; then
    unset mapa["DYJetsToLL_M-50_HT-70to100"]
    unset mapa["DYJetsToLL_M-50_HT-100to200"]
    unset mapa["DYJetsToLL_M-50_HT-200to400"]
    unset mapa["DYJetsToLL_M-50_HT-400to600"]
    unset mapa["DYJetsToLL_M-50_HT-600to800"]
    unset mapa["DYJetsToLL_M-50_HT-800to1200"]
    unset mapa["DYJetsToLL_M-50_HT-1200to2500"]
    unset mapa["DYJetsToLL_M-50_HT-2500toInf"]
fi
if ! $considerar_st; then
    unset mapa["ST_s-channel_4f_leptonDecays"]
    unset mapa["ST_t-channel_antitop_5f_InclusiveDecays"]
    unset mapa["ST_t-channel_top_5f_InclusiveDecays"]
    unset mapa["ST_tW_antitop_5f_inclusiveDecays"]
    unset mapa["ST_tW_top_5f_inclusiveDecays"]
fi
if ! $considerar_vv; then
    unset mapa["WW"]
    unset mapa["WZ"]
    unset mapa["ZZ"]
fi
if ! $considerar_wj; then
    unset mapa["WJetsToLNu_HT-70To100"]
    unset mapa["WJetsToLNu_HT-100To200"]
    unset mapa["WJetsToLNu_HT-200To400"]
    unset mapa["WJetsToLNu_HT-400To600"]
    unset mapa["WJetsToLNu_HT-600To800"]
    unset mapa["WJetsToLNu_HT-800To1200"]
    unset mapa["WJetsToLNu_HT-1200To2500"]
    unset mapa["WJetsToLNu_HT-2500ToInf"]
fi
if ! $considerar_tt; then
    unset mapa["TTToSemiLeptonic"]
    unset mapa["TTTo2L2Nu"]
    unset mapa["TTToHadronic"]
fi
if ! $considerar_signal_tau; then
    unset mapa["SignalTau_300GeV"]
    unset mapa["SignalTau_400GeV"]
    unset mapa["SignalTau_600GeV"]
    unset mapa["SignalTau_750GeV"]
    unset mapa["SignalTau_1000GeV"]
    unset mapa["SignalTau_1500GeV"]
    unset mapa["SignalTau_2000GeV"]
    unset mapa["SignalTau_3000GeV"]
fi
if ! $considerar_signal_ele; then
    unset mapa["SignalElectron_1TeV"]
    unset mapa["SignalElectron_2TeV"]
    unset mapa["SignalElectron_600GeV"]
fi
if ! $considerar_signal_mu; then
    unset mapa["SignalMuon_1TeV"]
    unset mapa["SignalMuon_2TeV"]
    unset mapa["SignalMuon_600GeV"]
fi
if ! $considerar_inclusive_dy; then
    unset mapa["DYJetsToLL_M-50_inclusive"]
    unset mapa["DYJetsToLL_M-10to50"]
fi
if ! $considerar_inclusive_wj; then
    unset mapa["WJetsToLNu_inclusive"]
fi
if ! $considerar_inclusive_ext_dy; then
    unset mapa["DYJetsToLL_M-50_ext"]
fi
if ! $considerar_inclusive_ext_wj; then
    unset mapa["WJetsToLNu_ext"]
fi
if ! $considerar_inclusive_ch3; then
    unset mapa["DYJetsToLL_M-50_CH3"]
fi
if ! $considerar_inclusive_dy_nlo; then
    unset mapa["DYJetsToLL_nlo_M-10to50"]
    unset mapa["DYJetsToLL_nlo_M-50"]
fi
if ! $considerar_qcd; then
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


# Lista de archivos no encontrados
archivos_faltantes=()

# Iterar sobre cada par clave-valor en el mapa
for nombre_base in "${!mapa[@]}"; do
    n_divisiones="${mapa[$nombre_base]}"

    # Contador de archivos encontrados
    contador=0

    cd $output_folder/metadata

    # Si n_divisiones es 1, verificar solo el archivo con el nombre base
    if [ "$n_divisiones" -eq 1 ]; then
        archivo="$nombre_base""_metadata.json"
        if [ ! -f "$archivo" ]; then
            archivos_faltantes+=("$archivo")
        fi
    else
        # Iterar sobre los archivos en el directorio
        for (( i=1; i<=$n_divisiones; i++ )); do
            archivo="${nombre_base}_${i}_metadata.json"
            if [ -f "$archivo" ]; then
                contador=$((contador + 1))
            else
                archivos_faltantes+=("$archivo")
            fi
        done
    fi

done


# Comentar (#)para ver lista de archivos faltantes

#: '
    if [ ${#archivos_faltantes[@]} -gt 0 ]; then
        printf '%s\n' "${archivos_faltantes[@]}"
    fi
#'


#######################################################
### Preparando la corrida de los archivos faltantes ###
#######################################################

# Activar proxy
echo $GRID_PASSWORD | voms-proxy-init --voms cms --pwstdin

# Obtener el shell de Singularity
#env PYTHONNOUSERSITE=1 singularity shell -B /afs -B /eos -B /cvmfs /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.3.0-py3.10 << EOF
if [ "$create_fileset" = "true" ]; then
    echo "Running make_fileset_lxplus.py ..."
    cd "$SCRIPT_DIR/wprime_plus_b/fileset/" || exit 1

    singularity shell -B /afs -B /eos -B /cvmfs \
/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.30-py3.10 <<EOF

PYTHONNOUSERSITE=1 python3 make_fileset_lxplus.py --year "$year"

EOF

else
    echo "make_fileset_lxplus.py is disabled. Skipping fileset creation."
fi

# Main directory
cd "$SCRIPT_DIR"


# Run build_filesets once before the loop
if [ "$create_partitions" = "true" ]; then
  python3 -c "
from utils import update_nsplit, build_filesets

print('::::: Running update_nsplit for the year $year :::::')
update_nsplit('$year')

print('::::: Creating sample partitions with build_filesets() for $year ::::::')

args = {
    'processor': '$processor',
    'lepton_flavor': '$lepton_flavor',
    'year': '$year',
    'run_systematics': '$run_systematics',
    'sample': 'TTToSemiLeptonic',
    'facility': 'lxplus',
    'global_redirector': '$global_redirector'
}
build_filesets(args)
"
else
  echo '::::: create_partitions=false → se saltan las particiones :::::'
fi


# Iterar sobre cada archivo faltante
for archivo_faltante in "${archivos_faltantes[@]}"; do
    # Remove suffix “_metadata.json”
    archivo_sin_ext="${archivo_faltante%_metadata.json}"

    # Check if it ends with _number (e.g., _2, _500
    if [[ "$archivo_sin_ext" =~ _[0-9]+$ ]]; then
        nsample=$(echo "$archivo_sin_ext" | awk -F'_' '{print $NF}')
        nombre_base="${archivo_sin_ext%_*}"
    else
        nsample=""
        nombre_base="$archivo_sin_ext"
    fi

    # Add nsample only if it is not empty
    extra_arg=""
    if [[ -n "$nsample" ]]; then
        extra_arg="--nsample $nsample"
    fi


    if [ "$processor" == "test" ]; then
        comando="python3 submit_lxplus.py \
            --processor $processor \
            --channel $channel \
            --lepton_flavor $lepton_flavor \
            --sample $nombre_base \
            --year $year \
            --nfiles "-1" \
            --executor $executor \
            --output_type $output_type \
            --run_systematics $run_systematics \
            --qcd_data_driven $qcd_data_driven \
            --output_folder $output_folder \
            $extra_arg"


    elif [ "$processor" == "top_tagger" ] || \
        [ "$processor" == "wplusjets" ] || \
        [ "$processor" == "signal" ] || \
        [ "$processor" == "qcd_hadronic_closure" ] || \
        [ "$processor" == "wjets" ] || \
        [ "$processor" == "ztoll" ] || \
        [ "$processor" == "zplusc" ] || \
        [ "$processor" == "btag_eff" ] || \
        [ "$processor" == "ctag_eff" ]; then

        comando="python3 submit_lxplus.py \
            --processor $processor \
            --lepton_flavor $lepton_flavor \
            --sample $nombre_base \
            --year $year \
            --nfiles "-1" \
            --executor $executor \
            --output_type $output_type \
            --run_systematics $run_systematics \
            --qcd_data_driven $qcd_data_driven \
            --unblinded $unblinded \
            --global_redirector $global_redirector \
            --output_folder $output_folder \
            $extra_arg"
    fi

    

    cd $SCRIPT_DIR

    # Submit jobs
    eval "$comando"
done


# Get the number of missing files
num_missing_files=${#archivos_faltantes[@]}

# Save the number of missing files in a file
echo "########################################"
echo "########### Resultados  ################"
echo "Número de archivos faltantes: $num_missing_files"
echo "########################################"

