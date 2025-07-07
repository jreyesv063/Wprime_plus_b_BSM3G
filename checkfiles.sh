#!/bin/bash

# Before running this script, be sure to grant execution permissions with the following command:
# chmod +x checkfiles.sh
# Script is used with the command ./checkfiles.sh

echo "########################################"
echo "########### Checking files  ############"
echo "########################################"


###################################
##### Variables a modificar  ######
###################################
# Indicar si se quiere crear el fileset o no: Importante ponerlo en true si se comentaron servidores.
create_fileset=false #Si se quiere crear el fileset, si no se quiere crear, ponerlo a false

# Directorio del archivo bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuración de opciones
considerar_MET=false
considerar_SingleMuon=true
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
considerar_qcd=true


considerar_signal_tau=false
considerar_signal_ele=false
considerar_signal_mu=false


considerar_inclusive_dy=false
considerar_inclusive_ext_dy=false
considerar_inclusive_ch3=false
considerar_dy=false


# Carpeta donde está el resultado de la corrida
outfolder="outs" 

: '
    El resto de código es automatico.

    consider_xxxx permite determinar si queremos considerar cierto background o data
'

#####################################
#####################################

# Ruta al archivo run.sh
archivo_run="run.sh"


# Extraer las variables del archivo run.sh
processor=$(grep -o 'processor=".*"' "$archivo_run" | cut -d'"' -f2)
channel=$(grep -o 'channel=".*"' "$archivo_run" | cut -d'"' -f2)
lepton_flavor=$(grep -o 'lepton_flavor=".*"' "$archivo_run" | cut -d'"' -f2)
year=$(grep -o 'year=".*"' "$archivo_run" | cut -d'"' -f2)
nfiles=$(grep -o 'nfiles=".*"' "$archivo_run" | cut -d'"' -f2)
executor=$(grep -o 'executor=".*"' "$archivo_run" | cut -d'"' -f2)
output_type=$(grep -o 'output_type=".*"' "$archivo_run" | cut -d'"' -f2)
run_systematics=$(grep -o 'run_systematics=".*"' "$archivo_run" | cut -d'"' -f2)




# Seleccionar el archivo YAML y actualizarlo solo si run_systematics es true
if [ "$run_systematics" == "true" ]; then
    # Llamar a la función Python solo si se corren sistemáticos
    echo "Updating nsplit for year $year"
    python3 -c "from utils import update_nsplit; update_nsplit('$year')"
    yaml_file="datasets_configs_systematics.yaml"
else
    yaml_file="datasets_configs.yaml"
fi


# Cambiar al directorio donde está el archivo
cd wprime_plus_b/configs/dataset


# Crear un mapa vacío
declare -A mapa


# Leer el archivo 
while IFS= read -r nombre_archivo && IFS= read -r divisiones; do

    # Extraer el nombre del archivo y el número de divisiones
    nombre_archivo=$(echo "$nombre_archivo" | sed 's/:$//')  # Eliminar los dos puntos al final del nombre del archivo
    divisiones=$(echo "$divisiones" | awk '{print $2}')
    mapa["$nombre_archivo"]=$divisiones


done < "$yaml_file"


# Identificar unicamente los dataset de data adecuados
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
    unset mapa["SignalTau_600GeV"]  
    unset mapa["SignalTau_1TeV"]  
    unset mapa["SignalTau_2TeV"]  
    unset mapa["SignalTau_3TeV"]  
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

    # Cambiar al directorio con los archivos de salida
    if [ $processor == "ttbar" ] || [ $processor == "wjets" ] || [ "$processor" == "ztoll" ]  || [ "$processor" == "qcd_abcd" ] || [ $processor == "wplusjets" ] || [ $processor == "qcd_hadronic" ]; then
        cd $SCRIPT_DIR/wprime_plus_b/$outfolder/$processor/$channel/$lepton_flavor/$year/metadata
        #cd ~/wprime_plus_b_new/wprime_plus_b/wprime_plus_b/$outfolder/$processor/$channel/$lepton_flavor/$year/metadata

    elif [ "$processor" == "top_tagger" ]  || [ "$processor" == "signal" ]; then
        cd $SCRIPT_DIR/wprime_plus_b/$outfolder/$processor/$lepton_flavor/$year/metadata
        #cd ~/wprime_plus_b_new/wprime_plus_b/wprime_plus_b/$outfolder/$processor/$lepton_flavor/$year/metadata
    fi


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

cd $SCRIPT_DIR/wprime_plus_b/fileset/

#######################################################
### Preparando la corrida de los archivos faltantes ###
#######################################################

# Activar proxy
echo $GRID_PASSWORD | voms-proxy-init --voms cms

# Obtener el shell de Singularity
singularity shell -B /afs -B /eos -B /cvmfs /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest-py3.10 << EOF

if [ "$create_fileset" = "true" ]; then
    echo "Ejecutando make_fileset_lxplus.py..."
    python make_fileset_lxplus.py
else
    echo "make_fileset_lxplus.py está desactivado"
fi


# Ejecutar el script 'make_fileset_lxplus.py' dentro de Singularity
#python make_fileset_lxplus.py
# Salir del shell de Singularity


exit

EOF


# Directorio principal
cd $SCRIPT_DIR



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


# Iterar sobre cada archivo faltante
for archivo_faltante in "${archivos_faltantes[@]}"; do
    # Eliminar sufijo "_metadata.json"
    archivo_sin_ext="${archivo_faltante%_metadata.json}"

    # Verificar si termina con _número (ej. _2, _500)
    if [[ "$archivo_sin_ext" =~ _[0-9]+$ ]]; then
        nsample=$(echo "$archivo_sin_ext" | awk -F'_' '{print $NF}')
        nombre_base="${archivo_sin_ext%_*}"
    else
        nsample=""
        nombre_base="$archivo_sin_ext"
    fi

    # Agregar nsample solo si no está vacío
    extra_arg=""
    if [[ -n "$nsample" ]]; then
        extra_arg="--nsample $nsample"
    fi

    if [ "$processor" == "ttbar" ] || [ "$processor" == "wjets" ] || [ "$processor" == "ztoll" ] || [ "$processor" == "qcd_abcd" ] || [ "$processor" == "wplusjets" ]  || [ "$processor" == "qcd_hadronic" ]; then
        comando="python3 submit_lxplus.py --processor $processor --channel $channel --lepton_flavor $lepton_flavor --sample $nombre_base --year $year --nfiles $nfiles --executor $executor --output_type $output_type $extra_arg --run_systematics $run_systematics"

    elif [ "$processor" == "top_tagger" ] || [ "$processor" == "signal" ]; then
        comando="python3 submit_lxplus.py --processor $processor --lepton_flavor $lepton_flavor --sample $nombre_base --year $year --nfiles $nfiles --executor $executor --output_type $output_type $extra_arg --run_systematics $run_systematics"
    fi

    # # Extraer el nombre base y el número de muestra del archivo faltante
    # nombre_base=$(echo "$archivo_faltante" | rev | cut -d'_' -f2- | rev)
    # nombre_base=$(echo "$nombre_base" | rev | cut -d'_' -f2- | rev)
    # nsample=$(echo "$archivo_faltante" | sed 's/_metadata.json//' | awk -F'_' '{print $(NF)}')

    # if [ "$processor" == "ttbar" ] || [ "$processor" == "wjets" ] || [ "$processor" == "ztoll" ] || [ "$processor" == "qcd_abcd" ] || [ $processor == "wplusjets" ]  || [ "$processor" == "qcd_hadronic" ]; then
    #     # Construir el comando python
    #     comando="python3 submit_lxplus.py --processor $processor --channel $channel --lepton_flavor $lepton_flavor --sample $nombre_base --year $year --nfiles $nfiles --executor $executor --output_type $output_type --nsample $nsample --run_systematics $run_systematics"

    # elif [ "$processor" == "top_tagger" ] || [ "$processor" == "signal" ]; then
    #     comando="python3 submit_lxplus.py --processor $processor --lepton_flavor $lepton_flavor --sample $nombre_base --year $year --nfiles $nfiles --executor $executor --output_type $output_type --nsample $nsample --run_systematics $run_systematics"
    # fi
    

    cd $SCRIPT_DIR
   
    # Enviar jobs
    eval "$comando"
done


# Obtener el número de archivos faltantes
num_archivos_faltantes=${#archivos_faltantes[@]}

# Guardar el número de archivos faltantes en un archivo
echo "########################################"
echo "########### Resultados  ################"
echo "Número de archivos faltantes: $num_archivos_faltantes"
echo "########################################"

echo "$num_archivos_faltantes" > archivos_faltantes.txt
