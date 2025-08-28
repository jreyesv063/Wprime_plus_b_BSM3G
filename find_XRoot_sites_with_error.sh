#!/bin/bash

# Mapa de host → centro (en construcción)
declare -A host_to_site=(
["xrootd.hep.kbfi.ee"]="T2_EE_Estonia"
    ["cmsxrootd.fnal.gov"]="T1_US_FNAL_Disk"
    ["xrootd-redir.ultralight.org"]="T2_US_Nebraska"
    ["xrootd-cms.infn.it"]="T1_IT_CNAF_Disk"
    ["xrootd-cms-psu0.fnal.gov"]="T3_US_FNALLPC"
    ["xrootd-cmsbdii.physik.rwth-aachen.de"]="T2_DE_RWTH"
    ["xrootd-cms.ucl.ac.be"]="T2_BE_UCL"
    ["xrootd-cms.ciemat.es"]="T2_ES_CIEMAT"
    ["xrootd-cms.hephy.oeaw.ac.at"]="T2_AT_Vienna"
    ["xrootd-cms2.gridpp.rl.ac.uk"]="T2_UK_London_IC"
    ["xrootd-cms-kr.kisti.re.kr"]="T3_KR_KISTI"
    ["cms-xrd-global.cern.ch"]="T2_CH_CERN"
    ["xrootd.unl.edu"]="T2_US_Nebraska"
    ["cmsxrootd.gridka.de"]="T1_DE_KIT_Disk"
    ["xrootd.cmsaf.mit.edu"]="T2_US_MIT"
    ["xrootd-cms.psu.edu"]="T3_US_FNALLPC"
    ["cms.sscc.uos.ac.kr"]="T3_KR_UOS"
    ["ccxrootd.in2p3.fr"]="T1_FR_CCIN2P3_Disk"
    ["ccxrootdtapedata.in2p3.fr"]="T1_FR_CCIN2P3_Tape"
    ["xrootd-cmst1.pic.es"]="T1_ES_PIC_Disk"
    ["xrootd.echo.stfc.ac.uk"]="T1_UK_RAL_Disk"
    ["xrootd.jinr-t1.ru"]="T1_RU_JINR_Disk"
    ["xrootd.rcac.purdue.edu"]="T2_US_Purdue"
    ["dcache-cms-xrootd.desy.de"]="T2_DE_DESY"
    ["se01.grid.nchc.org.tw"]="T2_TW_NCHC"
    ["xrootd.uerj.br"]="T2_BR_UERJ"
    ["xrootdiphc.in2p3.fr"]="T2_FR_IPHC"
    ["se01.indiacms.res.in"]="T2_IN_TIFR"
    ["cmsdcatape.tifr.res.in"]="T2_IN_TIFR"
    ["cmsstorage.tifr.res.in"]="T2_IN_TIFR"
    ["xrootd-lhcb.infn.it"]="T2_IT_Legnaro"
    ["xrootd-cms-roma1.infn.it"]="T2_IT_Rome"
    ["xrootd-cms.infn.kfki.hu"]="T2_HU_Budapest"
    ["xrootd-archive.cyfronet.pl"]="T2_PL_Cyfronet"
    ["xrootd.caltech.edu"]="T2_US_Caltech"
    ["xrootd.sao.sprace.org.br"]="T2_BR_SPRACE"
    ["xrootd.ipnl.in2p3.fr"]="T3_FR_IPNL"
    ["opendata.cern.ch"]="T3_CH_CERN_OpenData"
    ["xrootd-psi.grid.psi.ch"]="T3_CH_PSI"
    ["xrootd-hep.nd.edu"]="T3_US_NotreDame"
    ["xrootd.grid.baylor.edu"]="T3_US_Baylor"
    ["xrootd-cms-mirror.cr.cnaf.infn.it"]="T1_IT_CNAF_Disk"
    ["xrootd-local.unl.edu"]="T2_US_Nebraska"
    ["xrootd2.hepgrid.uerj.br"]="T2_BR_UERJ"
    ["xrootd01.jinr-t1.ru"]="T1_RU_JINR_Disk"
    ["grid142.kfki.hu"]="T2_HU_Budapest" 
    ["cmsxrootd1.physics.ntu.edu.tw"]="T2_TW_NCHC"
    ["xrootd-cms.cnaf.infn.it"]="T1_IT_CNAF_Disk"
    ["xrootd-cms.eki.ee"]="T2_EE_Estonia"
    ["se01.indiacms.res.in"]="T2_IN_TIFR"
    ["xrootd-legnaro.pd.infn.it"]="T2_IT_Legnaro"
    ["xrootd.cmsaf.mit.edu"]="T2_US_MIT"
    ["eos.cms.rcac.purdue.edu"]="T2_US_Purdue"
    ["maite.iihe.ac.be"]="T2_BE_IIHE"
    ["cmsxrootd.hep.wisc.edu"]="T2_US_Wisconsin"
    ["eoscms.cern.ch"]="T2_CH_CERN"
    ["eos01.grid.cyfronet.pl"]="T2_PL_Cyfronet"
    ["k8s-redir.ultralight.org"]="T2_US_Caltech"
    ["xrootd-cmst1-door.pic.es"]="T1_ES_PIC_Disk"
    ["cmsdcache-kit-disk.gridka.de"]="T1_DE_KIT_Disk"
    ["xrootd.cmsaf.vanderbilt.edu"]="T2_US_Vanderbilt"
    ["skynet013.crc.nd.edu"]="T3_US_NotreDame"
    ["cmsdcadisk.fnal.gov"]="T1_US_FNAL_Disk"
    ["xrootd-vanderbilt.sites.opensciencegrid.org"]="T2_US_Vanderbilt"
    ["sbgdcache.in2p3.fr"]="T1_FR_CCIN2P3_Disk"
    ["eos01.grid.cyfronet.pl:1094"]="T2_PL_Cyfronet"
    ["rdr.echo.stfc.ac.uk"]="T1_UK_RAL"
    ["t3se01.psi.ch"]="T3_CH_PSI"
    ["grid143.kfki.hu"]="T2_HU_Budapest"
    ["cmseos.fnal.gov"]="T1_US_FNAL_Disk"
)

# Mapa de años con sus sitios válidos
declare -A valid_sites=(
    ["2016APV"]="T1_US_FNAL_Disk T1_FR_CCIN2P3_Tape T1_FR_CCIN2P3_Disk T1_DE_KIT_Disk T1_ES_PIC_Disk T1_UK_RAL_Disk T1_RU_JINR_Disk T2_US_Purdue T2_US_Nebraska T2_DE_DESY T2_TW_NCHC T2_CH_CERN T2_DE_RWTH T2_UK_London_IC T2_BR_UERJ T2_FR_IPHC T2_US_MIT T2_IN_TIFR T2_BE_IIHE T2_US_Vanderbilt T2_IT_Legnaro T2_IT_Rome T3_US_FNALLPC T3_FR_IPNL T3_CH_CERN_OpenData T3_KR_UOS T3_IT_Trieste"
    ["2016"]="T1_US_FNAL_Disk T1_FR_CCIN2P3_Tape T1_FR_CCIN2P3_Disk T1_DE_KIT_Disk T1_ES_PIC_Disk T1_UK_RAL_Disk T1_RU_JINR_Disk T2_US_Purdue T2_US_Nebraska T2_DE_DESY T2_TW_NCHC T2_CH_CERN T2_DE_RWTH T2_UK_London_IC T2_US_MIT T2_BR_UERJ T2_FR_IPHC T2_IN_TIFR T2_BE_IIHE T2_US_Vanderbilt T2_IT_Legnaro T2_IT_Rome T3_US_FNALLPC T3_FR_IPNL T3_CH_CERN_OpenData T3_KR_UOS T3_IT_Trieste"
    ["2017"]="T1_FR_CCIN2P3_Tape T1_US_FNAL_Disk T2_US_Purdue T2_BE_UCL T2_DE_RWTH T2_CH_CERN T2_BE_IIHE T2_HU_Budapest T2_US_Vanderbilt T2_ES_CIEMAT T2_UK_London_IC T2_US_Nebraska T2_EE_Estonia T3_US_FNALLPC T3_FR_IPNL T3_IT_Trieste T3_CH_PSI T3_KR_KISTI T3_KR_UOS T3_US_NotreDame T3_US_Baylor"
    ["2018"]="T1_DE_KIT_Disk T1_IT_CNAF_Disk T1_UK_RAL_Disk T1_FR_CCIN2P3_Tape T1_FR_CCIN2P3_Disk T1_RU_JINR_Disk T1_US_FNAL_Disk T2_US_Purdue T2_DE_DESY T2_CH_CERN T2_DE_RWTH T2_US_Wisconsin T2_BE_UCL T2_PL_Cyfronet T2_US_Caltech T2_BR_SPRACE T2_UK_London_IC T2_IT_Rome T2_BE_IIHE T2_HU_Budapest T2_US_Vanderbilt T2_FR_IPHC T2_US_Nebraska T3_US_FNALLPC T3_IT_Trieste"
)

# Mostrar ayuda si no hay argumentos o con -h/--help
show_help() {
    echo "Uso: $0 [opciones]"
    echo "Opciones:"
    echo "  -y, --year YEAR       Especifica el año a procesar (2016APV, 2016, 2017, 2018)"
    echo "  -p, --processor PROC  Especifica el procesador (ej: top_tagger)"
    echo "  -l, --lepton LEPTON   Especifica el sabor de leptón (ej: tau)"
    echo "  -c, --channel CHAN    Especifica el canal (opcional)"
    echo "  -h, --help            Muestra esta ayuda"
    exit 0
}

# Procesar argumentos
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|-year)
            year="$2"
            shift 2
            ;;
        -p|-processor)
            processor="$2"
            shift 2
            ;;
        -l|-lepton)
            lepton_flavor="$2"
            shift 2
            ;;
        -c|-channel)
            channel="$2"
            shift 2
            ;;
        -h|-help)
            show_help
            ;;
        *)
            echo "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validar argumentos obligatorios
if [[ -z "$year" || -z "$processor" || -z "$lepton_flavor" ]]; then
    echo "Error: Faltan argumentos obligatorios"
    show_help
    exit 1
fi

# Verificar que el año es válido
if [[ ! -v valid_sites[$year] ]]; then
    echo "Error: Año no válido. Opciones: 2016APV, 2016, 2017, 2018"
    exit 1
fi

# Construir ruta según si hay channel o no
if [[ -n "$channel" ]]; then
    dir_to_check="condor/logs/$processor/$channel/$lepton_flavor/$year"
else
    dir_to_check="condor/logs/$processor/$lepton_flavor/$year"
fi

# Función para procesar un año específico
process_year() {
    local year=$1
    local dir_to_check=$2
    
    # Contador de errores y conjunto de centros/hosts únicos
    local error_count=0
    declare -A centers_or_unknown=()
    declare -A invalid_sites=()

    # Buscar archivos con "Error"
    IFS=$'\n'
    for filepath in $(grep -rl "Error" "$dir_to_check" 2>/dev/null); do
        error_line=$(grep "XRootD error" "$filepath")

        if [[ $error_line == *"XRootD error"* ]]; then
            filename=$(basename "$filepath")
            if [[ -n "$channel" ]]; then
                sample=$(echo "$filepath" | grep -oP "logs/.+?/$channel/$lepton_flavor/\d{4}(APV)?/\K[^/]+")
            else
                sample=$(echo "$filepath" | grep -oP "logs/.+?/$lepton_flavor/\d{4}(APV)?/\K[^/]+")
            fi
            host_line=$(grep -A1 "XRootD error" "$filepath" | tail -n1)
            host=$(echo "$host_line" | grep -oP 'root://\K[^:/]+')

            site=${host_to_site[$host]:-$host}  # Usa el nombre del host si no se encuentra en el mapa

            # Verificar si el sitio es válido para este año
            if [[ ! " ${valid_sites[$year]} " =~ " ${site} " ]]; then
                invalid_sites["$site"]=1
            fi

            echo "$sample"
            echo "${filename%.err} error en $host → $site"
            echo

            ((error_count++))
            centers_or_unknown["$site"]=1
        fi
    done

    # Mostrar resumen de errores
    echo "############################################"
    echo "######  Resumen de errores para $year ######"
    echo "############################################"
    
    if [ ${#invalid_sites[@]} -gt 0 ]; then
        echo "Sitios no válidos encontrados (comentar en make_fileset_lxplus.py):"
        for site in "${!invalid_sites[@]}"; do
            echo " - $site"
        done
        echo
    fi
    
    echo "Todos los sitios con errores:"
    for site in "${!centers_or_unknown[@]}"; do
        echo " - $site"
    done
    echo
    echo "Total de errores encontrados: $error_count"
    echo
}

# Procesar el año
echo "Procesando año: $year"
echo "Processor: $processor"
echo "Lepton flavor: $lepton_flavor"
[[ -n "$channel" ]] && echo "Channel: $channel"
echo

process_year "$year" "$dir_to_check"


# Sin channel
#./find_XRoot_sites_with_error.sh -p top_tagger -l tau -y 2017

# Con channel
#./find_XRoot_sites_with_error.sh -processor top_tagger -lepton tau -channel my_channel -year 2017 