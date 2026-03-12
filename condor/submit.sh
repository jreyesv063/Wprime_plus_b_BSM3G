#!/bin/bash

echo ">>> Starting Condor script"

# Network/XROOTD variables
export XRD_NETWORKSTACK=IPv4
export XRD_RUNFORKHANDLER=1
export X509_USER_PROXY=X509PATH

echo ">>> Checking proxy"
voms-proxy-info -all || echo "❌ voms-proxy-info -all failed"
voms-proxy-info -all -file "$X509_USER_PROXY" || echo "❌ voms-proxy-info -all -file failed"

echo ">>> Changing to directory: MAINDIRECTORY"
cd MAINDIRECTORY || { echo "❌ Failed to cd into MAINDIRECTORY"; exit 1; }


echo ">>> Python version:"
python --version

echo ">>> Coffea version:"
python -c "import coffea; print(coffea.__version__)"

echo ">>> Running command:"
echo "COMMAND"
COMMAND || { echo "❌ Command failed"; exit 2; }

echo "✅ Script finished successfully"
exit 0

