#!/bin/bash

echo ">>> Starting Condor script"

# Network/XROOTD variables (Robust configuration)
export XRD_NETWORKSTACK=IPv4
export XRD_RUNFORKHANDLER=1
export X509_USER_PROXY=X509PATH

# --- NEW TOLERANCE VARIABLES ADDED ---
export XRD_CONNECTIONRETRY=20      # Retry up to 20 times if the server fails
export XRD_REQUESTTIMEOUT=300      # Wait up to 5 minutes for a response
export XRD_STREAMTIMEOUT=300       # Wait up to 5 minutes for continuous data stream
export XRD_RECONNECTWAIT=5         # Wait 5 seconds between connection retries
# -----------------------------------------------

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