#!/bin/bash

export XRD_NETWORKSTACK=IPv4
export XRD_RUNFORKHANDLER=1
export X509_USER_PROXY=X509PATH
voms-proxy-info -all
voms-proxy-info -all -file X509PATH
cd MAINDIRECTORY

COMMAND