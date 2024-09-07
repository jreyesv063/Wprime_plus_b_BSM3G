# W' + b

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="left">
  <img width="300" src="https://i.imgur.com/OWhX13O.jpg" />
</p>

Python package for analyzing W' + b in the electron and muon channels. The analysis uses a columnar framework to process input tree-based [NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) files using the [coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries.


- [Data/MC filesets](Data/MC-filesets)
    * [Making the input filesets for Lxplus](#Making-the-input-filesets-for-Lxplus)
- [Submitting jobs at lxplus](#Submitting-jobs-at-lxplus)
- [Corrections and scale factors](#Corrections-and-scale-factors)
- [Luminosity](#luminosity)


## Data/MC filesets

#### Making the input filesets for Lxplus

It has been observed that, in lxplus, opening files through a concrete xrootd endpoint rather than a redirector is far more robust. Use the [make_fileset_lxplus.py](https://github.com/deoache/wprime_plus_b/blob/main/wprime_plus_b/fileset/make_fileset_lxplus.py) script to build the input filesets with xrootd endpoints:
```
# connect to lxplus 
ssh <your_username>@lxplus.cern.ch

# then activate your proxy
voms-proxy-init --voms cms

# clone the repository 
git clone -b refactor_highpt https://github.com/deoache/wprime_plus_b.git

# move to the fileset directory
cd wprime_plus_b/wprime_plus_b/fileset/

# get the singularity shell 
singularity shell -B /afs -B /eos -B /cvmfs /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest-py3.10

# run the 'make_fileset_lxplus' script
python make_fileset_lxplus.py

# exit the singularity
exit
```
We use the [dataset discovery tools](https://coffeateam.github.io/coffea/notebooks/dataset_discovery.html) from Coffea 2024, that's why we need to use a singularity shell in which we can use these tools.

The json files containing the datasets will be saved in the `wprime_plus_b/fileset` directory. This files are the input to the [build_filesets](https://github.com/deoache/wprime_plus_b/blob/main/utils.py#L36) function that divides each dataset into `nsplit` datasets (located in the `wprime_plus_b/fileset/<facility>` folder), which are the datasets read in the execution step. The `nsplit` of each dataset are defined [here](https://github.com/deoache/wprime_plus_b/blob/main/wprime_plus_b/configs/dataset/datasets_configs.yaml).

### Submitting condor jobs at lxplus 

To submit jobs at lxplus using HTCondor we use the [submit_coffeacasa.py](https://github.com/deoache/wprime_plus_b/blob/main/submit_lxplus.py) script. It will create the condor and executable files (using the [submit.sub](https://github.com/deoache/wprime_plus_b/blob/main/condor/submit.sub) and [submit.sh](https://github.com/deoache/wprime_plus_b/blob/main/condor/submit.sh) templates) needed to submit jobs, as well as the folders containing the logs and outputs within the `/condor` folder (click [here](https://batchdocs.web.cern.ch/local/quick.html) for more info). 

To see a list of arguments needed to run this script please enter the following in the terminal:

```bash
python3 submit_lxplus.py --help
```
The output should look something like this:

```
usage: submit_lxplus.py [-h] [--processor PROCESSOR] [--channel CHANNEL] [--lepton_flavor LEPTON_FLAVOR] [--sample SAMPLE] [--year YEAR] [--yearmod YEARMOD] [--executor EXECUTOR]
                        [--workers WORKERS] [--nfiles NFILES] [--output_type OUTPUT_TYPE] [--syst SYST] [--nsample NSAMPLE]

optional arguments:
  -h, --help            show this help message and exit
  --processor PROCESSOR
                        processor to be used {ttbar, ztoll, qcd, trigger_eff, btag_eff} (default ttbar)
  --channel CHANNEL     channel to be processed {2b1l, 1b1e1mu, 1b1l}
  --lepton_flavor LEPTON_FLAVOR
                        lepton flavor to be processed {mu, ele}
  --sample SAMPLE       sample key to be processed
  --year YEAR           year of the data {2016, 2016APV, 2017, 2018} (default 2017)
  --executor EXECUTOR   executor to be used {iterative, futures, dask} (default iterative)
  --workers WORKERS     number of workers to use with futures executor (default 4)
  --nfiles NFILES       number of .root files to be processed by sample. To run all files use -1 (default 1)
  --output_type OUTPUT_TYPE
                        type of output {hist, array}
  --syst SYST           systematic to apply {'nominal', 'jet', 'met', 'full'}
```
You need to have a valid grid proxy in the CMS VO. (see [here](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideLcgAccess) for details on how to register in the CMS VO). The needed grid proxy is obtained via the usual command
```bash
voms-proxy-init --voms cms
```
To execute a processor using some sample of a particular year type:
```bash
python3 submit_lxplus.py --processor <processor> --channel <channel> --lepton_flavor <lepton_flavor> --nfiles -1 --executor futures --output_type hist --year <year> --sample <sample>
```
You can watch the status of the Condor jobs typing
```bash
watch condor_q
```

The outpus will be saved to `wprime_plus_b/outs/<processor>/<channel>/<lepton_flavor>/<year>/`. Alternatively, you can modify [the output path](https://github.com/deoache/wprime_plus_b/blob/refactor_highpt/wprime_plus_b/utils/path_handler.py#L13) to use your EOS area as output path (make sure it's a Pathlib object). 

After the jobs have run, some of them may not have been executed successfully so we'll need to resubmit these jobs. This can be done using the `resubmit.py` script:
```bash
python3 resubmit.py --processor <processor> --channel <channel> --lepton_flavor <lepton_flavor> --year <year> --output_path <output_path> --resubmit <resubmit>
```
If `--resubmit True` missing jobs will be resubmitted, otherwise they'll just be printed in the screen. Make sure to set the correct `output_path` pointing to your output folder.




## Corrections and scale factors

We implemented particle-level corrections and event-level scale factors

### Particle-level corrections 

**JEC/JER corrections**: The basic idea behind the JEC corrections at CMS is the following: *"The detector response to particles is not linear and therefore it is not straightforward to translate the measured jet energy to the true particle or parton energy. The jet corrections are a set of tools that allows the proper mapping of the measured jet energy deposition to the particle-level jet energy"* (see https://twiki.cern.ch/twiki/bin/view/CMS/IntroToJEC).

We follow the recomendations by the Jet Energy Resolution and Corrections (JERC) group (see https://twiki.cern.ch/twiki/bin/viewauth/CMS/JECDataMC#Recommended_for_MC). In order to apply these corrections to the MC (in data, the corrections are already applied) we use the `jetmet_tools` from Coffea (https://coffeateam.github.io/coffea/modules/coffea.jetmet_tools.html). With these tools, we construct the [Jet and MET factories](wprime_plus_b/data/scripts/build_jec.py) which contain the JEC/JER corrections that are eventually loaded in the function [`jet_corrections`](wprime_plus_b/corrections/jec.py), which is the function we use in the processors to apply the corrections to the jet and MET objects.

**Note**: Since we modify the kinematic properties of jets, we must recalculate the MET. That's the work of the MET factory: it takes the corrected jets as an argument, and use them to recalculate the MET.

**Note:** These corrections must be applied before performing any kind of selection.

**MET phi modulation:** The distribution of true MET is independent of $\phi$ because of the rotational symmetry of the collisions around the beam axis. However, we observe that the reconstructed MET does depend on $\phi$. The MET $\phi$ distribution has roughly a sinusoidal curve with the period of $2\pi$. The possible causes of the modulation include anisotropic detector responses, inactive calorimeter cells, the detector misalignment, the displacement of the beam spot. The amplitude of the modulation increases roughly linearly with the number of the pile-up interactions.

We implement this correction [here](wprime_plus_b/corrections/met.py). This correction reduces the MET $\phi$ modulation. It is also a mitigation for the pile-up effects. 

(taken from https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookMetAnalysis#7_7_6_MET_Corrections)

### Event-level scale factors (SF)

We use the common json format for scale factors (SF), hence the requirement to install [correctionlib](https://github.com/cms-nanoAOD/correctionlib). The SF themselves can be found in the central [POG repository](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration), synced once a day with CVMFS: `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration`. A summary of their content can be found [here](https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/). The SF implemented are:

* [Pileup SF](wprime_plus_b/corrections/pileup.py)
* [Electron ID, Reconstruction and Trigger SF](wprime_plus_b/corrections/lepton.py) (see the `ElectronCorrector` class)
* [Muon ID, Iso and TriggerIso ](wprime_plus_b/corrections/lepton.py) (see the `MuonCorrector` class)
* [PileupJetId SF](wprime_plus_b/corrections/pujetid.py)
* L1PreFiring SF: These are read from the NanoAOD events as `events.L1PreFiringWeight.Nom/Up/Dn`.

*We derive our own set of trigger scale factors. 

* B-tagging: b-tagging weights are computed as (see https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods):

  $$w = \prod_{i=\text{tagged}} \frac{SF_{i} \cdot \varepsilon_i}{\varepsilon_i} \prod_{j=\text{not tagged}} \frac{1 - SF_{j} \cdot \varepsilon_j}{1-\varepsilon_j} $$
  
  where $\varepsilon_i$ is the MC b-tagging efficiency and $\text{SF}$ are the b-tagging scale factors. $\text{SF}_i$ and $\varepsilon_i$ are functions of the jet flavor, jet $p_T$, and jet $\eta$. It's important to notice that the two products are 1. over jets tagged at the respective working point, and 2. over jets not tagged at the respective working point. **This is not to be confused with the flavor of the jets**.
  
  We can see, then, that the calculation of these weights require the knowledge of the MC b-tagging efficiencies, which depend on the event kinematics. It's important to emphasize that **the BTV POG only provides the scale factors and it is the analyst responsibility to compute the MC b-tagging efficiencies for each jet flavor in their signal and background MC samples before applying the scale factors**. The calculation of the MC b-tagging efficiencies is describe [here](https://github.com/deoache/wprime_plus_b/blob/refactor/corrections/binder/btag_eff.ipynb).

  The computation of the b-tagging weights can be found [here](wprime_plus_b/corrections/btag.py)



## Luminosity

See luminosity recomendations for Run2 at https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2. To obtain the integrated luminosity type (on lxplus):

```
export PATH=$HOME/.local/bin:/afs/cern.ch/cms/lumi/brilconda-1.1.7/bin:$PATH
pip uninstall brilws
pip install --install-option="--prefix=$HOME/.local" brilws
```

* SingleMuon: type

```
brilcalc lumi -b "STABLE BEAMS" --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /fb -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt --hltpath HLT_IsoMu27_v*
```

output:
```
#Summary: 
+-----------------+-------+------+--------+-------------------+------------------+
| hltpath         | nfill | nrun | ncms   | totdelivered(/fb) | totrecorded(/fb) |
+-----------------+-------+------+--------+-------------------+------------------+
| HLT_IsoMu27_v10 | 13    | 36   | 8349   | 2.007255669       | 1.870333304      |
| HLT_IsoMu27_v11 | 9     | 21   | 5908   | 1.383159994       | 1.254273727      |
| HLT_IsoMu27_v12 | 47    | 122  | 46079  | 8.954672794       | 8.298296788      |
| HLT_IsoMu27_v13 | 91    | 218  | 124447 | 27.543983745      | 26.259684708     |
| HLT_IsoMu27_v14 | 2     | 13   | 4469   | 0.901025085       | 0.862255849      |
| HLT_IsoMu27_v8  | 2     | 3    | 1775   | 0.246872270       | 0.238466292      |
| HLT_IsoMu27_v9  | 11    | 44   | 14260  | 2.803797063       | 2.694566730      |
+-----------------+-------+------+--------+-------------------+------------------+
#Sum delivered : 43.840766620
#Sum recorded : 41.477877399
```

* SingleElectron: type

```
brilcalc lumi -b "STABLE BEAMS" --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /fb -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt --hltpath HLT_Ele35_WPTight_Gsf_v*
```
output:

```
#Summary: 
+--------------------------+-------+------+--------+-------------------+------------------+
| hltpath                  | nfill | nrun | ncms   | totdelivered(/fb) | totrecorded(/fb) |
+--------------------------+-------+------+--------+-------------------+------------------+
| HLT_Ele35_WPTight_Gsf_v1 | 2     | 3    | 1775   | 0.246872270       | 0.238466292      |
| HLT_Ele35_WPTight_Gsf_v2 | 11    | 44   | 14260  | 2.803797063       | 2.694566730      |
| HLT_Ele35_WPTight_Gsf_v3 | 13    | 36   | 8349   | 2.007255669       | 1.870333304      |
| HLT_Ele35_WPTight_Gsf_v4 | 9     | 21   | 5908   | 1.383159994       | 1.254273727      |
| HLT_Ele35_WPTight_Gsf_v5 | 20    | 66   | 22775  | 5.399580877       | 4.879405647      |
| HLT_Ele35_WPTight_Gsf_v6 | 27    | 56   | 23304  | 3.555091917       | 3.418891141      |
| HLT_Ele35_WPTight_Gsf_v7 | 93    | 231  | 128916 | 28.445008830      | 27.121940558     |
+--------------------------+-------+------+--------+-------------------+------------------+
#Sum delivered : 43.840766620
#Sum recorded : 41.477877399
```