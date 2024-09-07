import gzip
import cloudpickle
from pathlib import Path
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/JECDataMC#Recommended_for_MC
# JEC: https://github.com/cms-jet/JECDatabase/tree/master/textFiles
# JER: https://github.com/cms-jet/JRDatabase/tree/master/tarballs
jec_name_map = {
    "JetPt": "pt",
    "JetMass": "mass",
    "JetEta": "eta",
    "JetA": "area",
    "ptGenJet": "pt_gen",
    "ptRaw": "pt_raw",
    "massRaw": "mass_raw",
    "Rho": "event_rho",
    "METpt": "pt",
    "METphi": "phi",
    "JetPhi": "phi",
    "UnClusteredEnergyDeltaX": "MetUnclustEnUpDeltaX",
    "UnClusteredEnergyDeltaY": "MetUnclustEnUpDeltaY",
}

data_path = Path(Path.home(), "wprime_plus_b/wprime_plus_b/data")


def jet_factory_factory(files):
    ext = extractor()
    ext.add_weight_sets([f"* * {Path(data_path, file)}" for file in files])
    ext.finalize()
    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)


def get_mc_factories():
    jet_factory = {
        "2016APV": jet_factory_factory(
            files=[
                # JEC
                "JEC/MC/2016APV/Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs.jec.txt",
                "JEC/MC/2016APV/Summer19UL16APV_V7_MC_L2L3Residual_AK4PFchs.jec.txt",
                "JEC/MC/2016APV/Summer19UL16APV_V7_MC_L2Relative_AK4PFchs.jec.txt",
                "JEC/MC/2016APV/Summer19UL16APV_V7_MC_L3Absolute_AK4PFchs.jec.txt",
                "JEC/MC/2016APV/RegroupedV2_Summer19UL16APV_V7_MC_UncertaintySources_AK4PFchs.junc.txt",
                # JER
                "JER/MC/2016APV/Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs.jr.txt",
                "JER/MC/2016APV/Summer20UL16APV_JRV3_MC_SF_AK4PFchs.jersf.txt",
            ]
        ),
        "2016": jet_factory_factory(
            files=[
                # JEC
                "JEC/MC/2016/Summer19UL16_V7_MC_L1FastJet_AK4PFchs.jec.txt",
                "JEC/MC/2016/Summer19UL16_V7_MC_L2L3Residual_AK4PFchs.jec.txt",
                "JEC/MC/2016/Summer19UL16_V7_MC_L2Relative_AK4PFchs.jec.txt",
                "JEC/MC/2016/Summer19UL16_V7_MC_L3Absolute_AK4PFchs.jec.txt",
                "JEC/MC/2016/RegroupedV2_Summer19UL16_V7_MC_UncertaintySources_AK4PFchs.junc.txt",
                # JER
                "JER/MC/2016/Summer20UL16_JRV3_MC_PtResolution_AK4PFchs.jr.txt",
                "JER/MC/2016/Summer20UL16_JRV3_MC_SF_AK4PFchs.jersf.txt",
            ]
        ),
        "2017": jet_factory_factory(
            files=[
                # JEC
                "JEC/MC/2017/Summer19UL17_V5_MC_L1FastJet_AK4PFchs.jec.txt",
                # "JEC/MC/Summer19UL17_V5_MC_L1RC_AK4PFchs.jec.txt",
                "JEC/MC/2017/Summer19UL17_V5_MC_L2L3Residual_AK4PFchs.jec.txt",
                "JEC/MC/2017/Summer19UL17_V5_MC_L2Relative_AK4PFchs.jec.txt",
                # "JEC/MC/Summer19UL17_V5_MC_L2Residual_AK4PFchs.jec.txt",
                "JEC/MC/2017/Summer19UL17_V5_MC_L3Absolute_AK4PFchs.jec.txt",
                "JEC/MC/2017/RegroupedV2_Summer19UL17_V5_MC_UncertaintySources_AK4PFchs.junc.txt",
                # JER
                # "JER/MC/Summer19UL17_JRV3_MC_EtaResolution_AK4PFchs.jr.txt",
                # "JER/MC/Summer19UL17_JRV3_MC_PhiResolution_AK4PFchs.jr.txt",
                "JER/MC/2017/Summer19UL17_JRV3_MC_PtResolution_AK4PFchs.jr.txt",
                "JER/MC/2017/Summer19UL17_JRV3_MC_SF_AK4PFchs.jersf.txt",
            ]
        ),
        "2018": jet_factory_factory(
            files=[
                # JEC
                "JEC/MC/2018/Summer19UL18_V5_MC_L1FastJet_AK4PFchs.jec.txt",
                "JEC/MC/2018/Summer19UL18_V5_MC_L2L3Residual_AK4PFchs.jec.txt",
                "JEC/MC/2018/Summer19UL18_V5_MC_L2Relative_AK4PFchs.jec.txt",
                "JEC/MC/2018/Summer19UL18_V5_MC_L3Absolute_AK4PFchs.jec.txt",
                "JEC/MC/2018/RegroupedV2_Summer19UL18_V5_MC_UncertaintySources_AK4PFchs.junc.txt",
                # JER
                "JER/MC/2018/Summer19UL18_JRV2_MC_PtResolution_AK4PFchs.jr.txt",
                "JER/MC/2018/Summer19UL18_JRV2_MC_SF_AK4PFchs.jersf.txt",
            ]
        ),
    }
    met_factory = CorrectedMETFactory(jec_name_map)

    return jet_factory, met_factory


def save_factory(jet_factory, met_factory, name):
    # jme stuff not pickleable in coffea
    with gzip.open(f"{data_path}/{name}_jec_compiled.pkl.gz", "wb") as fout:
        cloudpickle.dump(
            {
                "jet_factory": jet_factory,
                "met_factory": met_factory,
            },
            fout,
        )

if __name__ == "__main__":
    save_factory(*get_mc_factories(), name="mc")
