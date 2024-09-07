import pickle
from coffea.lumi_tools import LumiMask

if __name__ == "__main__":
    lumi_masks = {
        "2016APV": LumiMask("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
        "2016": LumiMask("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
        "2017": LumiMask("Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"),
        "2018": LumiMask("Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"),
    }

    with open("lumi_masks.pkl", "wb") as handle:
        pickle.dump(lumi_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)
