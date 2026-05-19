import yaml
import awkward as ak
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

# =======================================
# Corrections: object - level
# =======================================
from wprime_plus_b.corrections.jec import apply_jet_corrections
from wprime_plus_b.corrections.rochester import apply_rochester_corrections
from wprime_plus_b.corrections.tau_energy import apply_tau_energy_scale_corrections
from wprime_plus_b.corrections.met import apply_met_unclustered, apply_met_phi_corrections

# =======================================
# Corrections: event-level
# =======================================
from wprime_plus_b.corrections.pileup import add_pileup_weight
from wprime_plus_b.corrections.pdfweights import add_pdf_weight
from wprime_plus_b.corrections.pujetid import add_pujetid_weight
from wprime_plus_b.corrections.genweight import add_genweight_weight
from wprime_plus_b.corrections.l1prefiring import add_l1prefiring_weight
from wprime_plus_b.corrections.psweights import add_particle_shower_weight
from wprime_plus_b.corrections.top_pt_reweighting import add_TopPtReweighting


from wprime_plus_b.corrections.tau import TauCorrector
from wprime_plus_b.corrections.btag import BTagCorrector
from wprime_plus_b.corrections.muon import MuonCorrector
from wprime_plus_b.corrections.electron import ElectronCorrector

# =======================================
# Selections: objects
# =======================================
from wprime_plus_b.object_identification.Z_selection import select_good_Z
from wprime_plus_b.object_identification.met_selection import select_good_met
from wprime_plus_b.object_identification.jet_selection import select_good_jets
from wprime_plus_b.object_identification.tau_selection import select_good_taus
from wprime_plus_b.object_identification.muon_selection import select_good_muons
from wprime_plus_b.object_identification.bjet_selection import select_good_bjets
from wprime_plus_b.object_identification.electron_selection import select_good_electrons


# =======================================
# General cuts
# =======================================
from wprime_plus_b.general_selections.HEM import get_HEM_cleaning
from wprime_plus_b.general_selections.lumi_mask import get_lumi_mask
from wprime_plus_b.general_selections.Stitching import get_stitching_mask
from wprime_plus_b.general_selections.jetvetomaps import jetvetomaps_mask
from wprime_plus_b.general_selections.met_filters import get_met_filters_mask
from wprime_plus_b.general_selections.good_vertex import get_good_vertex_mask
from wprime_plus_b.general_selections.triggers import get_trigger_mask, get_trigger_match_mask



# =======================================
# Systematic variations
# =======================================
from wprime_plus_b.processors.utils.utils_syst_var import Systematics


# =======================================
#  Plots
# =======================================
from wprime_plus_b.processors.utils.histograms import Histograms
from wprime_plus_b.processors.utils.histogram_2D import ISR_boost_plot

# =======================================
# Utils
# =======================================
from wprime_plus_b.processors.utils.analysis_utils import cross_cleaning, fill_sumw, fill_cutflow, delta_r_mask, get_mask_until_object 



class ZToLLProcessor(processor.ProcessorABC):
    def __init__(
        self,
        processor: str = "top_tagger",
        channel: str = "2b1l",
        lepton_flavor: str = "tau",
        year: str = "2017",
        syst: str = "nominal",
        output_type: str = "array",
        run_systematics: str = "false",
        qcd_data_driven: str = "false",
        output_folder: str = "",
        unblinded: str = "false",
        qcd_cr_B_TF_estimation: str = "false"        
    ):

        self.year = year
        self.lepton_flavor = lepton_flavor
        self.syst = (run_systematics == "true")
        self.output_type = output_type
        self.processor = processor        

        # define dictionary to store analysis variables
        self.features = {}
        # initialize dictionary of arrays
        self.array_dict = {}        

        if unblinded == "true":
            self.unblinded = True 
        else:
            self.unblinded = False


        # Load event selection criteria
        with open(f"wprime_plus_b/selection_criteria/{self.processor}/event_selection_criteria.yaml") as f:
            self.criteria = yaml.safe_load(f)


    def add_feature(self, name: str, var: ak.Array) -> None:
        """add a variable array to the out dictionary"""
        self.features = {**self.features, name: var}        

    def process(self, events):
        # Save the event index as events.event_index
        events["event_index"] = ak.local_index(events, axis=0)
        # get dataset name
        dataset = events.metadata["dataset"]
        # get number of events before selection
        nevents = len(events)
        # check if sample is MC
        self.is_mc = hasattr(events, "genWeight")

        # dictionary to store output data and metadata
        output = {}
        output["metadata"] = {}
        output["hist"] = {}
       
        output["metadata"].update({"raw_initial_nevents": nevents})

        # -------------------------------------------------------------
        #       Object corrections: 
        #   - JES & JER affect JETs, FatJets, and MET
        #   - Rochester affects Muons and MET.
        #   - TES affects Taus and MET.
        #   - MET_XY affects MET
        #   - MET Unclustered
        # -------------------------------------------------------------
        
        # AK4 jets (events.Jet) and MET type-I
        apply_jet_corrections(
            events=events,
            year=self.year,
            syst_var=self.syst,
            jet_case="AK4"      # events.JET
        )
        

        # Muons (events.Muon) Rochester corrections
        apply_rochester_corrections(
            events=events, 
            year=self.year,
            syst_var=self.syst
        )  
        

        # TES (events.Tau) Tau Energy Correction
        apply_tau_energy_scale_corrections(
            events=events, 
            year=self.year, 
            syst_var=self.syst
        )                

        # met_xy corrections (events.MET)
        apply_met_phi_corrections(
            events=events,
            year=self.year,
            syst_var=self.syst
        ) 

        # MET unclustered (events.MET) -> Systematic variations
        apply_met_unclustered(
            events=events,
            syst_var=self.syst
        )
        
        
        # ===============================================================
        #            Object identification
        # ===============================================================

        # Cross cleaning
        cc = self.criteria["cross_cleaning"][self.lepton_flavor]


         # Select good electrons
        good_electrons_masks = select_good_electrons(
            events=events,
            electron_pt_threshold= self.criteria["electron"][self.lepton_flavor]["pt"],
            electron_eta_threshold = self.criteria["electron"][self.lepton_flavor]["eta"],
            electron_id_wp= self.criteria["electron"][self.lepton_flavor]["id"],
            electron_iso_wp=self.criteria["electron"][self.lepton_flavor]["iso"],
            syst_var=self.syst
        )
        good_electrons = good_electrons_masks["nominal"]
        electrons = events.Electron[good_electrons]



        # select good muons
        good_muons_masks = select_good_muons(
            events=events,
            muon_pt_threshold = self.criteria["muon"][self.lepton_flavor]["pt"],
            muon_eta_threshold = self.criteria["muon"][self.lepton_flavor]["eta"],
            muon_id_wp = self.criteria["muon"][self.lepton_flavor]["id"],
            muon_iso_wp = self.criteria["muon"][self.lepton_flavor]["iso"],
            syst_var=self.syst
        )
        good_muons = (good_muons_masks["nominal"])
        muons = events.Muon[good_muons]

        
        # select good taus
        good_taus_masks = select_good_taus(
            events=events,
            tau_pt_threshold = self.criteria["tau"][self.lepton_flavor]["pt"],
            tau_eta_threshold = self.criteria["tau"][self.lepton_flavor]["eta"],
            tau_dz_threshold = self.criteria["tau"][self.lepton_flavor]["dz"],
            tau_vs_jet_pass = self.criteria["tau"][self.lepton_flavor]["fake_VSjet_pass"],
            tau_vs_ele =self.criteria["tau"][self.lepton_flavor]["fake_VSe"],
            tau_vs_mu = self.criteria["tau"][self.lepton_flavor]["fake_VSmu"],
            prong = self.criteria["tau"][self.lepton_flavor]["prongs"] ,
            syst_var=self.syst
        )
        good_taus = good_taus_masks["nominal"] 
        taus = events.Tau[good_taus]        


        # select good bjets
        good_bjets_masks = select_good_bjets(
            events=events,
            year=self.year,
            btag_working_point_pass = self.criteria["bjet"][self.lepton_flavor]["btag_wp_pass"],
            bjet_pt_threshold = self.criteria["bjet"][self.lepton_flavor]["pt"],
            bjet_eta_threshold = self.criteria["bjet"][self.lepton_flavor]["eta"],
            bjet_id_wp = self.criteria["bjet"][self.lepton_flavor]["jet_id"],
            bjet_pileup_id = self.criteria["bjet"][self.lepton_flavor]["pileup_id"],
            syst_var=self.syst
        )
        good_bjets = good_bjets_masks["nominal"] 
        bjets = events.Jet[good_bjets]
        

        # Jets candidates
        good_jets_masks = select_good_jets(
            events=events,
            year=self.year,
            jet_pt_threshold = self.criteria["jet"][self.lepton_flavor]["pt"],
            jet_eta_threshold = self.criteria["jet"][self.lepton_flavor]["eta"],
            jet_id_wp = self.criteria["jet"][self.lepton_flavor]["jet_id"],
            jet_pileup_id = self.criteria["jet"][self.lepton_flavor]["pileup_id"],
            syst_var=self.syst
        )
        good_jets = good_jets_masks["nominal"]
        jets = events.Jet[good_jets]
        
        
        good_met_masks = select_good_met(
            events = events,
            met_min = 0.0,
            year = self.year,
            syst_var=self.syst
        )

        # -----------------------------------------------------
        # Create a dictionary of objects to simplify handling
        # -----------------------------------------------------
        objects_tmp = {
            "bjets": bjets,
            "jets": jets,
            "electrons": electrons,
            "muons": muons,
            "taus": taus,
            "met": events.MET,
            "events": events
        }

        # ----------------
        # cross cleaning
        # ----------------
        objects = cross_cleaning(objects_tmp, cc)


        # Z boson
        Z_masks, objects = select_good_Z(
            objects=objects,
            lepton_flav=self.lepton_flavor,
            mass_min=self.criteria["Z"][self.lepton_flavor]["m_Z_min"],
            mass_max=self.criteria["Z"][self.lepton_flavor]["m_Z_max"],
            charge_selection=self.criteria["Z"][self.lepton_flavor]["charge"],
            cross_cleaning=cc,
            syst_var=self.syst
        )

        if self.syst and self.is_mc:
            objects_variations = {
                "bjet": good_bjets_masks,
                "jet": good_jets_masks,
                "electron": good_electrons_masks,
                "muon": good_muons_masks,
                "tau": good_taus_masks,
                "met": good_met_masks,
                "Z_boson": Z_masks
            }

        # ==============================================================
        #           Event-level weights
        # ==============================================================
        # set weights container
        weights_container = Weights(len(events), storeIndividual=True)

        if self.is_mc:
            # add gen weigths
            add_genweight_weight(events, weights_container)

            # add l1prefiring weigths
            add_l1prefiring_weight(events, weights_container, self.year)

            # add ps weigths
            add_particle_shower_weight(events, weights_container, self.year)

            # add pdf weigths: Remover output de pdf
            delta_pdf, pdf_weight_nominal = add_pdf_weight(events, weights_container, output, self.year)            

            # add top pt reweighting
            add_TopPtReweighting(events, weights_container, dataset, self.year)     

            # add pileup weigths
            add_pileup_weight(events, weights_container, self.year)


        else:
            delta_pdf = pdf_weight_nominal = ak.ones_like(events.MET.pt)

        objects["pdf_nominal"] = objects_tmp["pdf_nominal"] = pdf_weight_nominal
        objects["pdf_ratio"] = objects_tmp["pdf_ratio"] = delta_pdf/pdf_weight_nominal

        # =============================================================================
        #                    Event selection
        # =============================================================================    
        # make a PackedSelection object to store selection masks
        self.selections = PackedSelection(dtype='uint64')        

        # -------------------------
        #  Luminosity
        # -------------------------
        # add luminosity calibration mask (only to data)
        lumi_mask = get_lumi_mask(events = events , year = self.year)
        self.selections.add("lumi", lumi_mask)

        # --------------------------
        #  MET filters
        # -------------------------
        met_filters_mask =  get_met_filters_mask(events = events, year = self.year, is_mc = self.is_mc)
        self.selections.add("metfilters", met_filters_mask)


        # --------------------------
        #  Good vertex
        # -------------------------
        good_vertex_mask = get_good_vertex_mask(events = events)
        self.selections.add("goodvertex", good_vertex_mask)

        # ---------------------------
        # Jet veto
        # ---------------------------
        jet_veto_mask = jetvetomaps_mask(events, self.year)
        self.selections.add("veto_map", jet_veto_mask)

        # -------------------------------
        #        Trigger: OR 
        # -------------------------------
        trigger_names, trigger_mask = get_trigger_mask(
            events=events,
            lepton_flavor = self.lepton_flavor,
            year = self.year,
            trigger_case = self.criteria["trigger"][self.lepton_flavor]["main"],
            Or_HLT = self.criteria["trigger"][self.lepton_flavor]["OR_trigger"]
        )

        output["metadata"].update({"Triggers": trigger_names})
        print(trigger_mask)
        self.selections.add("trigger", trigger_mask)



        # -------------------------------
        #       Trigger Match: OR 
        #   Only to Muons and Electrons
        # -------------------------------
        trigger_match_mask = get_trigger_match_mask(
            objects=objects,
            year=self.year,    
            lepton_flavor=self.lepton_flavor,
            trigger_names=trigger_names
        )

        self.selections.add("trigger_match", trigger_match_mask)

        # --------------------------
        #     HEM cleaning
        # -------------------------        
        HEM_cleaning_mask = get_HEM_cleaning(events = events, 
            jets=events.Jet,
            electrons=events.Electron,
            year=self.year
        )
        self.selections.add("HEMCleaning", HEM_cleaning_mask)

        # --------------------------
        #     Stitiching  
        # -------------------------
        stitching_mask = get_stitching_mask(events= events, dataset_name=dataset)        
        self.selections.add("Stitching", stitching_mask)
        
        # --------------------------
        #  Z boson
        # --------------------------
        self.selections.add("Z_boson", Z_masks["nominal"])
        

        # --------------------------
        #  Number of leptons and jets
        # -------------------------
        # add number of leptons and jets
        self.selections.add("one_electron", ak.num(objects_tmp["electrons"]) == 1)
        self.selections.add("electron_veto", ak.num(objects["electrons"]) == 0)
        self.selections.add("two_electrons", ak.num(objects["electrons"]) == 2)

        self.selections.add("one_muon", ak.num(objects["muons"]) == 1)
        self.selections.add("muon_veto", ak.num(objects["muons"]) == 0)
        self.selections.add("two_muons", ak.num(objects["muons"]) == 2)

        self.selections.add("one_tau", ak.num(objects["taus"]) == 1)            
        self.selections.add("tau_veto", ak.num(objects["taus"]) == 0)
        self.selections.add("two_taus", ak.num(objects["taus"]) == 2)

        self.selections.add("one_bjet", ak.num(objects["bjets"]) == 1)
        self.selections.add("bjet_veto", ak.num(objects["bjets"]) == 0)

        
        # ====================================================
        #     Define selection regions for each channel
        # ===================================================
        region_selection = {
            "tau": [
                "goodvertex",
                "veto_map",
                "HEMCleaning",                
                "metfilters",
                "lumi",
                "Stitching",
                "trigger_match",
                "trigger",                
                "electron_veto",
                "muon_veto",
                "bjet_veto",
                "two_taus",
                "Z_boson"
            ],
            "mu": [
                "goodvertex",
                "veto_map",
                "HEMCleaning",                
                "metfilters",
                "Stitching",
                "lumi",
                "trigger_match",
                "trigger",
                "electron_veto",
                "tau_veto",
                "bjet_veto",
                "two_muons",
                "Z_boson"                
            ],
        }             


        # -------------------------
        # Object ID corrections
        # -------------------------
        # top boost weights
        if self.is_mc:
            has_bjet = any("bjet" in cut for cut in region_selection[self.lepton_flavor])

            # ************************************
            #          Pileup Jet ID
            # ************************************
            add_pujetid_weight(
                jets=objects["jets"],
                weights=weights_container,
                year=self.year,
                working_point = self.criteria["jet"][self.lepton_flavor]["pileup_id"],
                jet_mask = get_mask_until_object(selections=self.selections, cuts=region_selection[self.lepton_flavor], obj_name="bjet" if has_bjet else "none", include_cut=False, only_cut=True)
            )

            # ************************************
            #              Bjets
            # ************************************
            btag_corrector = BTagCorrector(
                jets=objects["jets"],
                dataset = dataset,
                year=self.year,
                tagger="deepJet",                
                weights=weights_container,
                pass_working_point=self.criteria["bjet"][self.lepton_flavor]["btag_wp_pass"],
                bjet_mask = get_mask_until_object(selections=self.selections, cuts=region_selection[self.lepton_flavor], obj_name="bjet" if has_bjet else "none", include_cut=False, only_cut=True),
                mask_eff_btag = get_mask_until_object(selections=self.selections, cuts=region_selection[self.lepton_flavor], obj_name="bjet" if has_bjet else "none", include_cut=False, only_cut=False)
            )            
            # add b-tagging weights
            btag_corrector.add_btag_weights(flavor="bc", correlated=False)
            btag_corrector.add_btag_weights(flavor="light", correlated=False)
            

            # ************************************
            # Electrons
            # ************************************
            electron_corrector = ElectronCorrector(
                electrons=objects["electrons"],
                weights=weights_container,
                year=self.year,
                electron_mask = get_mask_until_object(selections=self.selections, cuts=region_selection[self.lepton_flavor], obj_name="electron", include_cut=False, only_cut=True)
            )

            # add electron ID weights
            electron_corrector.add_id_weight(
                id_working_point = self.criteria["electron"][self.lepton_flavor]["id"]
            )

            # add electron reco weights
            electron_corrector.add_reco_weight("Above20")
            electron_corrector.add_reco_weight("Below20")


            # ************************************
            #               Muons
            # ************************************
            muon_corrector = MuonCorrector(
                muons=objects["muons"],
                weights=weights_container,
                year=self.year,
                id_wp = self.criteria["muon"][self.lepton_flavor]["id"],
                iso_wp = self.criteria["muon"][self.lepton_flavor]["iso"],
                pt_range = "MediumPt",
                muon_mask = get_mask_until_object(selections=self.selections, cuts=region_selection[self.lepton_flavor], obj_name="muon", include_cut=False, only_cut=True)
            )

            # add muon RECO weights
            muon_corrector.add_reco_weight()
            # add muon ID weights
            muon_corrector.add_id_weight()
            # add muon iso weights
            muon_corrector.add_iso_weight()


            # ************************************
            # Hadronic taus
            # ************************************
            tau_corrector = TauCorrector(
                taus=objects["taus"],
                weights=weights_container,
                year=self.year,
                tau_vs_jet = self.criteria["tau"][self.lepton_flavor]["fake_VSjet_pass"],
                tau_vs_ele = self.criteria["tau"][self.lepton_flavor]["fake_VSe"],
                tau_vs_mu = self.criteria["tau"][self.lepton_flavor]["fake_VSmu"],
                tau_mask = get_mask_until_object(selections=self.selections, cuts=region_selection[self.lepton_flavor], obj_name="tau", include_cut=False, only_cut=True)
            )
            tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()

            
            # ************************************
            #        Trigger weights
            # ************************************
            if self.criteria["trigger"][self.lepton_flavor]["main"] == "dimuon":
                muon_corrector.add_dimuon_trigger_weight(
                    trigger_mask=trigger_mask,
                    trigger_match_mask=trigger_match_mask
                )


        # ============================================================== 
        #  Save cutflow: Table with nominal values
        # ==============================================================    

        # -------------------------------------------------------------
        # sumw without ID weights
        # -------------------------------------------------------------
        output["metadata"]["main"] = {}
        # Weighted events
        fill_sumw(weights_container, self.is_mc, output["metadata"]["main"])
        output["metadata"]["main"]["weight_statistics"] = {}
        for weight, statistics in weights_container.weightStatistics.items():
            output["metadata"]["main"]["weight_statistics"][weight] = statistics

        
        fill_cutflow(region_selection[self.lepton_flavor], self.selections, "cutflow", output["metadata"]["main"], weights_container.weight())
            
        # =============================================================
        #             Histograms
        # =============================================================
        hist = Histograms(self.lepton_flavor, self.processor, objects, weights_container.weight(), self.selections, region_selection[self.lepton_flavor])
        output["hist"]["main"] = {}
        output["hist"]["main"]["nominal"] = hist.fill_histograms()
        output["hist"]["main"]["nominal"]["sumw_all_weights"] = ak.sum(weights_container.partial_weight(include=["genweight"]))
        output["hist"]["main"]["nominal"]["count"] = 1        


        # -------------------------------------------------------------
        # ISR boost weights: 2D histogram
        # -------------------------------------------------------------
        output["hist"]["main"]["ISR_boost_weight"] = {}
        ISR_boost_plot(
            objects = objects, 
            out = output["hist"]["main"]["ISR_boost_weight"] , 
            mask = self.selections.all(*region_selection[self.lepton_flavor]),
            weights_container = weights_container.weight()
        )

        
        # ============================================================== 
        #                     Systematics variations
        # ============================================================== 
        objects_tmp["events"] = objects["events"]
        if self.syst and self.is_mc:
            systematic_variations = Systematics(
                cc=cc,
                cr=None,
                year=self.year,
                objects=objects_tmp,
                table_name="cutflow",
                criteria=self.criteria,
                processor=self.processor,
                histograms=output["hist"]["main"],
                weights=weights_container,
                selections=self.selections,
                metadata=output["metadata"]["main"],
                variations=objects_variations,
                lepton_flavor=self.lepton_flavor,
                cut_names=region_selection[self.lepton_flavor]
            )

            systematic_variations.object_level()
            systematic_variations.event_level()


        return {dataset: output}
        
    def postprocess(self, accumulator):
        return accumulator 
