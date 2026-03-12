import yaml
import awkward as ak
from hist import Hist
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

# =======================================
# Corrections: object - level
# =======================================
from wprime_plus_b.corrections.jec import apply_jet_corrections
from wprime_plus_b.corrections.rochester import apply_rochester_corrections
from wprime_plus_b.corrections.tau_energy import apply_tau_energy_scale_corrections
from wprime_plus_b.corrections.met import apply_met_unclustered, apply_met_phi_corrections, met_recoil, add_met_trigger_corrections

# =======================================
# Corrections: event-level
# =======================================
from wprime_plus_b.corrections.pileup import add_pileup_weight
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
from wprime_plus_b.object_identification.jet_selection import select_good_jets
from wprime_plus_b.object_identification.tau_selection import select_good_taus
from wprime_plus_b.object_identification.muon_selection import select_good_muons
from wprime_plus_b.object_identification.bjet_selection import select_good_bjets
from wprime_plus_b.object_identification.wjet_selection import select_good_wjets
from wprime_plus_b.object_identification.topjet_selection import select_good_topjets
from wprime_plus_b.object_identification.electron_selection import select_good_electrons
from wprime_plus_b.object_identification.lightjet_selection import select_good_lightjets
from wprime_plus_b.object_identification.met_selection import select_good_delta_phi_jet_met, select_good_met

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

# Top tagger
from wprime_plus_b.processors.utils.utils_topXfinder import get_topXfinder_masks


# =======================================
# Utils
# =======================================
from wprime_plus_b.processors.utils.analysis_utils import cross_cleaning, fill_cutflow, delta_r_mask 



class BTagEfficiencyProcessor(processor.ProcessorABC):
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
        unblinded: str = "false"        
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

        self.qcd_data_driven = (
            qcd_data_driven == "true" and
            all(field in self.criteria.get("data_driven_qcd_estimation", {}) for field in ["cr_b", "cr_c", "cr_d"])
        )
        

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
        
        # AK8 jets (events.FatJet)
        apply_jet_corrections(
            events=events,
            year=self.year,
            syst_var=self.syst,
            jet_case="AK8"      # events.FatJET
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
        

        # Select good MET
        good_met_masks = select_good_met(
            events = events,
            met_min = self.criteria["met"][self.lepton_flavor]["met_min"],
            year = self.year,
            syst_var=self.syst
        )

        # Compute recoil-corrected MET using reconstructed leptons
        met_recoil(events = events, muons = muons)

        # -----------------------------------------------------
        # Create a dictionary of objects to simplify handling
        # -----------------------------------------------------
        objects_tmp = {
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

        # Select delta_phi_cut
        delta_phi_jet_met_masks = select_good_delta_phi_jet_met(
            events = objects["events"], 
            jets = objects["jets"],
            delta_phi_cut = self.criteria["met"][self.lepton_flavor]["delta_phi_jets_met"],
            invert_delta_phi_cut = self.criteria["met"][self.lepton_flavor]["invert_delta_phi"],
            syst_var=self.syst
        )

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

            # add top pt reweighting
            add_TopPtReweighting(events, weights_container, dataset)     

            # add pileup weigths
            add_pileup_weight(events, weights_container, self.year)

            # add pujetid weigths
            add_pujetid_weight(
                jets=objects["jets"],
                weights=weights_container,
                year=self.year,
                working_point = self.criteria["jet"][self.lepton_flavor]["pileup_id"]
            )


            # -----------------------
            # electron corrector
            # -----------------------
            electron_corrector = ElectronCorrector(
                electrons=objects["electrons"],
                weights=weights_container,
                year=self.year,
            )

            # add electron ID weights
            electron_corrector.add_id_weight(
                id_working_point = self.criteria["electron"][self.lepton_flavor]["id"]
            )

            # add electron reco weights
            electron_corrector.add_reco_weight("Above20")
            electron_corrector.add_reco_weight("Below20")


            # -----------------------
            # muon corrector
            # -----------------------
            muon_corrector = MuonCorrector(
                muons=objects["muons"],
                weights=weights_container,
                year=self.year,
                id_wp = self.criteria["muon"][self.lepton_flavor]["id"],
                iso_wp = self.criteria["muon"][self.lepton_flavor]["iso"],
                pt_range = "MediumPt"
            )

            # add muon RECO weights
            muon_corrector.add_reco_weight()
            # add muon ID weights
            muon_corrector.add_id_weight()
            # add muon iso weights
            muon_corrector.add_iso_weight()


            # add tau weights
            tau_corrector = TauCorrector(
                taus=objects["taus"],
                weights=weights_container,
                year=self.year,
                tau_vs_jet = self.criteria["tau"][self.lepton_flavor]["fake_VSjet_pass"],
                tau_vs_ele = self.criteria["tau"][self.lepton_flavor]["fake_VSe"],
                tau_vs_mu = self.criteria["tau"][self.lepton_flavor]["fake_VSmu"]
            )
            tau_corrector.add_id_weight_DeepTau2017v2p1VSe()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSmu()
            tau_corrector.add_id_weight_DeepTau2017v2p1VSjet()


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
        self.selections.add("trigger", trigger_mask)

      
        add_met_trigger_corrections(
            trigger_mask, 
            dataset, 
            objects["met"], 
            weights_container, 
            self.year
        ) 


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
        # deltaphi_cut cut: 
        # --------------------------   
        self.selections.add(f"delta_phi_jet_met", delta_phi_jet_met_masks["nominal"])

        # --------------------------
        #  Number of leptons and jets
        # -------------------------
        # add number of leptons and jets
        self.selections.add("one_electron", ak.num(objects_tmp["electrons"]) == 1)
        self.selections.add("electron_veto", ak.num(objects["electrons"]) == 0)

        self.selections.add("one_muon", ak.num(objects["muons"]) == 1)
        self.selections.add("muon_veto", ak.num(objects["muons"]) == 0)

        self.selections.add("one_tau", ak.num(objects["taus"]) == 1)            
        self.selections.add("tau_veto", ak.num(objects["taus"]) == 0)

        self.selections.add(f"met",  good_met_masks["nominal"])


        # ====================================================
        #     Define selection regions for each channel
        # ===================================================
        region_selection = {
            "tau": [
                "goodvertex",
                "veto_map",
                "HEMCleaning",                
                "Stitching",
                "lumi",
                "trigger",
                "trigger_match",
                "metfilters",
                "electron_veto",
                "muon_veto",
                "one_tau",
                "met",
                "delta_phi_jet_met"
            ],
            "mu": [
                "goodvertex",
                "veto_map",
                "HEMCleaning",                
                "Stitching",
                "lumi",
                "trigger",
                "trigger_match",
                "metfilters",
                "electron_veto",
                "tau_veto",
                "one_muon",
                "met"
            ],
        }      


        # 1. Aplicar la máscara de selección de región (sin b-tagging)
        region_mask = self.selections.all(*region_selection[self.lepton_flavor])
        
        # Filtramos los objetos para los eventos que pasaron
        presel_jets = objects["jets"][region_mask]
        presel_weights = weights_container.weight()[region_mask]

        if self.is_mc:
            # 2. Definir qué jets pasan el Working Point (WP)
            # Obtenemos el threshold del YAML (ej: DeepJet medium)
            btag_wp = self.criteria["bjet"][self.lepton_flavor]["btag_wp_pass"]
            # Nota: Asegúrate de tener btag_thresholds en tu yaml o pasarlo como variable
            threshold = self.criteria["btag_thresholds"][self.year][btag_wp]
            
            pass_btag = presel_jets.btagDeepFlavB > threshold

            # 3. Aplanar (flatten) y preparar para el histograma
            # El peso del evento se repite para cada jet del mismo evento
            flat_weights = ak.flatten(ak.broadcast_arrays(presel_weights, presel_jets.pt)[0])
            flat_pt = ak.flatten(presel_jets.pt)
            flat_eta = ak.flatten(abs(presel_jets.eta))
            flat_flavor = ak.flatten(presel_jets.hadronFlavour)
            flat_pass = ak.flatten(pass_btag)

            # 4. Llenar por sabor
            for flv_code, flv_name in zip([5, 4, 0], ["b", "c", "light"]):
                # Para light, incluimos todo lo que no sea b (5) o c (4)
                f_mask = (flat_flavor == flv_code) if flv_code != 0 else (flat_flavor != 5) & (flat_flavor != 4)
                
                # Denominador (Total de jets del sabor X en la región seleccionada)
                output["hist"].fill(
                    flavor=flv_name, status="total",
                    pt=flat_pt[f_mask], abseta=flat_eta[f_mask], weight=flat_weights[f_mask]
                )
                # Numerador (Jets que además pasaron el WP)
                output["hist"].fill(
                    flavor=flv_name, status="pass",
                    pt=flat_pt[f_mask & flat_pass], abseta=flat_eta[f_mask & flat_pass], weight=flat_weights[f_mask & flat_pass]
                )


        
        pt_bins = [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000]
        # Para eta, el POG dice que depende de la estadística. 
        # Usar 2 o 3 bins suele ser suficiente para capturar la diferencia Barrel/Endcap.
        eta_bins = [0, 1.2, 2.5] 

        self.output_hist = (
            Hist.new
            .StrCat(["b", "c", "light"], name="flavor")
            .StrCat(["pass", "total"], name="status")
            .Variable(pt_bins, name="pt")
            .Variable(eta_bins, name="abseta")
            .Weight()
        )

        
        return {dataset: output}
        
    def postprocess(self, accumulator):
        return accumulator 