import os
import sys
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import onnx
import onnxruntime as ort
import json

from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml import plot_utils, analysis_utils
from hipe4ml_converter.h4ml_converter import H4MLConverter
from parquet_to_root import parquet_to_root

# ========== Configuration ==========
# Parameters for training
N_ESTIMATORS = (20, 1000)
MAX_DEPTH = (2, 6)
LEARNING_RATE = (0.01, 0.1)
N_JOBS = 10
TIMEOUT=60
N_TRIALS=100

# pT and centrality bins and type for cut and JSON
PT_BINS = [(0, 2), (2, 100)]  # pT bins
CENT_BINS = [(0, 100)]  # centrality bins
CENT_Type = "kCentFT0C" # kCentFT0C, kCentFT0A, kCentFT0M

# Input tree files (from dielectronall table (and ReducedEventsExtended for Cent))
DATA_FILE = "treeData.root"
MC_FILE = "treeMC.root"

# Output directory
OUT_DIR = "251110_bdtModel"
Fig_DIR = "figBinary"
os.makedirs(Fig_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True) 

# Parameters for significance calculation
LUMI = 1706952.6 # Integrated luminosity in microbarn^-1 for pp TRD triggered data
XSEC_MUB = 0.944 # J/psi cross section in microbarn for 2 < pT < 8 GeV/c and |y|<0.9
ACC_EFF = 0.013 # Acceptance times efficiency of Jpsi from TRD analysis in 2 < pT < 8 GeV/c and |y|<0.9
BR_JPSI = 0.0597 # Branching ratio of Jpsi to e+e-
SIG_WINOW = 0.3 # Signal window width in GeV/c^2 [2.9 - 3.1]
deltaY = 1.8 # rapidity range |y| < 0.9
SB_LEFT = (2.6, 2.8)
SB_RIGHT = (3.2, 3.4)

# ========== Utils ==========
def infer_n_features(mh: ModelHandler, input_table=None):
    try:
        return mh.model.n_features_in_
    except AttributeError:
        if input_table is not None:
            return pd.read_parquet(input_table).shape[1]
        raise ValueError("Cannot determine n_features. Provide input table.")

def significance_array(y_score, bkg_mass, signal_efficiencies, lumi, xsec, acc_eff, delta_y, br, sig_win_width, sb_left, sb_right, n_points=100):
    thresholds = np.linspace(np.min(y_score), np.max(y_score), n_points)
    sig_array = []

    for i, thr in enumerate(thresholds[:-1]):
        selected = y_score > thr
        selected_bkg_mass = bkg_mass[selected]

        sideband_mask = ((selected_bkg_mass > sb_left[0]) & (selected_bkg_mass < sb_left[1])) | \
                        ((selected_bkg_mass > sb_right[0]) & (selected_bkg_mass < sb_right[1]))
        bkg_sb = selected_bkg_mass[sideband_mask]
        sb_width = (sb_left[1] - sb_left[0]) + (sb_right[1] - sb_right[0])

        bkg_est_in_window = (len(bkg_sb) / sb_width) * sig_win_width if sb_width > 0 else 0

        signal_eff = signal_efficiencies[i]
        signal_yield = xsec * acc_eff * signal_eff * lumi * delta_y * br

        S = signal_yield
        B = bkg_est_in_window
        if S + B > 0:
            sig = S / np.sqrt(S + B)
        else:
            sig = 0
        
        sig_array.append(sig)

    return np.array(sig_array), thresholds

def export_to_onnx(model_handler, prefix, feature_names, input_table=None):
    n_features = infer_n_features(model_handler, input_table)
    print(f"Inferred number of input features: {n_features}")

    converter = H4MLConverter(model_handler)
    onnx_model = converter.convert_model_onnx(1)

    for i, name in enumerate(feature_names):
        meta = onnx_model.metadata_props.add()
        meta.key = f"feature_{i}"
        meta.value = name

    out_path = f"{prefix}_onnx.onnx"
    converter.dump_model_onnx(out_path)
    print(f"Exported ONNX: {out_path}")

    model = onnx.load(out_path)
    for input_tensor in model.graph.input:
        dims = [d.dim_value if d.HasField("dim_value") else "None" for d in input_tensor.type.tensor_type.shape.dim]
        print(f"  - {input_tensor.name}: shape={dims}")


# ========== Main Training ==========
best_cuts_dict = {}
pre_cut = '1.8 < fMass < 5.0 and -0.4 < fDcaXY1 < 0.4 and -0.4 < fDcaZ1 < 0.4 and -0.4 < fDcaXY2 < 0.4 and -0.4 < fDcaZ2 < 0.4'

for centmin, centmax in CENT_BINS:
    for i, (ptmin, ptmax) in enumerate(PT_BINS):
        print(f"Processing cent {centmin}-{centmax}%, pT bin {ptmin} - {ptmax} GeV/c")
        tag = f'cent{centmin}_{centmax}_pt{ptmin}_{ptmax}'
        #cut_cent = f'{CENT_Type} >= {centmin} and {CENT_Type} < {centmax}'
        cut_pt = f'fPt >= {ptmin} and fPt < {ptmax}'
        #cut = f'{pre_cut} and {cut_cent} and {cut_pt}'
        cut = f'{pre_cut} and {cut_pt}'
        
        promptH_origin = TreeHandler(MC_FILE,'O2rtdielectronall')
        dataH = TreeHandler(DATA_FILE,'O2rtdielectronall')

        # Prepare datasets
        promptH = promptH_origin.get_subset(size=10000) # Limit size for faster testing
        promptH.apply_preselections(cut)
        promptH.apply_preselections(f'fMcDecision == 4 and 2.4 < fMass < 3.2')
        dataH = dataH.get_subset(cut)
        bkgH = dataH.get_subset('1.8 < fMass < 2.4 or 3.2 < fMass < 5.0', size=promptH.get_n_cand() * 2) # Neet to optimize size ratio
        train_test_data = train_test_generator([promptH, bkgH], [1, 0], test_size=0.5, random_state=42)
        
        # Plot distributions and correlations before training
        branches_to_remove = ["fSign", "fFilterMap", "fMcDecision", 
            "fDCAxyzTrk0KF", "fDCAxyzTrk1KF", "fDCAxyzBetweenTrksKF",
            "fDCAxyTrk0KF", "fDCAxyTrk1KF", "fDCAxyBetweenTrksKF",
            "fDeviationTrk0KF", "fDeviationTrk1KF", "fDeviationxyTrk0KF", "fDeviationxyTrk1KF",
            "fMassKFGeo", "fChi2OverNDFKFGeo", "fDecayLengthKFGeo", "fDecayLengthOverErrKFGeo",
            "fDecayLengthXYKFGeo", "fDecayLengthXYOverErrKFGeo", "fPseudoproperDecayTimeKFGeo", "fPseudoproperDecayTimeErrKFGeo",
            "fCosPAKFGeo", "fPairDCAxyz", "fPairDCAxy",
            "fDeviationPairKF", "fDeviationxyPairKF",
            "fMassKFGeoTop", "fChi2OverNDFKFGeoTop"]

        vars_to_draw = promptH.get_var_names()

        for b in branches_to_remove:
            if b in vars_to_draw:
                vars_to_draw.remove(b)

        leg_labels = ['background', 'signal']
        plot_utils.plot_distr([bkgH, promptH], vars_to_draw, bins=100, labels=leg_labels, log=True, density=True, figsize=(12, 7), alpha=0.3, grid=False)
        plt.savefig(f'{Fig_DIR}/plot_distr_origin_{tag}.png')

        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
        corr_plots = plot_utils.plot_corr([bkgH, promptH], vars_to_draw, leg_labels)
        corr_plots[0].savefig(f'{Fig_DIR}/plot_corr_bkg_origin_{tag}.png', bbox_inches='tight')
        corr_plots[1].savefig(f'{Fig_DIR}/plot_corr_sig_origin_{tag}.png', bbox_inches='tight')

        # Remove variables not used for training and plot distributions and correlations
        branches_to_remove_for_training = ["fEta", "fPhi",
            "fPt1", "fEta1", "fPhi1", "fTPCNClsFound1",
            "fDcaXY1", "fDcaZ1",
            "fTPCNSigmaPr1", "fTPCSignal1", "fTPCNSigmaPi1",
            "fTOFBeta1", "fTOFNSigmaPi1", "fTOFNSigmaPr1", "fTOFNSigmaEl1",
            "fPt2", "fEta2", "fPhi2", "fTPCNClsFound2",
            "fDcaXY2", "fDcaZ2",
            "fTPCNSigmaPr2", "fTPCSignal2", "fTPCNSigmaPi2",
            "fTOFBeta2", "fTOFNSigmaPi2", "fTOFNSigmaPr2", "fTOFNSigmaEl2",
            "fLz", "fLxy", "fTauz", "fTauxy", "fLzCov", "fLxyCov", "fTauzCov", "fTauxyCov"]        

        for b in branches_to_remove_for_training:
            if b in vars_to_draw:
                vars_to_draw.remove(b)

        leg_labels = ['background', 'signal']
        plot_utils.plot_distr([bkgH, promptH], vars_to_draw, bins=100, labels=leg_labels, log=True, density=True, figsize=(12, 7), alpha=0.3, grid=False)
        plt.savefig(f'{Fig_DIR}/plot_distr_pt{ptmin}_{ptmax}.png')

        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
        corr_plots = plot_utils.plot_corr([bkgH, promptH], vars_to_draw, leg_labels)
        corr_plots[0].savefig(f'{Fig_DIR}/plot_corr_bkg_{tag}.png', bbox_inches='tight')
        corr_plots[1].savefig(f'{Fig_DIR}/plot_corr_sig_{tag}.png', bbox_inches='tight')

        # Remove mass and pT from features
        features_for_train = vars_to_draw.copy()
        features_for_train.remove('fMass')
        features_for_train.remove('fPt')

        # Train model
        model_clf = xgb.XGBClassifier()
        model_hdl = ModelHandler(model_clf, features_for_train)

        hyper_pars_ranges = {'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH, 'learning_rate': LEARNING_RATE}
        model_hdl.optimize_params_optuna(train_test_data, hyper_pars_ranges, cross_val_scoring='roc_auc', timeout=TIMEOUT, n_jobs=N_JOBS, n_trials=N_TRIALS, direction='maximize')
        model_hdl.train_test_model(train_test_data)

        # Calculate significance
        y_pred_train = model_hdl.predict(train_test_data[0], False)
        y_pred_test = model_hdl.predict(train_test_data[2], False)

        dataH.apply_model_handler(model_hdl, False)
        bdt_scores = dataH.get_data_frame()["model_output"].to_numpy()
        mass_after_cut = dataH.get_data_frame()["fMass"].to_numpy()
        efficiency_array, _ = analysis_utils.bdt_efficiency_array(train_test_data[3], y_pred_test, n_points=100)
        sig_array, thresholds = significance_array(bdt_scores, mass_after_cut, efficiency_array, LUMI, XSEC_MUB, ACC_EFF, deltaY, BR_JPSI, SIG_WINOW, SB_LEFT, SB_RIGHT, n_points=100)

        # Find best cut
        best_idx = np.argmax(sig_array)
        best_cut = thresholds[best_idx]
        best_sig = sig_array[best_idx]

        print(f"Best cut for pT {ptmin}-{ptmax}: {best_cut:.3f}, Significance = {best_sig:.2f}")
        
        plt.figure()
        plt.plot(thresholds[:-1], sig_array, marker='o')
        plt.xlabel("BDT Score Threshold")
        plt.ylabel("Significance (S / √(S + B))")
        plt.grid(True)
        plt.title("Significance vs BDT Score Cut")
        plt.axvline(best_cut, color='r', linestyle='--', label=f"Best cut = {best_cut:.2f}")
        plt.legend()
        plt.savefig(f"{Fig_DIR}/significance_vs_cut_{tag}.png", dpi=300)

        # Plot distributions after applying best cut
        selected_data_hndl = dataH.get_subset(f"model_output > {best_cut:.2f}")
        best_cuts_dict[f"{centmin}_{centmax}_{ptmin}_{ptmax}"] = best_cut

        labels_list = ["after selection", "before selection"]
        colors_list = ['orangered', 'cornflowerblue']
        plot_utils.plot_distr([selected_data_hndl, dataH], column='fMass', bins=100, labels=labels_list, colors=colors_list, density=True, fill=True, histtype='step', alpha=0.5)
        plt.savefig(f'{Fig_DIR}/fMass_{tag}.png', dpi=300)
        labels_list2 = ["after selection"]
        colors_list2 = ['orangered']
        plot_utils.plot_distr(selected_data_hndl, column='fMass', bins=50, labels=labels_list2, colors=colors_list2, density=False, fill=True, histtype='step', alpha=0.5)
        plt.savefig(f'{Fig_DIR}/fMassTrue_{tag}.png', dpi=300)
        plot_utils.plot_distr([selected_data_hndl, dataH], column='fMass', bins=50, labels=labels_list, colors=colors_list, density=False, fill=True, histtype='step', alpha=0.5)
        plt.savefig(f'{Fig_DIR}/fMassTogetherTrue_{tag}.png', dpi=300)
        plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, False, labels_list, True, density=True)
        plt.savefig(f'{Fig_DIR}/ml_out_fig_{tag}.png', dpi=300)
        plot_utils.plot_roc_train_test(train_test_data[3], y_pred_test,train_test_data[1], y_pred_train, None, labels_list)
        plt.savefig(f'{Fig_DIR}/roc_train_test_fig_{tag}.png', dpi=300)
        plot_utils.plot_precision_recall(train_test_data[3], y_pred_test, labels_list)
        plt.savefig(f'{Fig_DIR}/precision_recall_{tag}.png', dpi=300)
        eff, thr = analysis_utils.bdt_efficiency_array(train_test_data[3], y_pred_test, n_points=10)
        plot_utils.plot_bdt_eff(thr, eff)
        plt.savefig(f'{Fig_DIR}/bdt_efficiency_{tag}.png', dpi=300)
        plot_utils.plot_feature_imp(train_test_data[2], train_test_data[3], model_hdl)
        plt.savefig(f'{Fig_DIR}/feature_importance_{tag}.png', dpi=300)

        # Save parquet, root and pkl
        parquet_path = f'{OUT_DIR}/bdt_{tag}.parquet.gzip'
        df = dataH.get_data_frame()
        df.to_parquet(parquet_path, index=False, compression='gzip')
        parquet_to_root(parquet_path, f'{OUT_DIR}/bdt_{tag}.root', treename='parquettree', verbose=False)
        model_hdl.dump_model_handler(f'{OUT_DIR}/modelBDT_{tag}.pkl')

        # Export to ONNX
        features_onnx = [f.replace("f", "k", 1) if f.startswith("f") else f for f in features_for_train]
        export_to_onnx(model_hdl, prefix=f"{OUT_DIR}/modelBDT_{tag}", feature_names=features_onnx, input_table=parquet_path)

print("Training and evaluation completed. Start JSON export.")

# ========== JSON Export ==========
json_dict = {
    "TestCut": {
        "type": "Binary",
        "title": "MyBDTModel",
        "inputFeatures": features_onnx,
        "modelFiles": [],
        "cent": CENT_Type
    }
}

for centmin, centmax in CENT_BINS:
    cent_key = f"AddCentCut-Cent{centmin:02d}{centmax:02d}"
    json_dict["TestCut"][cent_key] = {
        "centMin": centmin,
        "centMax": centmax
    }
    for i, (ptmin, ptmax) in enumerate(PT_BINS):
        pt_key = f"AddPtCut-pTBin{i+1}"
        model_name = f"cent_{centmin}_{centmax}_pt{ptmin}_{ptmax}_onnx.onnx"
        cut_value = best_cuts_dict[f"{centmin}_{centmax}_{ptmin}_{ptmax}"]

        json_dict["TestCut"]["modelFiles"].append(model_name)
        json_dict["TestCut"][cent_key][pt_key] = {
            "pTMin": ptmin,
            "pTMax": ptmax,
            "AddMLCut-background": {
                "var": "kBdtBackground",
                "cut": cut_value,
                "exclude": False
            }
        }

# Save JSON file
json_output_path = os.path.join(OUT_DIR, "bdt_config.json")
with open(json_output_path, "w") as f_json:
    json.dump(json_dict, f_json, indent=2)

with open(json_output_path, "r") as f_json_read:
    data = json.load(f_json_read)

hyperloop_str = json.dumps(data, separators=(',', ':'))
print("Hyperloop JSON string:\n")
print(hyperloop_str)

escaped_str = json.dumps(hyperloop_str)
print("\nLocal JSON string:\n")
print(escaped_str)

print(f"\nJSON saved to {json_output_path}")