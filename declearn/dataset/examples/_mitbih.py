import logging
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb
from scipy.signal import filtfilt, firwin

logger = logging.getLogger(__name__)

FS = 360 

# All records required by the AAMI EC57 split (DS1 ∪ DS2).
_AAMI_RECORDS = [
    "100", "101", "103", "105", "106", "108", "109", "111", "112", "113",
    "114", "115", "116", "117", "118", "119", "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
    "213", "214", "215", "219", "220", "221", "222", "223", "228", "230",
    "231", "232", "233", "234",
]


def _ensure_downloaded(folder: Optional[str]) -> None:
    """Download any missing MIT-BIH records from PhysioNet."""
    if folder is None:
        return
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    missing = [r for r in _AAMI_RECORDS if not (folder_path / f"{r}.hea").exists()]
    if not missing:
        return
    logger.info(
        "Downloading %d missing MIT-BIH record(s) from PhysioNet: %s",
        len(missing), missing,
    )
    try:
        wfdb.dl_database(
            "mitdb",
            dl_dir=str(folder_path),
            keep_subdirs=False,
            records=missing,
            annotators=["atr"],  # type: ignore[arg-type]
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download MIT-BIH records {missing} to '{folder}'. "
            "Download manually from https://physionet.org/content/mitdb/ "
            "and extract .hea/.dat/.atr files into that folder.\n"
            f"Original error: {exc}"
        ) from exc
    logger.info("Download complete.")


def load_mit_bih(folder: Optional[str]= None, window_len = 64, mode="design"):
    assert mode in ("design", "final"), f"mode must be 'design' or 'final', got {repr(mode)}"

    _ensure_downloaded(folder)
    X_all, y_all, RR_all = _load_mit_bih(folder, window_len)
    
    classes_n = np.unique(np.concatenate(list(y_all.values()), axis=0))
    label_encoder = {val: idx for idx, val in enumerate(classes_n)}
    y_all_encoded = {}
    for patient, patient_labels in y_all.items():
        y_all_encoded[patient] = np.array([label_encoder[val] for val in patient_labels])


    return X_all, y_all_encoded, RR_all



def _load_mit_bih(folder: Optional[str]= None, window_len = 64, extension="atr"):
    PACED_RECORDS = {'102', '104', '107', '217'}
    files = [f for f in Path(folder).iterdir() if f.is_file() and f.suffix == ".hea"]
    y_all = {}
    X_all = {}
    RR_all = {}

    half_window_len = window_len // 2 
    AAMI_MAP = {"N":"N", "L":"N", "R":"N", "e":"N", "j":"N", 
                            "A":"S", "a":"S", "S":"S", "J":"S", 
                            "V":"V", "E":"V", 
                            #"F":"F", #Removed F and Q as there are not relevant to arrhythmia detection  
                            # "/":'Q', "f": "Q", "Q":"Q",
                            }

    beat_symbols = list(AAMI_MAP.keys())



    for f in files:
        if f.stem in PACED_RECORDS:
            continue

        record = wfdb.rdrecord(record_name=f.with_suffix(''))
        annotation = wfdb.rdann(record_name=str(f.with_suffix('')), extension=extension)
        
        clean_signal = _preprocess_ecg(record.p_signal[:, 0])

        in_flutter = False 
        in_flutter_indices = set()
        for i, sym in enumerate(annotation.symbol):
            if sym == '[':
                in_flutter = True 
            elif sym == ']':
                in_flutter = False 
            elif in_flutter:
                in_flutter_indices.add(i)

        valid_beats = [] 
        for i , sample_idx in enumerate(annotation.sample):
            if annotation.symbol[i] not in beat_symbols:
                continue 
            if i in in_flutter_indices:
                continue 
            if  sample_idx-half_window_len < 0 or sample_idx+half_window_len > record.sig_len:
                continue  
            
            valid_beats.append((i, sample_idx, AAMI_MAP[annotation.symbol[i]]))
       
        x, y, rr = [], [], []
        
        for pos, (i, sample_idx, label) in enumerate(valid_beats):
            pre_rr  = (valid_beats[pos][1] - valid_beats[pos-1][1]) / FS if pos > 0 else 0.0
            post_rr = (valid_beats[pos+1][1] - valid_beats[pos][1]) / FS if pos < len(valid_beats)-1 else 0.0
            local_mean_rr = np.mean(np.diff([b[1] for b in valid_beats[max(0, pos-5):pos+1]])) / FS if pos > 0 else pre_rr
            ratio = pre_rr / post_rr if post_rr > 0 else 1.0

            x.append(clean_signal[sample_idx-half_window_len:sample_idx+half_window_len])
            y.append(label)
            rr.append([pre_rr, post_rr, ratio, local_mean_rr])


        X_all[f.stem] = np.expand_dims(np.stack(x), axis=1).astype("float32")
        # X_all[f.stem] = np.permute_dims(np.stack(x), axes=(0,2,1)).astype("float32")
        y_all[f.stem] = np.array(y)
        RR_all[f.stem] = np.array(rr, dtype="float32")


    return X_all, y_all, RR_all



def _preprocess_ecg(signal: np.ndarray) -> np.ndarray:
    """
    4-step FIR Kaiser pipeline from Sravan Kumar et al. (2015):
    1. HPF 0.5 Hz  -> remove baseline wander
    2. BSF 59.5-60.5 Hz -> remove power line interference
    3. LPF 100 Hz  -> remove EMG noise
    4. Moving average -> smooth
    """
    hp = firwin(57, cutoff=0.5, window=('kaiser', 8.6), pass_zero=False, fs=FS)
    signal = filtfilt(hp, 1.0, signal)

    bs = firwin(57, cutoff=[59.5, 60.5], window=('kaiser', 8.6), pass_zero=True, fs=FS)
    signal = filtfilt(bs, 1.0, signal)

    lp = firwin(57, cutoff=100.0, window=('kaiser', 8.6), pass_zero=True, fs=FS)
    signal = filtfilt(lp, 1.0, signal)

    kernel = np.ones(5) / 5
    signal = np.convolve(signal, kernel, mode='same')

    return signal






