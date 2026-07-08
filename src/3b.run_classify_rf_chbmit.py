"""Alternative classical classifier: Random Forest on SCTN burstiness features.

Same harness as src/3.run_classify_chbmit.py (per-patient, honest LORO +
stratified reference, confusion matrix) but with a Random Forest instead of the
SVM — a quick check that the FFT-vs-SCTN / seizure-detection story is not an
artifact of one classifier. Writes output_data/results/rf_classifier_report.csv
and rf_confusion_matrix.png.

Usage:
    python -u src/3b.run_classify_rf_chbmit.py [chb01 chb03 ...]

(An SNN classifier would be the natural src/3c on the SNN branch, once we leave
the classical-classifier branch.)
"""
import importlib.util
from pathlib import Path

_SVM = Path(__file__).resolve().parent / "3.run_classify_chbmit.py"
_spec = importlib.util.spec_from_file_location("classify3", _SVM)
_clf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_clf)

if __name__ == "__main__":
    _clf.main(classifier="rf")
