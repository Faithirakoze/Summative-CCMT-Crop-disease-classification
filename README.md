# Crop Pest and Disease Classification (CCMT): Traditional ML vs. Deep Learning


This project classifies photos of crop leaves into 22 crop-and-condition classes (for example `Maize_leaf_blight` or `Cassava_mosaic`) and compares two approaches on identical data splits: a classical pipeline (hand-crafted features with Scikit-Learn classifiers) and deep learning (a small CNN trained from scratch and a fine-tuned MobileNetV2).

## Links
- Video presentation: https://youtu.be/_hbEPaRXxcw?si=Db0xKPsEdQQjh8Zn
- Google Drive folder (raw dataset, saved models, TensorBoard logs, result tables, figures): (https://drive.google.com/drive/folders/1aU7QZwTk1ftpWaRlurVa25g-81sdFEOo?usp=sharing)

## Dataset
CCMT, *Dataset for Crop Pest and Disease Detection* (Mendeley Data, DOI [10.17632/bwh3zbpkpv.1](https://doi.org/10.17632/bwh3zbpkpv.1)): leaf photos of cashew, cassava, maize and tomato collected on farms in Ghana. Only the raw images are used. The raw folder holds 25,220 images (the dataset paper reports 24,881); 50 unreadable files were removed, leaving 25,170 images in 22 classes. The largest class is about 13 times the size of the smallest. The dataset is not included in this repository (see "Reproducing the results").

## Results (test set, evaluated once)

| Model | Accuracy | Macro-F1 | ROC-AUC | Avg. precision | False alarms on healthy leaves |
|---|---|---|---|---|---|
| Random Forest (tuned) | 0.701 | 0.685 | 0.972 | 0.731 | 24.3% |
| MobileNetV2 (fine-tuned) | 0.872 | 0.865 | 0.995 | 0.916 | 5.2% |

Validation macro-F1 for all models:

| Model | Val macro-F1 |
|---|---|
| Logistic Regression (best C = 0.01) | 0.525 |
| SVM (RBF, untuned) | 0.593 |
| Random Forest (tuned) | 0.685 |
| Baseline CNN (two runs) | 0.737 and 0.696 |
| MobileNetV2, frozen base | 0.790 |
| MobileNetV2, fine-tuned (class weights) | 0.864 |
| MobileNetV2, fine-tuned (no class weights) | 0.868 |

The two test-set models were fixed beforehand, using validation macro-F1 within each family. Main observations:
- On validation, only 3.4% of the class-weighted fine-tuned MobileNetV2's errors cross crops, against 40.1% for the baseline Random Forest. Its remaining errors are mostly among similar-looking tomato diseases.
- Class weighting changed which classes were confused more than how many errors occurred (validation, one run per setting).
- Training the same CNN twice gave macro-F1 scores about 4 points apart, so small differences between single runs should not be over-interpreted.

## Repository contents
- `Faith_Irakoze_Crop_Disease_Classification.ipynb`: the full pipeline (data loading, cleaning, split, features, classical models, deep models, error analysis, test evaluation, experiment table)
- `requirements.txt`: Python dependencies
- `README.md`: this file

Datasets, model weights, TensorBoard logs, result tables and figures are on Google Drive (link above), not in this repository.

## Reproducing the results
The notebook was developed in Google Colab (Python [confirm version, 3.13 in my sessions]).

1. Download the dataset from the Mendeley link above and zip the `Raw Data/CCMT Dataset` folder as `ccmt_raw.zip`.
2. Create `My Drive/ccmt_project/` and upload the zip there. The path is set by `DRIVE_DIR` in the setup cell.
3. Open the notebook in Colab: [Open in Colab](https://colab.research.google.com/github/[username]/[repo]/blob/main/Faith_Irakoze_Crop_Disease_Classification.ipynb)
4. Install dependencies if needed: `pip install -r requirements.txt`.
5. Use a CPU runtime for Sections 1 to 6 and a GPU runtime (T4) from Section 7, then run the cells in order.

Expensive steps (unreadable-file scan, data split, feature extraction, scikit-learn models) are cached on Drive and reloaded on re-runs. The deep learning cells retrain from scratch and take roughly an hour in total (CNN about 38 min, MobileNetV2 head about 7 min, fine-tuning about 12 min, the no-class-weights run about 17 min), and the Random Forest search takes about 25 minutes.

**Reproducibility notes.** All seeds are 42 and the split files are saved on Drive. Deep learning results are not bit-for-bit reproducible: expect small differences when retraining. TensorBoard logs for the classical runs and for the first CNN run were written after the fact from saved results; the later runs were logged live.

## Known limitations
One run per setting (two for the CNN), a possible overlap of near-duplicate images across splits, small test classes (31 to 59 images), ImageNet pretraining that the classical pipeline lacks, and images from a single country. Forty-six images that TensorFlow's decoder rejects were kept and read with OpenCV so that all models use identical splits.

## Acknowledgements
- Dataset: P. K. Mensah et al., "CCMT: Dataset for crop pest and disease detection," *Data in Brief*, vol. 49, art. 109306, 2023. [Check and state the dataset's license from its Mendeley page.]
- Pretrained model: MobileNetV2 with ImageNet weights, from `keras.applications`.
- AI assistance: I used AI tools to help me understand machine learning concepts, troubleshoot code, and improve the structure and clarity of my report and presentation.