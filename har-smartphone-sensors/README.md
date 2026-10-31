# Human Activity Recognition Using Smartphone Sensors

Classifies 6 daily activities — **Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying** — from smartphone accelerometer + gyroscope data, using both classical ML and deep learning.

## Dataset
[UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones): 30 volunteers wearing a Samsung Galaxy S II on the waist, sampled at 50 Hz. Signals are split into 2.56 s windows (128 readings, 50% overlap).

- **10,299 samples** (7,352 train / 2,947 test, split by subject)
- **561 engineered features** (time + frequency domain)
- **9 raw inertial channels** (body acc, gyro, total acc × XYZ)

## Approach
| Pipeline | Input | Models |
|---|---|---|
| Classical ML | 561 features | Logistic Regression, SVM (RBF), Random Forest |
| Deep Learning | Raw 128×9 windows | 1D-CNN + LSTM |

Expected test accuracy: ~95–96% (SVM), ~92–94% (CNN-LSTM). Main confusion is **Sitting vs Standing** — static postures with similar signals.

## Project structure
```
├── src/
│   ├── download_data.py   # fetch & extract dataset
│   ├── data_loader.py     # load features / raw signals
│   ├── train_ml.py        # classical models + confusion matrices
│   ├── train_dl.py        # CNN-LSTM on raw signals
│   └── predict.py         # inference with a saved model
├── notebooks/
│   └── EDA.ipynb          # exploratory analysis
├── models/                # saved models (git-ignored)
├── results/               # scores & plots
├── requirements.txt
└── README.md
```

## Quick start
```bash
git clone https://github.com/QuantumWebber/har-smartphone-sensors.git
cd har-smartphone-sensors
pip install -r requirements.txt

python src/download_data.py   # ~60 MB
python src/train_ml.py        # classical models
python src/train_dl.py        # deep learning model
python src/predict.py --model svm_rbf --n 10
```

## Results
After training, scores land in `results/ml_scores.json` and `results/dl_scores.json`, with confusion matrices in `results/`.

## Author
**Mohammed Afzal** — B.Tech Engineering Physics, DTU
[Portfolio](https://jatinmaggo.vercel.app) · [GitHub](https://github.com/QuantumWebber)

## License
MIT
