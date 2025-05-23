# Dataset Overview and Processing Pipeline

## Dataset Details

* **Records**: 581,012
* **Features**: 54 (excluding the target variable `class`)
* **Memory Usage**: 243.80 MB
* **Target Variable**: `class` (7 classes: 1–7)

## Feature Summary

* Includes topographic (e.g., Elevation, Slope), spatial (e.g., distances to hydrology/roadways/fire points), light-related (Hillshade), and categorical indicators (Wilderness Areas, Soil Types as one-hot encoded).
* All features are integer type.
* **No missing values**.

## Target Class Distribution

```
Class 1: 211,840
Class 2: 283,301
Class 3: 35,754
Class 4: 2,747
Class 5: 9,493
Class 6: 17,367
Class 7: 20,510
```

## Data Preparation Workflow

* **Loading**: Dataset loaded without issues.
* **Label Encoding**: `class` column encoded to 0–6.
* **Feature Scaling**: Applied to ensure normalized input values.
* **Split**:

  * Train: 406,707 samples
  * Validation: 58,102 samples
  * Test: 116,203 samples

## Model Tuning Summary

* **Best Validation Accuracy**: 87.41%
* **Optimal Hyperparameters**:

  * Architecture: `simple`
  * Dropout Rate: 0.2
  * Learning Rate: 0.001
  * Batch Size: 64

### Top Performing Configurations

```
1. Acc: 0.8741 | Dropout: 0.2 | LR: 0.001 | Batch: 64
2. Acc: 0.8725 | Dropout: 0.2 | LR: 0.001 | Batch: 128
3. Acc: 0.8700 | Dropout: 0.2 | LR: 0.001 | Batch: 32
4. Acc: 0.8635 | Dropout: 0.2 | LR: 0.005 | Batch: 128
5. Acc: 0.8529 | Dropout: 0.3 | LR: 0.001 | Batch: 64
```

## Conclusion

The dataset was efficiently preprocessed and trained using a simple neural network architecture. The tuning process yielded strong performance, indicating that even basic models can achieve high accuracy with well-prepared features.
