import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

class ForestCoverClassifier:
    def __init__(self, data_path=None, dataframe=None):
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        
        if dataframe is not None:
            self.data = dataframe.copy()
        elif data_path:
            self.load_data(data_path)
        
        self.cover_types = {
            1: 'Spruce/Fir',
            2: 'Lodgepole Pine', 
            3: 'Ponderosa Pine',
            4: 'Cottonwood/Willow',
            5: 'Aspen',
            6: 'Douglas-fir',
            7: 'Krummholz'
        }
    
    def load_data(self, data_path):
        try:
            self.data = pd.read_csv(data_path)
            print(f"Dataset loaded successfully. Shape: {self.data.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("=== DATASET OVERVIEW ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n=== COLUMN INFORMATION ===")
        print(self.data.info())
        
        print("\n=== STATISTICAL SUMMARY ===")
        print(self.data.describe())
        
        print("\n=== MISSING VALUES ===")
        missing_vals = self.data.isnull().sum()
        if missing_vals.sum() == 0:
            print("No missing values found!")
        else:
            print(missing_vals[missing_vals > 0])
        
        target_col = self.data.columns[-1]
        print(f"\n=== TARGET DISTRIBUTION ({target_col}) ===")
        target_counts = self.data[target_col].value_counts().sort_index()
        print(target_counts)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        target_counts.plot(kind='bar')
        plt.title('Cover Type Distribution')
        plt.xlabel('Cover Type')
        plt.ylabel('Count')
        plt.xticks(range(len(self.cover_types)), 
                  [self.cover_types[i+1] for i in range(len(self.cover_types))], 
                  rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.pie(target_counts.values, labels=[self.cover_types[i+1] for i in range(len(self.cover_types))], 
                autopct='%1.1f%%')
        plt.title('Cover Type Percentage')
        
        plt.subplot(2, 1, 2)
        num_features = self.data.select_dtypes(include=[np.number]).columns[:10]
        corr_matrix = self.data[num_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix (First 10 Features)')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, target_column=None, test_size=0.2, val_size=0.1, random_state=42):
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        if target_column is None:
            target_column = self.data.columns[-1]
        
        print(f"Using '{target_column}' as target variable")
        
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Number of classes: {y.nunique()}")
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples") 
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.y_train_encoded = self.y_train - 1
        self.y_val_encoded = self.y_val - 1
        self.y_test_encoded = self.y_test - 1
        
        self.y_train_cat = to_categorical(self.y_train_encoded)
        self.y_val_cat = to_categorical(self.y_val_encoded)
        self.y_test_cat = to_categorical(self.y_test_encoded)
        
        print(f"\nPreprocessing complete!")
        print(f"Feature scaling applied")
        print(f"Labels encoded to 0-{y.nunique()-1} range")
    
    def build_model(self, architecture='deep', input_dim=None, num_classes=7, 
                   dropout_rate=0.3, learning_rate=0.001):

        if input_dim is None:
            input_dim = self.X_train_scaled.shape[1]
        
        tf.keras.backend.clear_session()
        
        if architecture == 'simple':
            self.model = self._build_simple_model(input_dim, num_classes, dropout_rate)
        elif architecture == 'deep':
            self.model = self._build_deep_model(input_dim, num_classes, dropout_rate)
        elif architecture == 'wide':
            self.model = self._build_wide_model(input_dim, num_classes, dropout_rate)
        else:
            raise ValueError("Architecture must be 'simple', 'deep', or 'wide'")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n{architecture.capitalize()} model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        return self.model
    
    def _build_simple_model(self, input_dim, num_classes, dropout_rate):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def _build_deep_model(self, input_dim, num_classes, dropout_rate):
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def _build_wide_model(self, input_dim, num_classes, dropout_rate):
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def train_model(self, epochs=100, batch_size=32, patience=10, verbose=1):
        if self.model is None:
            print("No model built. Please build a model first.")
            return
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("Starting model training...")
        
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train_cat,
            validation_data=(self.X_val_scaled, self.y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("Training completed!")
        return self.history
    
    def plot_training_history(self):
        if self.history is None:
            print("No training history available. Train a model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, dataset='test'):
        if self.model is None:
            print("No model available. Please train a model first.")
            return
        
        if dataset == 'test':
            X, y_true, y_cat = self.X_test_scaled, self.y_test_encoded, self.y_test_cat
        elif dataset == 'val':
            X, y_true, y_cat = self.X_val_scaled, self.y_val_encoded, self.y_val_cat
        elif dataset == 'train':
            X, y_true, y_cat = self.X_train_scaled, self.y_train_encoded, self.y_train_cat
        else:
            raise ValueError("Dataset must be 'test', 'val', or 'train'")
        
        y_pred_proba = self.model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"=== {dataset.upper()} SET EVALUATION ===")
        print(f"Accuracy: {accuracy:.4f}")
        
        target_names = [self.cover_types[i+1] for i in range(7)]
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names,
                   yticklabels=target_names)
        plt.title(f'Confusion Matrix - {dataset.upper()} Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return accuracy, y_pred, y_pred_proba
    
    def hyperparameter_tuning(self, param_grid=None, max_trials=20):
        if param_grid is None:
            param_grid = {
                'architecture': ['simple', 'deep', 'wide'],
                'dropout_rate': [0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.001, 0.005, 0.01],
                'batch_size': [32, 64, 128]
            }
        
        best_accuracy = 0
        best_params = {}
        results = []
        
        print("Starting hyperparameter tuning...")
        
        import itertools
        
        keys, values = zip(*param_grid.items())
        combinations = list(itertools.product(*values))
        
        if len(combinations) > max_trials:
            combinations = combinations[:max_trials]
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\nTrial {i+1}/{len(combinations)}: {params}")
            
            try:
                self.build_model(
                    architecture=params['architecture'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate']
                )
                
                self.train_model(
                    epochs=50, 
                    batch_size=params['batch_size'],
                    patience=5,
                    verbose=0
                )
                
                accuracy, _, _ = self.evaluate_model('val')
                
                results.append({**params, 'val_accuracy': accuracy})
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = params.copy()
                    print(f"New best accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"Error in trial {i+1}: {e}")
                continue
        
        print(f"\n=== HYPERPARAMETER TUNING RESULTS ===")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Best parameters: {best_params}")
        
        results_df = pd.DataFrame(results)
        print(f"\nTop 5 configurations:")
        print(results_df.nlargest(5, 'val_accuracy'))
        
        return best_params, results_df
    
    def compare_with_baseline(self):
        """Compare neural network with Random Forest baseline."""
        print("=== BASELINE COMPARISON (Random Forest) ===")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        
        rf_pred = rf.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)
        
        print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")
        
        if self.model is not None:
            nn_accuracy, _, _ = self.evaluate_model('test')
            print(f"Neural Network Test Accuracy: {nn_accuracy:.4f}")
            
            improvement = ((nn_accuracy - rf_accuracy) / rf_accuracy) * 100
            print(f"Improvement: {improvement:+.2f}%")
        
        return rf_accuracy

def main():
    
    classifier = ForestCoverClassifier(data_path='cover_data.csv')

    classifier.explore_data()
    
    classifier.preprocess_data()
    
    classifier.build_model(architecture='deep')
    classifier.train_model(epochs=100, batch_size=64)
    
    classifier.plot_training_history()
    
    classifier.evaluate_model('test')
    
    classifier.compare_with_baseline()
    
    best_params, results = classifier.hyperparameter_tuning()

if __name__ == "__main__":
    main()