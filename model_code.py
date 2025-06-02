#!/usr/bin/env python3
"""
Real Data Training - Using Your WiFi Dataset
Trains on perfectly_balanced_wifi_dataset.csv to predict actual service usage patterns
Creates minimal files for router deployment
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class RealDataServicePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.service_names = []
        self.service_priorities = {}
        self.temporal_features = []
        
    def load_and_analyze_dataset(self, csv_file='perfectly_balanced_wifi_dataset.csv'):
        """Load your actual dataset and analyze it"""
        print("📂 LOADING YOUR ACTUAL DATASET")
        print("=" * 50)
        
        # Load the dataset
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Convert timestamp to datetime if needed
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"📊 Unique services: {df['service_name'].nunique()}")
        print(f"🎯 Target: optimal_bandwidth_allocation (mean: {df['optimal_bandwidth_allocation'].mean():.2f} Mbps)")
        
        # Analyze services
        service_analysis = df.groupby('service_name').agg({
            'optimal_bandwidth_allocation': ['count', 'mean', 'std', 'max'],
            'service_priority': lambda x: x.iloc[0],  # Get first value instead of first()
            'throughput': 'mean'
        }).round(2)
        
        print(f"\n📈 SERVICE ANALYSIS:")
        print(service_analysis.head(10))
        
        return df
    
    def create_daily_aggregated_data(self, df):
        """Aggregate your data by date and create training samples"""
        print(f"\n🔄 CREATING DAILY AGGREGATED DATA")
        print("=" * 50)
        
        # Add date column
        df['date'] = df['timestamp'].dt.date
        
        # Get unique services from your actual data
        self.service_names = sorted(df['service_name'].unique())
        print(f"📊 Found {len(self.service_names)} unique services in your data:")
        
        # Calculate service priorities from your data
        service_priority_data = df.groupby('service_name')['service_priority'].apply(lambda x: x.iloc[0]).to_dict()
        service_bandwidth_avg = df.groupby('service_name')['optimal_bandwidth_allocation'].mean().to_dict()
        
        for i, service in enumerate(self.service_names[:15]):  # Show top 15
            priority = service_priority_data.get(service, 0)
            avg_bw = service_bandwidth_avg.get(service, 0)
            print(f"   {i+1:2d}. {service:30} (Priority: {priority}, Avg: {avg_bw:.2f} Mbps)")
        
        if len(self.service_names) > 15:
            print(f"   ... and {len(self.service_names) - 15} more services")
        
        # Create daily aggregated dataset
        print(f"\n📈 Aggregating by date...")
        daily_data = []
        unique_dates = sorted(df['date'].unique())
        
        print(f"📅 Processing {len(unique_dates)} unique dates...")
        
        for date in unique_dates:
            day_data = df[df['date'] == date]
            
            # Extract temporal features from your actual data
            first_row = day_data.iloc[0]
            temporal_features = {
                'hour_avg': day_data['hour_of_day'].mean(),
                'day_of_week': first_row['day_of_week'],
                'is_weekend': int(first_row['is_weekend']),
                'is_business_hours': int(day_data['is_business_hours'].any()),
                'is_peak_hours': int(day_data['is_peak_hours'].any()),
                'is_wfh_core_hours': int(day_data['is_wfh_core_hours'].any()),
                'avg_network_utilization': day_data['network_utilization'].mean(),
                'avg_signal_strength': day_data['signal_strength'].mean(),
                'avg_latency': day_data['latency'].mean(),
                'total_devices': day_data['num_connected_devices'].mean(),
                'avg_throughput': day_data['throughput'].mean(),
                'avg_quality_satisfaction': day_data['quality_satisfaction'].mean(),
            }
            
            # Get actual service bandwidth allocations from your data
            service_allocations = {}
            for service in self.service_names:
                service_data = day_data[day_data['service_name'] == service]
                if len(service_data) > 0:
                    # Use actual optimal_bandwidth_allocation from your data
                    avg_allocation = service_data['optimal_bandwidth_allocation'].mean()
                    service_allocations[f'bandwidth_{service}'] = avg_allocation
                else:
                    service_allocations[f'bandwidth_{service}'] = 0.0
            
            # Store service priorities
            for service in self.service_names:
                service_data = day_data[day_data['service_name'] == service]
                if len(service_data) > 0:
                    self.service_priorities[service] = service_data['service_priority'].iloc[0]
            
            # Combine all features
            day_record = {**temporal_features, **service_allocations, 'date': date}
            daily_data.append(day_record)
        
        daily_df = pd.DataFrame(daily_data)
        print(f"✅ Created daily dataset: {daily_df.shape}")
        
        return daily_df
    
    def prepare_training_data(self, daily_df):
        """Prepare features and targets for training"""
        print(f"\n🎯 PREPARING TRAINING DATA")
        print("=" * 50)
        
        # Define temporal features from your actual data
        self.temporal_features = [
            'hour_avg', 'day_of_week', 'is_weekend', 'is_business_hours',
            'is_peak_hours', 'is_wfh_core_hours', 'avg_network_utilization',
            'avg_signal_strength', 'avg_latency', 'total_devices',
            'avg_throughput', 'avg_quality_satisfaction'
        ]
        
        # Input features (X) - temporal data
        X = daily_df[self.temporal_features].values.astype(np.float32)
        
        # Target features (y) - service bandwidth allocations
        service_columns = [f'bandwidth_{service}' for service in self.service_names]
        y = daily_df[service_columns].values.astype(np.float32)
        
        print(f"📊 Input shape (X): {X.shape} - temporal features from your data")
        print(f"🎯 Output shape (y): {y.shape} - actual service bandwidth allocations")
        
        print(f"\n📈 Your actual bandwidth statistics:")
        print(f"   Daily total range: {y.sum(axis=1).min():.2f} - {y.sum(axis=1).max():.2f} Mbps")
        print(f"   Average daily total: {y.sum(axis=1).mean():.2f} Mbps")
        print(f"   Standard deviation: {y.sum(axis=1).std():.2f} Mbps")
        
        return X, y
    
    def create_neural_network(self, input_dim, output_dim):
        """Create neural network optimized for your data"""
        print(f"\n🏗️ BUILDING NEURAL NETWORK")
        print("=" * 50)
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),
            
            # Hidden layers - optimized for your service count
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer - one neuron per service
            layers.Dense(output_dim, activation='relu')  # ReLU to ensure positive bandwidth
        ])
        
        # Compile with appropriate loss for bandwidth prediction
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers in your data
            metrics=['mae', 'mse']
        )
        
        print(f"✅ Model architecture:")
        print(f"   Input: {input_dim} temporal features")
        print(f"   Output: {output_dim} services")
        print(f"   Parameters: {model.count_params():,}")
        
        return model
    
    def train_on_real_data(self, X, y):
        """Train model on your actual data"""
        print(f"\n🚀 TRAINING ON YOUR REAL DATA")
        print("=" * 50)
        
        # Split your data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"📊 Training split:")
        print(f"   Training: {X_train.shape[0]} days")
        print(f"   Validation: {X_val.shape[0]} days")
        print(f"   Test: {X_test.shape[0]} days")
        
        # Create model
        self.model = self.create_neural_network(X.shape[1], y.shape[1])
        
        # Training callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        print(f"\n🎯 Training on your historical patterns...")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate on test set
        print(f"\n📊 EVALUATING MODEL PERFORMANCE")
        print("=" * 50)
        
        test_predictions = self.model.predict(X_test_scaled, verbose=0)
        
        # Overall performance
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        overall_mae = mean_absolute_error(y_test, test_predictions)
        overall_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        print(f"📈 Overall Performance:")
        print(f"   MAE: {overall_mae:.3f} Mbps")
        print(f"   RMSE: {overall_rmse:.3f} Mbps")
        print(f"   Mean actual total: {y_test.sum(axis=1).mean():.2f} Mbps")
        print(f"   Mean predicted total: {test_predictions.sum(axis=1).mean():.2f} Mbps")
        
        # Top service performance
        print(f"\n📊 Top 10 Service Prediction Accuracy:")
        service_performance = []
        for i, service in enumerate(self.service_names[:10]):
            service_mae = mean_absolute_error(y_test[:, i], test_predictions[:, i])
            service_performance.append((service, service_mae))
            print(f"   {service:30}: MAE {service_mae:.3f} Mbps")
        
        return {
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'service_performance': service_performance,
            'history': history.history
        }
    
    def save_router_files(self):
        """Save minimal files for router deployment"""
        print(f"\n💾 SAVING ROUTER DEPLOYMENT FILES")
        print("=" * 50)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open('service_predictor.tflite', 'wb') as f:
            f.write(tflite_model)
        
        model_size = len(tflite_model) / 1024
        
        # Create router config with everything needed
        router_config = {
            "services": self.service_names,
            "service_priorities": {k: int(v) for k, v in self.service_priorities.items()},  # Convert int64 to int
            "temporal_features": self.temporal_features,
            "scaler_params": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist()
            },
            "model_info": {
                "input_features": len(self.temporal_features),
                "output_services": len(self.service_names),
                "model_size_kb": round(model_size, 1),
                "trained_on": "perfectly_balanced_wifi_dataset.csv",
                "created_at": datetime.now().isoformat()
            }
        }
        
        # Save router config
        with open('router_config.json', 'w') as f:
            json.dump(router_config, f, indent=2)
        
        print(f"✅ Router files created:")
        print(f"   📱 service_predictor.tflite ({model_size:.1f} KB)")
        print(f"   📄 router_config.json")
        
        return model_size

def main():
    """Main training pipeline"""
    print("🚀 REAL DATA SERVICE USAGE PREDICTOR")
    print("🎯 Training on YOUR actual WiFi dataset")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = RealDataServicePredictor()
        
        # Load and analyze your actual dataset
        df = predictor.load_and_analyze_dataset('perfectly_balanced_wifi_dataset.csv')
        
        # Create daily aggregated data from your samples
        daily_df = predictor.create_daily_aggregated_data(df)
        
        # Prepare training data
        X, y = predictor.prepare_training_data(daily_df)
        
        # Train on your real data
        performance = predictor.train_on_real_data(X, y)
        
        # Save router files
        model_size = predictor.save_router_files()
        
        print(f"\n🎉 SUCCESS! MODEL TRAINED ON YOUR REAL DATA")
        print("=" * 60)
        print(f"📊 Model learned patterns from {len(predictor.service_names)} actual services")
        print(f"📈 Prediction accuracy: {performance['overall_mae']:.2f} Mbps MAE")
        print(f"📱 Model size: {model_size:.1f} KB (router-ready)")
        
        print(f"\n📁 Files ready for router deployment:")
        print(f"   ✅ service_predictor.tflite")
        print(f"   ✅ router_config.json")
        
        print(f"\n🎯 Next: Use router_inference.py to predict service usage!")
        
        return True
        
    except FileNotFoundError:
        print("❌ Error: perfectly_balanced_wifi_dataset.csv not found")
        print("💡 Make sure your dataset file is in the current directory")
        return False
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()