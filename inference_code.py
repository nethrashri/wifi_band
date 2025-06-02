#!/usr/bin/env python3
"""
Router Inference - Using Real Data Model
Predicts service usage and priorities based on your actual historical data
Usage: python router_inference_real.py [epoch_timestamp]
"""

import json
import sys
import numpy as np
from datetime import datetime, timedelta

# Import TensorFlow Lite
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TF_AVAILABLE = False
    except ImportError:
        print("❌ Error: TensorFlow Lite not available")
        print("Install: pip install tensorflow or pip install tflite-runtime")
        sys.exit(1)

class RealDataRouterPredictor:
    def __init__(self, model_path="service_predictor.tflite", config_path="router_config.json"):
        """Initialize predictor using your real data model"""
        
        # Load router config
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_path}")
            print("Run real_data_training.py first to train on your dataset")
            sys.exit(1)
        
        # Load TFLite model
        try:
            if TF_AVAILABLE:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
            else:
                self.interpreter = tflite.Interpreter(model_path=model_path)
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Run real_data_training.py first to create the model")
            sys.exit(1)
        
        # Extract configuration
        self.services = self.config['services']
        self.service_priorities = self.config['service_priorities']
        self.temporal_features = self.config['temporal_features']
        
        # Scaler parameters
        scaler = self.config['scaler_params']
        self.scaler_mean = np.array(scaler['mean'], dtype=np.float32)
        self.scaler_scale = np.array(scaler['scale'], dtype=np.float32)
        
        print(f"✅ Real data predictor loaded!")
        print(f"📊 Trained on {len(self.services)} actual services from your dataset")
        print(f"📱 Model size: {self.config['model_info']['model_size_kb']} KB")
    
    def extract_temporal_features(self, epoch_timestamp):
        """Extract temporal features matching your training data"""
        dt = datetime.fromtimestamp(epoch_timestamp)
        
        # Calculate features exactly as in training
        features = {
            'hour_avg': dt.hour,  # Average hour of day
            'day_of_week': dt.weekday(),  # 0=Monday, 6=Sunday
            'is_weekend': 1.0 if dt.weekday() >= 5 else 0.0,
            'is_business_hours': 1.0 if 9 <= dt.hour <= 17 else 0.0,
            'is_peak_hours': 1.0 if dt.hour in [8, 9, 17, 18, 19, 20] else 0.0,
            'is_wfh_core_hours': 1.0 if 9 <= dt.hour <= 15 else 0.0,
            'avg_network_utilization': 0.5,  # Default assumption
            'avg_signal_strength': -45.0,    # Default assumption
            'avg_latency': 20.0,             # Default assumption
            'total_devices': 15.0,           # Default assumption
            'avg_throughput': 50.0,          # Default assumption
            'avg_quality_satisfaction': 8.5   # Default assumption
        }
        
        return np.array([features[feature] for feature in self.temporal_features], dtype=np.float32)
    
    def predict_service_usage(self, epoch_timestamp):
        """Predict service usage and priorities for given timestamp"""
        
        # Extract temporal features
        features = self.extract_temporal_features(epoch_timestamp)
        
        # Scale features using learned parameters
        features_scaled = (features - self.scaler_mean) / self.scaler_scale
        
        # Run TFLite inference
        self.interpreter.set_tensor(self.input_details[0]['index'], features_scaled.reshape(1, -1))
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0.1)
        
        # Create service predictions with priorities
        service_predictions = []
        for i, service in enumerate(self.services):
            predicted_bandwidth = float(predictions[i])
            priority = self.service_priorities.get(service, 0)
            
            service_predictions.append({
                'service_name': service,
                'predicted_bandwidth_mbps': round(predicted_bandwidth, 2),
                'priority': priority,
                'usage_percentage': round((predicted_bandwidth / predictions.sum()) * 100, 1)
            })
        
        # Sort by predicted bandwidth (highest first)
        service_predictions.sort(key=lambda x: x['predicted_bandwidth_mbps'], reverse=True)
        
        # Create result
        dt = datetime.fromtimestamp(epoch_timestamp)
        result = {
            'timestamp': epoch_timestamp,
            'date': dt.strftime('%Y-%m-%d'),
            'day_of_week': dt.strftime('%A'),
            'time': dt.strftime('%H:%M:%S'),
            'total_predicted_bandwidth': round(float(predictions.sum()), 2),
            'service_predictions': service_predictions,
            'top_5_services': service_predictions[:5],
            'model_info': {
                'trained_on_real_data': True,
                'total_services': len(self.services),
                'prediction_based_on': 'perfectly_balanced_wifi_dataset.csv'
            }
        }
        
        return result

def display_prediction_results(result):
    """Display prediction results in router-friendly format"""
    print(f"\n📊 SERVICE USAGE PREDICTION")
    print("=" * 50)
    print(f"📅 Date: {result['date']} ({result['day_of_week']})")
    print(f"⏰ Time: {result['time']}")
    print(f"🌐 Total Bandwidth: {result['total_predicted_bandwidth']} Mbps")
    
    print(f"\n🎯 TOP 10 SERVICES BY PREDICTED USAGE:")
    print("-" * 60)
    print(f"{'Rank':<4} {'Service':<25} {'Bandwidth':<12} {'Priority':<8} {'%':<6}")
    print("-" * 60)
    
    for i, service in enumerate(result['service_predictions'][:10]):
        rank = i + 1
        name = service['service_name'][:24]  # Truncate long names
        bandwidth = f"{service['predicted_bandwidth_mbps']} Mbps"
        priority = service['priority']
        percentage = f"{service['usage_percentage']}%"
        
        print(f"{rank:<4} {name:<25} {bandwidth:<12} {priority:<8} {percentage:<6}")
    
    if len(result['service_predictions']) > 10:
        remaining = len(result['service_predictions']) - 10
        remaining_bandwidth = sum(s['predicted_bandwidth_mbps'] for s in result['service_predictions'][10:])
        print(f"{'...':<4} {f'+ {remaining} more services':<25} {f'{remaining_bandwidth:.2f} Mbps':<12}")
    
    print("-" * 60)
    
    # High priority services summary
    high_priority_services = [s for s in result['service_predictions'] if s['priority'] >= 3]
    if high_priority_services:
        print(f"\n⚡ HIGH PRIORITY SERVICES (Priority >= 3):")
        for service in high_priority_services[:5]:
            print(f"   🔸 {service['service_name']}: {service['predicted_bandwidth_mbps']} Mbps (Priority {service['priority']})")
    
    # Business hours insight
    current_time = datetime.fromtimestamp(result['timestamp'])
    if 9 <= current_time.hour <= 17 and current_time.weekday() < 5:
        print(f"\n💼 Business Hours Active - Higher priority for business services")
    elif current_time.weekday() >= 5:
        print(f"\n🏠 Weekend - Higher usage expected for entertainment services")
    
    return result

def main():
    """Main router inference function"""
    if len(sys.argv) < 2:
        print("🔮 Router Service Usage Predictor (Real Data)")
        print("=" * 45)
        print("Trained on your actual WiFi dataset")
        print("\nUsage: python router_inference_real.py [epoch_timestamp]")
        print("\nExample epochs:")
        
        current = int(datetime.now().timestamp())
        christmas = int(datetime(2024, 12, 25, 14, 0).timestamp())
        monday_morning = int(datetime(2024, 8, 19, 9, 0).timestamp())
        weekend_evening = int(datetime(2024, 8, 17, 20, 0).timestamp())
        
        print(f"  Current time: {current}")
        print(f"  Christmas 2PM: {christmas}")
        print(f"  Monday 9AM: {monday_morning}")
        print(f"  Saturday 8PM: {weekend_evening}")
        
        print(f"\nTest: python router_inference_real.py {current}")
        return
    
    # Get epoch from command line
    try:
        epoch = int(sys.argv[1])
        
        # Validate epoch (reasonable range)
        min_epoch = int(datetime(2020, 1, 1).timestamp())
        max_epoch = int(datetime(2030, 12, 31).timestamp())
        
        if not (min_epoch <= epoch <= max_epoch):
            raise ValueError("Epoch timestamp out of reasonable range")
            
    except ValueError as e:
        print(f"❌ Invalid epoch: {sys.argv[1]}")
        print("Must be a valid Unix timestamp")
        return
    
    # Initialize predictor
    try:
        predictor = RealDataRouterPredictor()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Make prediction
    try:
        print(f"🔮 Predicting service usage for epoch: {epoch}")
        result = predictor.predict_service_usage(epoch)
        
        # Display results
        display_prediction_results(result)
        
        # Save result for router logs
        output_file = f"prediction_{epoch}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Router-friendly summary
        top_service = result['service_predictions'][0]
        print(f"\n🎯 ROUTER SUMMARY:")
        print(f"   Total bandwidth needed: {result['total_predicted_bandwidth']} Mbps")
        print(f"   Top service: {top_service['service_name']} ({top_service['predicted_bandwidth_mbps']} Mbps)")
        print(f"   High priority services: {len([s for s in result['service_predictions'] if s['priority'] >= 3])}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

def predict_api(epoch_timestamp):
    """API function for router integration"""
    try:
        predictor = RealDataRouterPredictor()
        return predictor.predict_service_usage(epoch_timestamp)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    main()