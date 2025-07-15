#!/usr/bin/env python3
"""
AI-Enhanced Weather-Taxi Productivity Analysis
Comparing traditional taxi operations vs AI-assisted weather intelligence systems
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy.stats import pearsonr, ttest_ind
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AIWeatherProductivityAnalyzer:
    """Analyzes productivity gains from AI-enhanced weather intelligence for taxi drivers"""
    
    def __init__(self):
        self.traditional_data = None
        self.ai_enhanced_data = None
        self.comparison_results = {}
        self.ai_model = None
        self.output_dir = Path("ai_productivity_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_traditional_taxi_data(self, n_samples=5000, start_date='2024-06-01'):
        """Generate traditional taxi operation data (reactive to weather)"""
        print("Generating traditional taxi operation data...")
        
        np.random.seed(42)
        date_start = datetime.strptime(start_date, '%Y-%m-%d')
        
        traditional_records = []
        
        for i in range(n_samples):
            # Random timestamp
            timestamp = date_start + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Generate weather conditions
            is_raining = np.random.random() < 0.25
            rain_intensity = np.random.exponential(3) if is_raining else 0
            temperature = np.random.normal(22, 8)
            wind_speed = np.random.exponential(8)
            visibility = np.random.normal(15, 5) if not is_raining else np.random.normal(6, 3)
            visibility = max(0.5, min(20, visibility))
            
            # Traditional driver behavior (reactive, no prediction)
            base_productivity = 1.0
            
            # Time-based patterns
            if hour in [7, 8, 9, 17, 18, 19]:  # Rush hour
                base_productivity *= 1.4
            if day_of_week >= 5:  # Weekend
                base_productivity *= 1.2
                
            # REACTIVE weather response (drivers only react after weather starts)
            weather_response_delay = np.random.uniform(15, 45)  # 15-45 min delay to react
            
            if is_raining:
                # Delayed response to rain
                if rain_intensity > 5:  # Heavy rain
                    weather_multiplier = 2.2  # Lower than optimal due to poor positioning
                elif rain_intensity > 1:  # Light rain  
                    weather_multiplier = 1.6
                else:  # Drizzle
                    weather_multiplier = 1.3
            else:
                weather_multiplier = 1.0
                
            # Temperature effects (no prediction)
            if temperature < 5 or temperature > 35:
                weather_multiplier *= 1.2  # Limited response
                
            # Poor positioning due to lack of prediction
            positioning_efficiency = np.random.uniform(0.6, 0.8)  # 60-80% efficiency
            
            total_productivity = base_productivity * weather_multiplier * positioning_efficiency
            
            # Calculate metrics
            base_revenue_per_min = 52.3
            revenue_per_min = base_revenue_per_min * total_productivity
            
            # Trip duration affected by poor weather preparation
            base_trip_duration = 32.4
            weather_delay_factor = 1.0
            if is_raining:
                weather_delay_factor = 1.3 + (rain_intensity * 0.05)  # More delay due to poor preparation
                
            trip_duration = base_trip_duration * weather_delay_factor
            
            # Wait time (longer due to reactive positioning)
            base_wait_time = 6.8
            if is_raining:
                wait_time = base_wait_time * (1.5 + rain_intensity * 0.1)
            else:
                wait_time = base_wait_time * np.random.uniform(0.8, 1.2)
                
            # Utilization rate
            utilization = min(95, max(40, 65 * positioning_efficiency * (1 + weather_multiplier * 0.1)))
            
            # Daily earnings calculation
            daily_hours = 10
            daily_earnings = revenue_per_min * 60 * daily_hours * (utilization / 100)
            
            traditional_records.append({
                'timestamp': timestamp,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_raining': is_raining,
                'rain_intensity': round(rain_intensity, 2),
                'temperature': round(temperature, 1),
                'wind_speed': round(wind_speed, 1),
                'visibility': round(visibility, 1),
                'revenue_per_min': round(revenue_per_min, 2),
                'trip_duration': round(trip_duration, 1),
                'wait_time': round(wait_time, 1),
                'utilization_rate': round(utilization, 1),
                'daily_earnings': round(daily_earnings),
                'positioning_efficiency': round(positioning_efficiency, 2),
                'weather_response_delay': round(weather_response_delay, 1),
                'operation_type': 'traditional'
            })
            
        self.traditional_data = pd.DataFrame(traditional_records)
        self.traditional_data.to_csv(self.output_dir / 'traditional_operations.csv', index=False)
        print(f"Generated {len(self.traditional_data)} traditional operation records")
        
    def generate_ai_enhanced_data(self, n_samples=5000, start_date='2024-06-01'):
        """Generate AI-enhanced taxi operation data (predictive and optimized)"""
        print("Generating AI-enhanced taxi operation data...")
        
        np.random.seed(43)  # Different seed for variation
        date_start = datetime.strptime(start_date, '%Y-%m-%d')
        
        ai_enhanced_records = []
        
        for i in range(n_samples):
            # Random timestamp
            timestamp = date_start + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Generate same weather conditions as traditional for fair comparison
            is_raining = np.random.random() < 0.25
            rain_intensity = np.random.exponential(3) if is_raining else 0
            temperature = np.random.normal(22, 8)
            wind_speed = np.random.exponential(8)
            visibility = np.random.normal(15, 5) if not is_raining else np.random.normal(6, 3)
            visibility = max(0.5, min(20, visibility))
            
            # AI-enhanced driver behavior (predictive and optimized)
            base_productivity = 1.0
            
            # Time-based patterns (AI optimizes these)
            if hour in [7, 8, 9, 17, 18, 19]:  # Rush hour
                base_productivity *= 1.6  # Better optimization than traditional
            if day_of_week >= 5:  # Weekend
                base_productivity *= 1.35  # AI identifies better weekend opportunities
                
            # PREDICTIVE weather response (AI forecasts 3 hours ahead)
            ai_forecast_accuracy = 0.87  # 87% forecast accuracy
            weather_prediction_lead_time = np.random.uniform(60, 180)  # 1-3 hours advance notice
            
            if is_raining:
                # Proactive positioning based on AI predictions
                if rain_intensity > 5:  # Heavy rain predicted
                    weather_multiplier = 3.1  # Much better positioning
                elif rain_intensity > 1:  # Light rain predicted
                    weather_multiplier = 2.4
                else:  # Drizzle predicted
                    weather_multiplier = 1.8
            else:
                weather_multiplier = 1.0
                
            # AI-optimized temperature response
            if temperature < 5 or temperature > 35:
                weather_multiplier *= 1.4  # Better preparation for extreme temperatures
                
            # AI-optimized positioning (machine learning based)
            ai_positioning_efficiency = np.random.uniform(0.85, 0.95)  # 85-95% efficiency
            
            # AI surge pricing optimization
            ai_pricing_multiplier = 1.05 + (weather_multiplier - 1) * 0.3
            
            total_productivity = base_productivity * weather_multiplier * ai_positioning_efficiency * ai_pricing_multiplier
            
            # Calculate enhanced metrics
            base_revenue_per_min = 52.3
            revenue_per_min = base_revenue_per_min * total_productivity
            
            # Trip duration optimized by AI routing
            base_trip_duration = 32.4
            ai_route_optimization = 0.88  # 12% faster routes on average
            weather_delay_factor = 1.0
            if is_raining:
                weather_delay_factor = 1.1 + (rain_intensity * 0.02)  # Much less delay due to prediction
                
            trip_duration = base_trip_duration * weather_delay_factor * ai_route_optimization
            
            # Wait time (much shorter due to predictive positioning)
            base_wait_time = 6.8
            ai_wait_reduction = 0.62  # 38% reduction in wait times
            if is_raining:
                wait_time = base_wait_time * ai_wait_reduction * (1.2 + rain_intensity * 0.05)
            else:
                wait_time = base_wait_time * ai_wait_reduction * np.random.uniform(0.9, 1.1)
                
            # Higher utilization rate due to AI optimization
            utilization = min(95, max(65, 83 * ai_positioning_efficiency * (1 + weather_multiplier * 0.08)))
            
            # Daily earnings calculation (higher due to all optimizations)
            daily_hours = 10
            daily_earnings = revenue_per_min * 60 * daily_hours * (utilization / 100)
            
            # AI-specific metrics
            ai_confidence_score = np.random.uniform(0.75, 0.95)
            prediction_accuracy = ai_forecast_accuracy if is_raining else 0.92
            
            ai_enhanced_records.append({
                'timestamp': timestamp,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_raining': is_raining,
                'rain_intensity': round(rain_intensity, 2),
                'temperature': round(temperature, 1),
                'wind_speed': round(wind_speed, 1),
                'visibility': round(visibility, 1),
                'revenue_per_min': round(revenue_per_min, 2),
                'trip_duration': round(trip_duration, 1),
                'wait_time': round(wait_time, 1),
                'utilization_rate': round(utilization, 1),
                'daily_earnings': round(daily_earnings),
                'positioning_efficiency': round(ai_positioning_efficiency, 2),
                'ai_prediction_lead_time': round(weather_prediction_lead_time, 1),
                'ai_confidence_score': round(ai_confidence_score, 2),
                'prediction_accuracy': round(prediction_accuracy, 2),
                'operation_type': 'ai_enhanced'
            })
            
        self.ai_enhanced_data = pd.DataFrame(ai_enhanced_records)
        self.ai_enhanced_data.to_csv(self.output_dir / 'ai_enhanced_operations.csv', index=False)
        print(f"Generated {len(self.ai_enhanced_data)} AI-enhanced operation records")
