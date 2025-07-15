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
