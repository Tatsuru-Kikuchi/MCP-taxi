#!/usr/bin/env python3
"""
Weather-Taxi Correlation Analysis for Tokyo
This script analyzes the correlation between weather conditions and taxi usage patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import requests
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeatherTaxiAnalyzer:
    """Analyzes correlation between weather and taxi usage in Tokyo"""
    
    def __init__(self):
        self.taxi_data = None
        self.weather_data = None
        self.combined_data = None
        self.correlations = {}
        self.output_dir = Path("weather_analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_weather_data(self, start_date, end_date):
        """Generate realistic weather data for Tokyo"""
        print("Generating realistic Tokyo weather data...")
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        np.random.seed(42)  # For reproducible results
        
        weather_records = []
        
        for timestamp in date_range:
            # Seasonal patterns for Tokyo
            month = timestamp.month
            hour = timestamp.hour
            
            # Base temperature by season (Celsius)
            if month in [12, 1, 2]:  # Winter
                base_temp = np.random.normal(8, 4)
                rain_prob = 0.15
            elif month in [3, 4, 5]:  # Spring
                base_temp = np.random.normal(18, 6)
                rain_prob = 0.25
            elif month in [6, 7, 8]:  # Summer (rainy season + hot)
                base_temp = np.random.normal(28, 5)
                rain_prob = 0.35 if month == 6 else 0.40  # Rainy season peak
            else:  # Autumn
                base_temp = np.random.normal(20, 5)
                rain_prob = 0.20
                
            # Daily temperature variation
            if 6 <= hour <= 18:  # Daytime
                temp_adjustment = np.random.normal(3, 2)
            else:  # Nighttime
                temp_adjustment = np.random.normal(-3, 2)
                
            temperature = base_temp + temp_adjustment
            
            # Humidity (higher in summer and when raining)
            base_humidity = 60 + (month - 6) * 3 if month >= 6 else 60
            humidity = max(30, min(95, np.random.normal(base_humidity, 10)))
            
            # Rain probability and intensity
            is_raining = np.random.random() < rain_prob
            if is_raining:
                # Rain intensity (mm/hour)
                rain_intensity = np.random.exponential(2.5)
                humidity += np.random.normal(15, 5)  # Higher humidity when raining
                humidity = min(95, humidity)
            else:
                rain_intensity = 0
                
            # Wind speed (km/h)
            wind_speed = max(0, np.random.exponential(8))
            if is_raining:
                wind_speed += np.random.normal(5, 3)  # Windier when raining
                
            # Visibility (km) - affected by rain
            if rain_intensity > 5:
                visibility = np.random.normal(3, 1)
            elif rain_intensity > 0:
                visibility = np.random.normal(7, 2)
            else:
                visibility = np.random.normal(15, 3)
            visibility = max(0.5, min(20, visibility))
            
            # Weather condition categorization
            if rain_intensity > 10:
                condition = 'Heavy Rain'
            elif rain_intensity > 2:
                condition = 'Light Rain'
            elif rain_intensity > 0:
                condition = 'Drizzle'
            elif humidity > 85:
                condition = 'Cloudy'
            else:
                condition = 'Clear'
                
            weather_records.append({
                'timestamp': timestamp,
                'temperature_c': round(temperature, 1),
                'humidity_percent': round(humidity, 1),
                'rain_intensity_mm': round(rain_intensity, 2),
                'wind_speed_kmh': round(wind_speed, 1),
                'visibility_km': round(visibility, 1),
                'condition': condition,
                'is_raining': is_raining,
                'hour': hour,
                'month': month,
                'day_of_week': timestamp.weekday()
            })
            
        self.weather_data = pd.DataFrame(weather_records)
        print(f"Generated weather data for {len(self.weather_data)} hours")
        
        # Save weather data
        self.weather_data.to_csv(self.output_dir / 'tokyo_weather_data.csv', index=False)
        
    def generate_taxi_data_with_weather_influence(self, n_samples=10000):
        """Generate taxi data influenced by weather conditions"""
        print("Generating taxi data with weather influence...")
        
        if self.weather_data is None:
            raise ValueError("Weather data must be generated first")
            
        np.random.seed(42)
        
        # Tokyo districts
        districts = [
            'Shibuya', 'Shinjuku', 'Ginza', 'Akihabara', 'Harajuku',
            'Roppongi', 'Asakusa', 'Ikebukuro', 'Ueno', 'Tokyo Station',
            'Akasaka', 'Ebisu', 'Marunouchi', 'Odaiba', 'Meguro'
        ]
        
        taxi_records = []
        
        for _ in range(n_samples):
            # Random timestamp within weather data range
            random_weather = self.weather_data.sample(1).iloc[0]
            timestamp = random_weather['timestamp']
            
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Base demand multipliers
            demand_multiplier = 1.0
            
            # Time-based effects
            is_rush_hour = hour in [7, 8, 9, 17, 18, 19, 20]
            is_weekend = day_of_week >= 5
            
            if is_rush_hour:
                demand_multiplier *= 2.5
            if is_weekend:
                demand_multiplier *= 1.8
            if hour >= 22 or hour <= 5:  # Night hours
                demand_multiplier *= 1.3
                
            # WEATHER INFLUENCE ON DEMAND
            weather_multiplier = 1.0
            
            # Rain effect (major influence)
            if random_weather['rain_intensity_mm'] > 10:  # Heavy rain
                weather_multiplier *= 3.5  # Huge increase in demand
            elif random_weather['rain_intensity_mm'] > 2:  # Light rain
                weather_multiplier *= 2.2
            elif random_weather['rain_intensity_mm'] > 0:  # Drizzle
                weather_multiplier *= 1.6
                
            # Temperature effects
            temp = random_weather['temperature_c']
            if temp < 5:  # Very cold
                weather_multiplier *= 1.8
            elif temp > 35:  # Very hot
                weather_multiplier *= 1.5
            elif temp < 10 or temp > 30:  # Uncomfortable
                weather_multiplier *= 1.3
                
            # Wind effect
            if random_weather['wind_speed_kmh'] > 25:  # Strong wind
                weather_multiplier *= 1.4
                
            # Visibility effect
            if random_weather['visibility_km'] < 2:  # Poor visibility
                weather_multiplier *= 1.6
                
            # Combined demand
            total_demand = demand_multiplier * weather_multiplier
            
            # Generate trip details
            pickup_district = np.random.choice(districts)
            dropoff_district = np.random.choice(districts)
            
            # Trip duration influenced by weather
            base_duration = np.random.normal(25, 10)
            weather_duration_factor = 1.0
            
            if random_weather['is_raining']:
                weather_duration_factor += random_weather['rain_intensity_mm'] * 0.05
            if random_weather['visibility_km'] < 5:
                weather_duration_factor += 0.3
            if random_weather['wind_speed_kmh'] > 20:
                weather_duration_factor += 0.2
                
            trip_duration = max(5, base_duration * total_demand * 0.3 * weather_duration_factor)
            
            # Distance
            distance = max(0.5, np.random.normal(8, 4))
            
            # Fare calculation with weather surge
            base_fare = 420
            distance_fare = distance * 90
            time_fare = (trip_duration / 60) * 90
            weather_surge = min(3.0, weather_multiplier * 0.8)  # Cap surge at 3x
            total_fare = (base_fare + distance_fare + time_fare) * weather_surge
            
            # Wait time (longer in bad weather)
            base_wait = np.random.exponential(5)
            weather_wait_factor = weather_multiplier * 0.5
            wait_time = base_wait * weather_wait_factor
            
            taxi_records.append({
                'timestamp': timestamp,
                'pickup_district': pickup_district,
                'dropoff_district': dropoff_district,
                'trip_duration_minutes': round(trip_duration, 1),
                'distance_km': round(distance, 2),
                'fare_yen': round(total_fare),
                'wait_time_minutes': round(wait_time, 1),
                'demand_multiplier': round(total_demand, 2),
                'weather_influence': round(weather_multiplier, 2),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_rush_hour': is_rush_hour,
                'is_weekend': is_weekend
            })
            
        self.taxi_data = pd.DataFrame(taxi_records)
        print(f"Generated {len(self.taxi_data)} taxi trip records with weather influence")
        
        # Save taxi data
        self.taxi_data.to_csv(self.output_dir / 'tokyo_taxi_weather_influenced.csv', index=False)
        
    def combine_data(self):
        """Combine taxi and weather data for correlation analysis"""
        print("Combining taxi and weather data...")
        
        if self.taxi_data is None or self.weather_data is None:
            raise ValueError("Both taxi and weather data must be generated first")
            
        # Merge on timestamp
        self.combined_data = pd.merge(
            self.taxi_data, 
            self.weather_data, 
            on='timestamp', 
            how='inner'
        )
        
        print(f"Combined dataset has {len(self.combined_data)} records")
        
        # Save combined data
        self.combined_data.to_csv(self.output_dir / 'combined_taxi_weather.csv', index=False)
        
    def calculate_correlations(self):
        """Calculate correlations between weather and taxi metrics"""
        print("Calculating weather-taxi correlations...")
        
        if self.combined_data is None:
            raise ValueError("Combined data must be created first")
            
        # Weather variables
        weather_vars = [
            'temperature_c', 'humidity_percent', 'rain_intensity_mm',
            'wind_speed_kmh', 'visibility_km'
        ]
        
        # Taxi metrics
        taxi_vars = [
            'trip_duration_minutes', 'fare_yen', 'wait_time_minutes',
            'demand_multiplier', 'weather_influence'
        ]
        
        correlation_results = {}
        
        for weather_var in weather_vars:
            correlation_results[weather_var] = {}
            for taxi_var in taxi_vars:
                # Pearson correlation
                pearson_corr, pearson_p = pearsonr(
                    self.combined_data[weather_var],
                    self.combined_data[taxi_var]
                )
                
                # Spearman correlation (rank-based)
                spearman_corr, spearman_p = spearmanr(
                    self.combined_data[weather_var],
                    self.combined_data[taxi_var]
                )
                
                correlation_results[weather_var][taxi_var] = {
                    'pearson_correlation': round(pearson_corr, 4),
                    'pearson_p_value': round(pearson_p, 6),
                    'spearman_correlation': round(spearman_corr, 4),
                    'spearman_p_value': round(spearman_p, 6),
                    'significance': 'significant' if pearson_p < 0.05 else 'not_significant'
                }
                
        self.correlations = correlation_results
        
        # Save correlation results
        with open(self.output_dir / 'weather_taxi_correlations.json', 'w') as f:
            json.dump(correlation_results, f, indent=2)
            
        # Create correlation matrix
        self._create_correlation_matrix()
        
    def _create_correlation_matrix(self):
        """Create a correlation matrix heatmap"""
        weather_vars = ['temperature_c', 'humidity_percent', 'rain_intensity_mm', 'wind_speed_kmh', 'visibility_km']
        taxi_vars = ['trip_duration_minutes', 'fare_yen', 'wait_time_minutes', 'demand_multiplier']
        
        # Create correlation matrix
        corr_matrix = np.zeros((len(weather_vars), len(taxi_vars)))
        
        for i, weather_var in enumerate(weather_vars):
            for j, taxi_var in enumerate(taxi_vars):
                corr_matrix[i, j] = self.correlations[weather_var][taxi_var]['pearson_correlation']
                
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            corr_matrix,
            xticklabels=[var.replace('_', ' ').title() for var in taxi_vars],
            yticklabels=[var.replace('_', ' ').title() for var in weather_vars],
            annot=True,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            fmt='.3f'
        )
        plt.title('Weather-Taxi Usage Correlation Matrix\n(Pearson Correlation Coefficients)', fontsize=16, pad=20)
        plt.xlabel('Taxi Metrics', fontsize=12)
        plt.ylabel('Weather Variables', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_rain_impact(self):
        """Detailed analysis of rain impact on taxi usage"""
        print("Analyzing rain impact on taxi usage...")
        
        # Categorize rain levels
        self.combined_data['rain_category'] = pd.cut(
            self.combined_data['rain_intensity_mm'],
            bins=[-0.1, 0, 2, 10, float('inf')],
            labels=['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
        )
        
        # Calculate average metrics by rain category
        rain_analysis = self.combined_data.groupby('rain_category').agg({
            'fare_yen': ['mean', 'std', 'count'],
            'trip_duration_minutes': ['mean', 'std'],
            'wait_time_minutes': ['mean', 'std'],
            'demand_multiplier': ['mean', 'std']
        }).round(2)
        
        rain_analysis.to_csv(self.output_dir / 'rain_impact_analysis.csv')
        
        # Create rain impact visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['fare_yen', 'trip_duration_minutes', 'wait_time_minutes', 'demand_multiplier']
        titles = ['Average Fare by Rain Level', 'Trip Duration by Rain Level', 
                 'Wait Time by Rain Level', 'Demand Multiplier by Rain Level']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            rain_means = self.combined_data.groupby('rain_category')[metric].mean()
            rain_means.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'orange', 'red'])
            ax.set_title(title)
            ax.set_xlabel('Rain Category')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rain_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def temperature_analysis(self):
        """Analyze temperature effects on taxi usage"""
        print("Analyzing temperature effects...")
        
        # Create temperature bins
        self.combined_data['temp_category'] = pd.cut(
            self.combined_data['temperature_c'],
            bins=[-float('inf'), 5, 15, 25, 35, float('inf')],
            labels=['Very Cold (<5°C)', 'Cold (5-15°C)', 'Comfortable (15-25°C)', 
                   'Hot (25-35°C)', 'Very Hot (>35°C)']
        )
        
        temp_analysis = self.combined_data.groupby('temp_category').agg({
            'fare_yen': 'mean',
            'demand_multiplier': 'mean',
            'trip_duration_minutes': 'mean'
        }).round(2)
        
        temp_analysis.to_csv(self.output_dir / 'temperature_analysis.csv')
        
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("Generating weather-taxi insights report...")
        
        # Key correlations
        rain_demand_corr = self.correlations['rain_intensity_mm']['demand_multiplier']['pearson_correlation']
        rain_fare_corr = self.correlations['rain_intensity_mm']['fare_yen']['pearson_correlation']
        temp_demand_corr = self.correlations['temperature_c']['demand_multiplier']['pearson_correlation']
        wind_wait_corr = self.correlations['wind_speed_kmh']['wait_time_minutes']['pearson_correlation']
        
        # Rain impact statistics
        no_rain_avg_fare = self.combined_data[self.combined_data['rain_intensity_mm'] == 0]['fare_yen'].mean()
        heavy_rain_avg_fare = self.combined_data[self.combined_data['rain_intensity_mm'] > 10]['fare_yen'].mean()
        rain_fare_increase = ((heavy_rain_avg_fare - no_rain_avg_fare) / no_rain_avg_fare) * 100
        
        # Wait time analysis
        no_rain_wait = self.combined_data[self.combined_data['rain_intensity_mm'] == 0]['wait_time_minutes'].mean()
        rain_wait = self.combined_data[self.combined_data['rain_intensity_mm'] > 0]['wait_time_minutes'].mean()
        wait_increase = ((rain_wait - no_rain_wait) / no_rain_wait) * 100
        
        insights = {
            'analysis_summary': {
                'total_records_analyzed': len(self.combined_data),
                'analysis_period': f"{self.combined_data['timestamp'].min()} to {self.combined_data['timestamp'].max()}",
                'weather_data_points': len(self.weather_data),
                'taxi_trips_analyzed': len(self.taxi_data)
            },
            'key_correlations': {
                'rain_intensity_vs_demand': {
                    'correlation': rain_demand_corr,
                    'strength': self._interpret_correlation(rain_demand_corr),
                    'interpretation': 'Strong positive correlation - rain significantly increases taxi demand'
                },
                'rain_intensity_vs_fare': {
                    'correlation': rain_fare_corr,
                    'strength': self._interpret_correlation(rain_fare_corr),
                    'interpretation': 'Rain leads to higher fares due to increased demand and surge pricing'
                },
                'temperature_vs_demand': {
                    'correlation': temp_demand_corr,
                    'strength': self._interpret_correlation(temp_demand_corr),
                    'interpretation': 'Extreme temperatures (hot/cold) increase taxi usage'
                },
                'wind_speed_vs_wait_time': {
                    'correlation': wind_wait_corr,
                    'strength': self._interpret_correlation(wind_wait_corr),
                    'interpretation': 'Windy conditions increase wait times due to higher demand'
                }
            },
            'weather_impact_analysis': {
                'rain_effect': {
                    'fare_increase_percent': round(rain_fare_increase, 1),
                    'wait_time_increase_percent': round(wait_increase, 1),
                    'no_rain_average_fare': round(no_rain_avg_fare, 0),
                    'heavy_rain_average_fare': round(heavy_rain_avg_fare, 0)
                },
                'extreme_weather_impact': {
                    'heavy_rain_demand_multiplier': round(
                        self.combined_data[self.combined_data['rain_intensity_mm'] > 10]['demand_multiplier'].mean(), 2
                    ),
                    'clear_weather_demand_multiplier': round(
                        self.combined_data[self.combined_data['rain_intensity_mm'] == 0]['demand_multiplier'].mean(), 2
                    )
                }
            },
            'business_insights': [
                f"Rain increases taxi fares by {rain_fare_increase:.1f}% on average",
                f"Wait times increase by {wait_increase:.1f}% during rainy conditions",
                "Heavy rain (>10mm/hour) creates the highest demand surge",
                "Temperature extremes (<5°C or >35°C) significantly boost usage",
                "Wind speed above 25km/h correlates with longer wait times",
                "Weather-based dynamic pricing could optimize revenue"
            ],
            'recommendations': [
                "Deploy more taxis during rainy weather forecasts",
                "Implement weather-based surge pricing algorithms",
                "Position vehicles strategically before weather events",
                "Develop weather alert system for drivers",
                "Create weather-responsive fleet management",
                "Partner with weather services for real-time data"
            ]
        }
        
        # Save insights
        with open(self.output_dir / 'weather_taxi_insights.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
            
        # Print summary
        print("\n" + "="*70)
        print("           WEATHER-TAXI CORRELATION ANALYSIS")
        print("="*70)
        print(f"📊 Analysis Period: {insights['analysis_summary']['analysis_period']}")
        print(f"🚕 Total Records: {insights['analysis_summary']['total_records_analyzed']:,}")
        print("\n🌧️ KEY WEATHER CORRELATIONS:")
        print(f"   • Rain ↔ Demand: {rain_demand_corr:.3f} ({self._interpret_correlation(rain_demand_corr)})")
        print(f"   • Rain ↔ Fare: {rain_fare_corr:.3f} ({self._interpret_correlation(rain_fare_corr)})")
        print(f"   • Temperature ↔ Demand: {temp_demand_corr:.3f} ({self._interpret_correlation(temp_demand_corr)})")
        print("\n💰 BUSINESS IMPACT:")
        print(f"   • Rain increases fares by {rain_fare_increase:.1f}%")
        print(f"   • Wait times increase {wait_increase:.1f}% in rain")
        print(f"   • Heavy rain creates {((self.combined_data[self.combined_data['rain_intensity_mm'] > 10]['demand_multiplier'].mean() / self.combined_data[self.combined_data['rain_intensity_mm'] == 0]['demand_multiplier'].mean()) - 1) * 100:.0f}% demand surge")
        print("\n💡 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(insights['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        print("="*70)
        
        return insights
        
    def _interpret_correlation(self, corr):
        """Interpret correlation strength"""
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "Very Strong"
        elif abs_corr >= 0.5:
            return "Strong"
        elif abs_corr >= 0.3:
            return "Moderate"
        elif abs_corr >= 0.1:
            return "Weak"
        else:
            return "Very Weak"
            
    def run_full_weather_analysis(self, start_date='2024-06-01', end_date='2024-07-01', n_taxi_samples=15000):
        """Run complete weather-taxi correlation analysis"""
        print("Starting comprehensive weather-taxi correlation analysis...")
        print("="*60)
        
        # Generate data
        self.generate_weather_data(start_date, end_date)
        self.generate_taxi_data_with_weather_influence(n_taxi_samples)
        self.combine_data()
        
        # Run analyses
        self.calculate_correlations()
        self.analyze_rain_impact()
        self.temperature_analysis()
        insights = self.generate_insights_report()
        
        print("\n✅ Analysis completed! Check 'weather_analysis_results' folder for outputs.")
        print("📁 Files generated:")
        for file in self.output_dir.glob('*'):
            print(f"   • {file.name}")
            
        return insights
        
def main():
    """Main execution function"""
    analyzer = WeatherTaxiAnalyzer()
    analyzer.run_full_weather_analysis()
    
if __name__ == "__main__":
    main()
