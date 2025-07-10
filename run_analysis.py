#!/usr/bin/env python3
"""
Tokyo Taxi Analysis Framework
This script performs comprehensive analysis of taxi congestion and productivity in Tokyo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TokyoTaxiAnalyzer:
    """Tokyo Taxi Data Analyzer"""
    
    def __init__(self):
        self.data = None
        self.results = {}
        self.output_dir = Path("analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_sample_data(self, n_samples=10000):
        """Generate realistic sample taxi data for Tokyo"""
        print("Generating sample Tokyo taxi data...")
        
        # Tokyo districts
        districts = [
            'Shibuya', 'Shinjuku', 'Ginza', 'Akihabara', 'Harajuku',
            'Roppongi', 'Asakusa', 'Ikebukuro', 'Ueno', 'Tokyo Station',
            'Akasaka', 'Ebisu', 'Marunouchi', 'Odaiba', 'Meguro'
        ]
        
        # Generate timestamps for the last 30 days
        start_date = datetime.now() - timedelta(days=30)
        
        np.random.seed(42)  # For reproducible results
        
        data = []
        for _ in range(n_samples):
            timestamp = start_date + timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Rush hour and weekend effects
            is_rush_hour = hour in [7, 8, 9, 17, 18, 19, 20]
            is_weekend = day_of_week >= 5
            
            # Base demand multipliers
            demand_multiplier = 1.0
            if is_rush_hour:
                demand_multiplier *= 2.5
            if is_weekend:
                demand_multiplier *= 1.8
            if hour >= 22 or hour <= 5:  # Night hours
                demand_multiplier *= 1.3
                
            pickup_district = np.random.choice(districts, p=self._get_district_probabilities(hour, is_weekend))
            dropoff_district = np.random.choice(districts)
            
            # Trip duration (minutes) - influenced by congestion
            base_duration = np.random.normal(25, 10)
            congestion_factor = demand_multiplier * 0.3
            trip_duration = max(5, base_duration * (1 + congestion_factor))
            
            # Distance (km)
            distance = np.random.normal(8, 4)
            distance = max(0.5, distance)
            
            # Fare calculation
            base_fare = 420  # Base fare in yen
            distance_fare = distance * 90
            time_fare = (trip_duration / 60) * 90
            surge_multiplier = 1 + (demand_multiplier - 1) * 0.5
            total_fare = (base_fare + distance_fare + time_fare) * surge_multiplier
            
            # Wait time for pickup
            wait_time = np.random.exponential(5) * demand_multiplier
            
            data.append({
                'timestamp': timestamp,
                'pickup_district': pickup_district,
                'dropoff_district': dropoff_district,
                'trip_duration_minutes': round(trip_duration, 1),
                'distance_km': round(distance, 2),
                'fare_yen': round(total_fare),
                'wait_time_minutes': round(wait_time, 1),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_rush_hour': is_rush_hour,
                'is_weekend': is_weekend,
                'demand_level': self._categorize_demand(demand_multiplier)
            })
            
        self.data = pd.DataFrame(data)
        self.data['date'] = self.data['timestamp'].dt.date
        
        # Save raw data
        self.data.to_csv(self.output_dir / 'tokyo_taxi_data.csv', index=False)
        print(f"Generated {len(self.data)} taxi trip records")
        
    def _get_district_probabilities(self, hour, is_weekend):
        """Get pickup probabilities for different districts based on time"""
        # Business districts more popular during weekdays/business hours
        # Entertainment districts more popular during evenings/weekends
        
        base_probs = np.array([0.12, 0.15, 0.10, 0.08, 0.07, 0.09, 0.06, 0.08, 0.05, 0.10, 0.04, 0.02, 0.02, 0.01, 0.01])
        
        if 9 <= hour <= 17 and not is_weekend:  # Business hours
            # Boost business districts
            business_boost = [1.5, 1.3, 2.0, 1.0, 0.8, 0.9, 0.7, 1.0, 0.8, 2.5, 1.8, 1.2, 2.2, 0.6, 1.0]
            base_probs *= business_boost
        elif (hour >= 19 or hour <= 2) or is_weekend:  # Evening/night or weekend
            # Boost entertainment districts
            entertainment_boost = [2.0, 2.5, 1.2, 1.5, 2.0, 2.8, 1.3, 1.8, 1.0, 1.0, 1.5, 1.8, 0.8, 1.5, 1.2]
            base_probs *= entertainment_boost
            
        return base_probs / base_probs.sum()
    
    def _categorize_demand(self, multiplier):
        """Categorize demand level"""
        if multiplier < 1.2:
            return 'Low'
        elif multiplier < 2.0:
            return 'Medium'
        else:
            return 'High'
            
    def analyze_congestion_patterns(self):
        """Analyze congestion patterns throughout the day and week"""
        print("Analyzing congestion patterns...")
        
        # Hourly congestion analysis
        hourly_stats = self.data.groupby('hour').agg({
            'trip_duration_minutes': ['mean', 'std'],
            'wait_time_minutes': ['mean', 'std'],
            'fare_yen': 'mean'
        }).round(2)
        
        # Daily congestion analysis
        daily_stats = self.data.groupby('day_of_week').agg({
            'trip_duration_minutes': ['mean', 'std'],
            'wait_time_minutes': ['mean', 'std'],
            'fare_yen': 'mean'
        }).round(2)
        
        # District congestion analysis
        district_stats = self.data.groupby('pickup_district').agg({
            'trip_duration_minutes': ['mean', 'count'],
            'wait_time_minutes': 'mean',
            'fare_yen': 'mean'
        }).round(2)
        
        self.results['hourly_congestion'] = hourly_stats
        self.results['daily_congestion'] = daily_stats
        self.results['district_congestion'] = district_stats
        
        # Save results
        hourly_stats.to_csv(self.output_dir / 'hourly_congestion.csv')
        daily_stats.to_csv(self.output_dir / 'daily_congestion.csv')
        district_stats.to_csv(self.output_dir / 'district_congestion.csv')
        
        print("Congestion analysis completed")
        
    def analyze_productivity_metrics(self):
        """Analyze taxi productivity metrics"""
        print("Analyzing productivity metrics...")
        
        # Calculate productivity metrics
        self.data['revenue_per_minute'] = self.data['fare_yen'] / (self.data['trip_duration_minutes'] + self.data['wait_time_minutes'])
        self.data['trips_per_hour'] = 60 / (self.data['trip_duration_minutes'] + self.data['wait_time_minutes'])
        
        # Productivity by time of day
        productivity_hourly = self.data.groupby('hour').agg({
            'revenue_per_minute': ['mean', 'std'],
            'trips_per_hour': ['mean', 'std'],
            'fare_yen': 'mean'
        }).round(2)
        
        # Productivity by district
        productivity_district = self.data.groupby('pickup_district').agg({
            'revenue_per_minute': ['mean', 'std'],
            'trips_per_hour': ['mean', 'std'],
            'fare_yen': 'mean'
        }).round(2)
        
        # Demand level analysis
        demand_analysis = self.data.groupby('demand_level').agg({
            'trip_duration_minutes': 'mean',
            'wait_time_minutes': 'mean',
            'fare_yen': 'mean',
            'revenue_per_minute': 'mean'
        }).round(2)
        
        self.results['productivity_hourly'] = productivity_hourly
        self.results['productivity_district'] = productivity_district
        self.results['demand_analysis'] = demand_analysis
        
        # Save results
        productivity_hourly.to_csv(self.output_dir / 'productivity_hourly.csv')
        productivity_district.to_csv(self.output_dir / 'productivity_district.csv')
        demand_analysis.to_csv(self.output_dir / 'demand_analysis.csv')
        
        print("Productivity analysis completed")
        
    def generate_visualizations(self):
        """Generate visualization plots"""
        print("Generating visualizations...")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Hourly trip duration and wait times
        ax1 = plt.subplot(3, 3, 1)
        hourly_data = self.data.groupby('hour')[['trip_duration_minutes', 'wait_time_minutes']].mean()
        hourly_data.plot(kind='bar', ax=ax1)
        ax1.set_title('Average Trip Duration & Wait Time by Hour')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Minutes')
        plt.xticks(rotation=45)
        
        # 2. Demand level distribution
        ax2 = plt.subplot(3, 3, 2)
        demand_counts = self.data['demand_level'].value_counts()
        demand_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Distribution of Demand Levels')
        
        # 3. District pickup frequency
        ax3 = plt.subplot(3, 3, 3)
        district_counts = self.data['pickup_district'].value_counts().head(10)
        district_counts.plot(kind='barh', ax=ax3)
        ax3.set_title('Top 10 Pickup Districts')
        
        # 4. Revenue per minute by hour
        ax4 = plt.subplot(3, 3, 4)
        hourly_revenue = self.data.groupby('hour')['revenue_per_minute'].mean()
        hourly_revenue.plot(kind='line', marker='o', ax=ax4)
        ax4.set_title('Revenue per Minute by Hour')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Yen per Minute')
        
        # 5. Weekly pattern
        ax5 = plt.subplot(3, 3, 5)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_data = self.data.groupby('day_of_week')['trip_duration_minutes'].mean()
        weekly_data.index = days
        weekly_data.plot(kind='bar', ax=ax5)
        ax5.set_title('Average Trip Duration by Day of Week')
        ax5.set_ylabel('Minutes')
        plt.xticks(rotation=45)
        
        # 6. Fare distribution
        ax6 = plt.subplot(3, 3, 6)
        self.data['fare_yen'].hist(bins=50, ax=ax6, alpha=0.7)
        ax6.set_title('Distribution of Taxi Fares')
        ax6.set_xlabel('Fare (Yen)')
        ax6.set_ylabel('Frequency')
        
        # 7. Distance vs Duration scatter
        ax7 = plt.subplot(3, 3, 7)
        scatter = ax7.scatter(self.data['distance_km'], self.data['trip_duration_minutes'], 
                            c=self.data['fare_yen'], alpha=0.6, cmap='viridis')
        ax7.set_title('Distance vs Trip Duration')
        ax7.set_xlabel('Distance (km)')
        ax7.set_ylabel('Duration (minutes)')
        plt.colorbar(scatter, ax=ax7, label='Fare (Yen)')
        
        # 8. Rush hour comparison
        ax8 = plt.subplot(3, 3, 8)
        rush_comparison = self.data.groupby('is_rush_hour')[['trip_duration_minutes', 'wait_time_minutes', 'fare_yen']].mean()
        rush_comparison.index = ['Non-Rush', 'Rush Hour']
        rush_comparison.plot(kind='bar', ax=ax8)
        ax8.set_title('Rush Hour vs Non-Rush Hour Comparison')
        plt.xticks(rotation=45)
        
        # 9. Correlation heatmap
        ax9 = plt.subplot(3, 3, 9)
        numeric_cols = ['trip_duration_minutes', 'distance_km', 'fare_yen', 'wait_time_minutes', 'revenue_per_minute']
        correlation_matrix = self.data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9)
        ax9.set_title('Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tokyo_taxi_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("Generating summary report...")
        
        # Calculate key metrics
        total_trips = len(self.data)
        avg_trip_duration = self.data['trip_duration_minutes'].mean()
        avg_wait_time = self.data['wait_time_minutes'].mean()
        avg_fare = self.data['fare_yen'].mean()
        avg_revenue_per_minute = self.data['revenue_per_minute'].mean()
        
        # Peak hours analysis
        peak_hour = self.data.groupby('hour')['trip_duration_minutes'].mean().idxmax()
        best_revenue_hour = self.data.groupby('hour')['revenue_per_minute'].mean().idxmax()
        
        # Best and worst districts
        best_district = self.data.groupby('pickup_district')['revenue_per_minute'].mean().idxmax()
        worst_district = self.data.groupby('pickup_district')['revenue_per_minute'].mean().idxmin()
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'data_period': f"{self.data['date'].min()} to {self.data['date'].max()}",
            'total_trips_analyzed': total_trips,
            'key_metrics': {
                'average_trip_duration_minutes': round(avg_trip_duration, 2),
                'average_wait_time_minutes': round(avg_wait_time, 2),
                'average_fare_yen': round(avg_fare, 2),
                'average_revenue_per_minute_yen': round(avg_revenue_per_minute, 2)
            },
            'congestion_insights': {
                'peak_congestion_hour': int(peak_hour),
                'rush_hour_impact': {
                    'avg_duration_rush': round(self.data[self.data['is_rush_hour']]['trip_duration_minutes'].mean(), 2),
                    'avg_duration_non_rush': round(self.data[~self.data['is_rush_hour']]['trip_duration_minutes'].mean(), 2)
                }
            },
            'productivity_insights': {
                'best_revenue_hour': int(best_revenue_hour),
                'most_productive_district': best_district,
                'least_productive_district': worst_district,
                'weekend_vs_weekday': {
                    'weekend_avg_revenue': round(self.data[self.data['is_weekend']]['revenue_per_minute'].mean(), 2),
                    'weekday_avg_revenue': round(self.data[~self.data['is_weekend']]['revenue_per_minute'].mean(), 2)
                }
            },
            'recommendations': [
                f"Focus operations during hour {best_revenue_hour} for maximum revenue",
                f"Increase taxi availability in {best_district} district",
                "Consider surge pricing during rush hours to balance supply and demand",
                "Optimize routes in high-congestion areas to reduce trip duration"
            ]
        }
        
        # Save report
        with open(self.output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Print summary
        print("\n" + "="*60)
        print("           TOKYO TAXI ANALYSIS SUMMARY")
        print("="*60)
        print(f"Analysis Period: {report['data_period']}")
        print(f"Total Trips Analyzed: {report['total_trips_analyzed']:,}")
        print("\nKey Metrics:")
        print(f"  • Average Trip Duration: {report['key_metrics']['average_trip_duration_minutes']:.1f} minutes")
        print(f"  • Average Wait Time: {report['key_metrics']['average_wait_time_minutes']:.1f} minutes")
        print(f"  • Average Fare: ¥{report['key_metrics']['average_fare_yen']:,.0f}")
        print(f"  • Average Revenue/Minute: ¥{report['key_metrics']['average_revenue_per_minute_yen']:.1f}")
        print("\nKey Insights:")
        print(f"  • Peak congestion occurs at {report['congestion_insights']['peak_congestion_hour']}:00")
        print(f"  • Best revenue hour: {report['productivity_insights']['best_revenue_hour']}:00")
        print(f"  • Most productive district: {report['productivity_insights']['most_productive_district']}")
        print(f"  • Least productive district: {report['productivity_insights']['least_productive_district']}")
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        print("="*60)
        
        self.results['summary_report'] = report
        print("Summary report saved to analysis_summary.json")
        
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Tokyo Taxi Analysis Framework...")
        print("="*50)
        
        # Generate sample data if not provided
        if self.data is None:
            self.generate_sample_data()
            
        # Run all analyses
        self.analyze_congestion_patterns()
        self.analyze_productivity_metrics()
        self.generate_visualizations()
        self.generate_summary_report()
        
        print("\nAnalysis completed! Check the 'analysis_results' folder for outputs.")
        print("Files generated:")
        for file in self.output_dir.glob('*'):
            print(f"  • {file.name}")
            
def main():
    """Main execution function"""
    analyzer = TokyoTaxiAnalyzer()
    analyzer.run_full_analysis()
    
if __name__ == "__main__":
    main()
