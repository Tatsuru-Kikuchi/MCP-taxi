#!/usr/bin/env python3
"""
Complete AI-Enhanced Weather-Taxi Productivity Analysis
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

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AIWeatherProductivityAnalyzer:
    """Analyzes productivity gains from AI-enhanced weather intelligence for taxi drivers"""
    
    def __init__(self):
        self.traditional_data = None
        self.ai_enhanced_data = None
        self.comparison_results = {}
        self.output_dir = Path("ai_productivity_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def run_full_analysis(self, traditional_samples=5000, ai_samples=5000):
        """Run the complete AI productivity analysis"""
        print("🚀 Starting comprehensive AI weather productivity analysis...")
        print("="*70)
        
        # Generate data
        self.generate_traditional_taxi_data(traditional_samples)
        self.generate_ai_enhanced_data(ai_samples)
        
        # Run comparisons
        self.calculate_productivity_comparison()
        self.create_visualization_charts()
        report = self.generate_comprehensive_report()
        
        print("\\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")
        
        return report

def main():
    """Main execution function"""
    analyzer = AIWeatherProductivityAnalyzer()
    results = analyzer.run_full_analysis()
    
    # Print key findings
    print("\\n🎯 KEY FINDINGS:")
    for finding in results['analysis_summary']['key_findings']:
        print(f"   • {finding}")

if __name__ == "__main__":
    main()
