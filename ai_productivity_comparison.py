# Full pipeline integration
from run_ai_productivity_analysis import AIWeatherProductivityAnalyzer
from ai_productivity_visualization_methods import AIProductivityVisualizer

# Run analysis
analyzer = AIWeatherProductivityAnalyzer()
results = analyzer.run_full_analysis()

# Create visualizations
visualizer = AIProductivityVisualizer()
visualizer.load_data(
    traditional_data=analyzer.traditional_data,
    ai_enhanced_data=analyzer.ai_enhanced_data,
    comparison_results=results
)

visualizer.create_comprehensive_dashboard()
visualizer.create_weather_impact_analysis()
