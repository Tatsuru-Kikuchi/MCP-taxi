        
    def calculate_productivity_comparison(self):
        """Calculate comprehensive productivity comparison between traditional and AI-enhanced operations"""
        print("Calculating productivity comparison...")
        
        if self.traditional_data is None or self.ai_enhanced_data is None:
            raise ValueError("Both traditional and AI-enhanced data must be generated first")
            
        # Key metrics comparison
        traditional_metrics = {
            'avg_revenue_per_min': self.traditional_data['revenue_per_min'].mean(),
            'avg_trip_duration': self.traditional_data['trip_duration'].mean(),
            'avg_wait_time': self.traditional_data['wait_time'].mean(),
            'avg_utilization': self.traditional_data['utilization_rate'].mean(),
            'avg_daily_earnings': self.traditional_data['daily_earnings'].mean(),
            'avg_positioning_efficiency': self.traditional_data['positioning_efficiency'].mean()
        }
        
        ai_enhanced_metrics = {
            'avg_revenue_per_min': self.ai_enhanced_data['revenue_per_min'].mean(),
            'avg_trip_duration': self.ai_enhanced_data['trip_duration'].mean(),
            'avg_wait_time': self.ai_enhanced_data['wait_time'].mean(),
            'avg_utilization': self.ai_enhanced_data['utilization_rate'].mean(),
            'avg_daily_earnings': self.ai_enhanced_data['daily_earnings'].mean(),
            'avg_positioning_efficiency': self.ai_enhanced_data['positioning_efficiency'].mean(),
            'avg_prediction_accuracy': self.ai_enhanced_data['prediction_accuracy'].mean(),
            'avg_ai_confidence': self.ai_enhanced_data['ai_confidence_score'].mean()
        }
        
        # Calculate improvements
        improvements = {}
        for key in traditional_metrics:
            if key.startswith('avg_'):
                metric_name = key[4:]  # Remove 'avg_' prefix
                traditional_val = traditional_metrics[key]
                ai_val = ai_enhanced_metrics[key]
                
                # For metrics where lower is better (trip_duration, wait_time)
                if metric_name in ['trip_duration', 'wait_time']:
                    improvement_pct = ((traditional_val - ai_val) / traditional_val) * 100
                else:
                    improvement_pct = ((ai_val - traditional_val) / traditional_val) * 100
                    
                improvements[metric_name] = {
                    'traditional_value': round(traditional_val, 2),
                    'ai_enhanced_value': round(ai_val, 2),
                    'improvement_percent': round(improvement_pct, 1),
                    'improvement_direction': 'increase' if improvement_pct > 0 else 'decrease'
                }
        
        # Statistical significance testing
        significance_tests = {}
        metrics_to_test = ['revenue_per_min', 'trip_duration', 'wait_time', 'utilization_rate', 'daily_earnings']
        
        for metric in metrics_to_test:
            t_stat, p_value = ttest_ind(
                self.traditional_data[metric], 
                self.ai_enhanced_data[metric]
            )
            significance_tests[metric] = {
                't_statistic': round(t_stat, 4),
                'p_value': round(p_value, 6),
                'significant': p_value < 0.05
            }
            
        # Weather-specific analysis
        weather_comparison = self._analyze_weather_specific_performance()
        
        # ROI calculation
        roi_analysis = self._calculate_roi_analysis(traditional_metrics, ai_enhanced_metrics)
        
        self.comparison_results = {
            'traditional_metrics': traditional_metrics,
            'ai_enhanced_metrics': ai_enhanced_metrics,
            'improvements': improvements,
            'significance_tests': significance_tests,
            'weather_comparison': weather_comparison,
            'roi_analysis': roi_analysis,
            'analysis_summary': {
                'total_records_analyzed': len(self.traditional_data) + len(self.ai_enhanced_data),
                'analysis_date': datetime.now().isoformat(),
                'key_findings': [
                    f"AI increases revenue per minute by {improvements['revenue_per_min']['improvement_percent']:.1f}%",
                    f"Trip duration reduced by {improvements['trip_duration']['improvement_percent']:.1f}%",
                    f"Wait time reduced by {improvements['wait_time']['improvement_percent']:.1f}%",
                    f"Utilization improved by {improvements['utilization_rate']['improvement_percent']:.1f}%",
                    f"Daily earnings increased by {improvements['daily_earnings']['improvement_percent']:.1f}%"
                ]
            }
        }
        
        # Save results
        with open(self.output_dir / 'productivity_comparison.json', 'w') as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
            
        return self.comparison_results
        
    def _analyze_weather_specific_performance(self):
        """Analyze performance differences in specific weather conditions"""
        weather_conditions = ['clear', 'rainy', 'extreme_temp', 'poor_visibility']
        weather_comparison = {}
        
        for condition in weather_conditions:
            if condition == 'clear':
                trad_subset = self.traditional_data[
                    (self.traditional_data['rain_intensity'] == 0) & 
                    (self.traditional_data['temperature'].between(15, 25))
                ]
                ai_subset = self.ai_enhanced_data[
                    (self.ai_enhanced_data['rain_intensity'] == 0) & 
                    (self.ai_enhanced_data['temperature'].between(15, 25))
                ]
            elif condition == 'rainy':
                trad_subset = self.traditional_data[self.traditional_data['rain_intensity'] > 0]
                ai_subset = self.ai_enhanced_data[self.ai_enhanced_data['rain_intensity'] > 0]
            elif condition == 'extreme_temp':
                trad_subset = self.traditional_data[
                    (self.traditional_data['temperature'] < 5) | 
                    (self.traditional_data['temperature'] > 35)
                ]
                ai_subset = self.ai_enhanced_data[
                    (self.ai_enhanced_data['temperature'] < 5) | 
                    (self.ai_enhanced_data['temperature'] > 35)
                ]
            elif condition == 'poor_visibility':
                trad_subset = self.traditional_data[self.traditional_data['visibility'] < 5]
                ai_subset = self.ai_enhanced_data[self.ai_enhanced_data['visibility'] < 5]
                
            if len(trad_subset) > 0 and len(ai_subset) > 0:
                weather_comparison[condition] = {
                    'traditional_revenue': trad_subset['revenue_per_min'].mean(),
                    'ai_enhanced_revenue': ai_subset['revenue_per_min'].mean(),
                    'traditional_wait_time': trad_subset['wait_time'].mean(),
                    'ai_enhanced_wait_time': ai_subset['wait_time'].mean(),
                    'revenue_improvement': ((ai_subset['revenue_per_min'].mean() - trad_subset['revenue_per_min'].mean()) / trad_subset['revenue_per_min'].mean()) * 100,
                    'wait_time_reduction': ((trad_subset['wait_time'].mean() - ai_subset['wait_time'].mean()) / trad_subset['wait_time'].mean()) * 100,
                    'sample_size': {'traditional': len(trad_subset), 'ai_enhanced': len(ai_subset)}
                }
                
        return weather_comparison
        
    def _calculate_roi_analysis(self, traditional_metrics, ai_enhanced_metrics):
        """Calculate ROI for AI implementation"""
        
        # Annual calculations
        working_days_per_year = 300
        daily_earnings_improvement = ai_enhanced_metrics['avg_daily_earnings'] - traditional_metrics['avg_daily_earnings']
        annual_earnings_improvement = daily_earnings_improvement * working_days_per_year
        
        # Estimated AI implementation costs
        ai_development_cost = 150000  # ¥150,000 per driver for system development and deployment
        ai_monthly_operating_cost = 2500  # ¥2,500 per month per driver for data and processing
        ai_annual_operating_cost = ai_monthly_operating_cost * 12
        
        # ROI calculation
        first_year_net_benefit = annual_earnings_improvement - ai_development_cost - ai_annual_operating_cost
        roi_percentage = (first_year_net_benefit / ai_development_cost) * 100
        payback_period_months = ai_development_cost / (daily_earnings_improvement * 25)  # 25 working days per month
        
        return {
            'daily_earnings_improvement': round(daily_earnings_improvement),
            'annual_earnings_improvement': round(annual_earnings_improvement),
            'ai_development_cost': ai_development_cost,
            'ai_annual_operating_cost': ai_annual_operating_cost,
            'first_year_net_benefit': round(first_year_net_benefit),
            'roi_percentage': round(roi_percentage, 1),
            'payback_period_months': round(payback_period_months, 1),
            'break_even_analysis': {
                'break_even_achieved': first_year_net_benefit > 0,
                'years_to_break_even': max(0.1, payback_period_months / 12)
            }
        }
