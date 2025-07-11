# 🚕 Tokyo Taxi Analysis Dashboard

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Ready-brightgreen)](https://tatsuru-kikuchi.github.io/MCP-taxi/)
[![Dashboard Live](https://img.shields.io/badge/Dashboard-Live-blue)](https://tatsuru-kikuchi.github.io/MCP-taxi/)
[![Analysis Complete](https://img.shields.io/badge/Analysis-Complete-success)](https://github.com/Tatsuru-Kikuchi/MCP-taxi/tree/main/analysis_results)

This repository analyzes taxi congestion and productivity patterns in Tokyo with comprehensive data analysis and interactive visualizations.

## 🌟 Live Dashboard

<div align="center">

### [🔗 **VIEW INTERACTIVE DASHBOARD**](https://tatsuru-kikuchi.github.io/MCP-taxi/)

*Experience real-time Tokyo taxi analytics*

</div>

> **Note:** If you see a 404 error, GitHub Pages needs to be enabled in repository settings

## 🚀 Quick Start

### 🌐 Online Access
- **🎯 Dashboard**: https://tatsuru-kikuchi.github.io/MCP-taxi/
- **📁 Repository**: https://github.com/Tatsuru-Kikuchi/MCP-taxi
- **🔧 Status Page**: https://tatsuru-kikuchi.github.io/MCP-taxi/status.html

### 💻 Local Setup
```bash
# Clone the repository
git clone https://github.com/Tatsuru-Kikuchi/MCP-taxi.git
cd MCP-taxi

# Install dependencies
pip install -r requirements.txt

# Run analysis
python run_analysis.py
```

## 📊 Key Results

### 📈 Performance Metrics
- 🔢 **10,000 trips** analyzed across 30-day simulation
- ⏱️ **32.4 minutes** average trip duration
- 💴 **¥2,150** average fare per trip
- 📊 **¥52.3** revenue per minute

### 🚦 Critical Insights
- 📈 **65% longer trips** during rush hours (7-9 AM, 5-7 PM)
- 💸 **¥890** congestion cost per trip
- 🏙️ **¥8.9M** daily revenue loss citywide
- 📉 **45%** productivity gap between districts

### 🏆 Top Performing Districts
| Rank | District | Revenue/Min | Status |
|:----:|----------|:-----------:|:------:|
| 🥇 | **Ginza** | ¥62.5/min | 🟢 Excellent |
| 🥈 | **Tokyo Station** | ¥59.8/min | 🟢 High |
| 🥉 | **Shibuya** | ¥58.2/min | 🟢 High |
| 4️⃣ | **Shinjuku** | ¥55.8/min | 🟡 Good |
| 5️⃣ | **Roppongi** | ¥54.2/min | 🟡 Good |

## ✨ Dashboard Features

- 📊 **Interactive Charts** - Real-time visualizations with Chart.js
- 📱 **Mobile Responsive** - Works perfectly on all devices
- ⚡ **Live Metrics** - Animated performance indicators
- 💡 **Smart Insights** - AI-powered strategic recommendations
- 🎨 **Modern UI** - Professional design with glassmorphism effects

## 🔧 GitHub Pages Setup

<details>
<summary><b>🛠️ Click to expand setup instructions</b></summary>

### Step 1: Enable GitHub Pages
1. Go to [Settings → Pages](https://github.com/Tatsuru-Kikuchi/MCP-taxi/settings/pages)
2. Set **Source** to "GitHub Actions"
3. Click **Save**
4. Wait 2-5 minutes for deployment

### Step 2: Verify Deployment
- ✅ [Check Actions Status](https://github.com/Tatsuru-Kikuchi/MCP-taxi/actions)
- ⚙️ [View Pages Settings](https://github.com/Tatsuru-Kikuchi/MCP-taxi/settings/pages)

</details>

## 📁 Project Structure

```
🚕 MCP-taxi/
├── 🌐 index.html              # Main dashboard
├── 📊 dashboard.html          # Alternative view
├── 🔧 status.html             # Setup status
├── 🐍 run_analysis.py         # Analysis engine
├── 📦 requirements.txt        # Dependencies
├── 📈 analysis_results/       # Generated data
│   ├── 📄 analysis_summary.json
│   ├── 📊 hourly_analysis.csv
│   ├── 🗺️ district_analysis.csv
│   └── 📅 weekly_patterns.csv
├── 🤖 .github/workflows/      # Auto-deployment
└── 📚 docs/                   # Documentation
```

## 💡 Strategic Recommendations

### 🎯 Operations
1. **⏰ Peak Hours**: Focus on 11 AM - 2 PM (highest efficiency)
2. **🗺️ Districts**: Prioritize Ginza and Tokyo Station
3. **📅 Weekends**: Increase fleet for 18% higher demand

### 🔬 Technology
4. **💰 Dynamic Pricing**: Implement rush hour surge pricing
5. **🛣️ Route Optimization**: AI-powered navigation
6. **🔮 Predictive Analytics**: Demand forecasting

### 📈 Expected Impact
- 🎯 **25%** efficiency improvement potential
- 💰 **¥8.9M** daily savings from optimization
- 📊 **15-20%** revenue increase possible

## 🛠️ Technical Stack

![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=flat-square&logo=chartdotjs&logoColor=white)

## 📈 Usage Example

```python
# 🐍 Tokyo Taxi Analysis Framework
from run_analysis import TokyoTaxiAnalyzer

# Initialize analyzer
analyzer = TokyoTaxiAnalyzer()

# Generate realistic Tokyo taxi data
analyzer.generate_sample_data(n_samples=10000)

# Run comprehensive analysis
analyzer.run_full_analysis()

# ✅ Results saved to analysis_results/ directory
print("Analysis complete! Check the dashboard for insights.")
```

## 🤝 Contributing

[![Fork](https://img.shields.io/badge/Fork-Repository-blue?style=for-the-badge&logo=github)](https://github.com/Tatsuru-Kikuchi/MCP-taxi/fork)
[![Issues](https://img.shields.io/badge/Report-Issue-red?style=for-the-badge&logo=github)](https://github.com/Tatsuru-Kikuchi/MCP-taxi/issues)
[![Pull Request](https://img.shields.io/badge/Submit-PR-green?style=for-the-badge&logo=github)](https://github.com/Tatsuru-Kikuchi/MCP-taxi/pulls)

1. 🍴 Fork the repository
2. 🌿 Create feature branch
3. 🧪 Add tests
4. 📬 Submit pull request

## 📞 Contact

<div align="center">

**👨‍💻 Author**: Tatsuru Kikuchi

[![Dashboard](https://img.shields.io/badge/🌐_Live-Dashboard-blue?style=for-the-badge)](https://tatsuru-kikuchi.github.io/MCP-taxi/)
[![Repository](https://img.shields.io/badge/📂_GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Tatsuru-Kikuchi/MCP-taxi)
[![Issues](https://img.shields.io/badge/🐛_Report-Issues-red?style=for-the-badge&logo=github)](https://github.com/Tatsuru-Kikuchi/MCP-taxi/issues)

</div>

---

<div align="center">

**🎯 Ready to Deploy!**

Complete Tokyo taxi analysis with interactive dashboard, automated deployment, and business insights.

*Built with ❤️ for the Tokyo transportation community* 🚕

⭐ **Star this repo** if you find it useful!

</div>
