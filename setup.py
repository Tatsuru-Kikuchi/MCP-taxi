#!/usr/bin/env python3
"""
Setup script for Tokyo Taxi Route Optimization
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tokyo-taxi-optimizer",
    version="1.0.0",
    author="Tatsuru Kikuchi",
    author_email="tatsuru.kikuchi@example.com",
    description="AI-powered route optimization for Tokyo taxi services using real-time traffic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tatsuru-Kikuchi/MCP-taxi",
    project_urls={
        "Bug Tracker": "https://github.com/Tatsuru-Kikuchi/MCP-taxi/issues",
        "Documentation": "https://github.com/Tatsuru-Kikuchi/MCP-taxi/blob/main/README.md",
        "Source Code": "https://github.com/Tatsuru-Kikuchi/MCP-taxi",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "scikit-learn>=1.3.0",
        ],
        "viz": [
            "folium>=0.14.0",
            "plotly>=5.15.0",
            "dash>=2.11.0",
        ],
        "geo": [
            "geopandas>=0.13.0",
            "geopy>=2.3.0",
        ],
        "performance": [
            "numba>=0.57.0",
        ],
        "all": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "scikit-learn>=1.3.0",
            "folium>=0.14.0",
            "plotly>=5.15.0",
            "dash>=2.11.0",
            "geopandas>=0.13.0",
            "geopy>=2.3.0",
            "numba>=0.57.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tokyo-taxi-optimize=run_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "tokyo",
        "taxi",
        "route optimization",
        "artificial intelligence",
        "machine learning",
        "traffic analysis",
        "transportation",
        "ODPT",
        "geospatial",
        "real-time data",
    ],
    zip_safe=False,
)
