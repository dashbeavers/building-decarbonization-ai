# AI for Building Decarbonization

AI-powered system for targeting building electrification outreach using energy data analysis and automated email generation.

## Overview

This project develops an intelligent pipeline that identifies optimal buildings for heat pump installations and generates personalized marketing outreach. The system processes municipal energy data, applies AI filtering criteria, and creates targeted communications for clean energy adoption.

## Project Structure

building-decarbonization-ai/
├── README.md                               # Project overview and documentation
├── src/
│   ├── building_decarbonization_system.py  # Main AI system and interactive dashboard
│   └── heat_pump_adoption_analysis.py      # Statistical analysis and correlation validation
├── docs/
│   └── technical_specification.pdf         # Complete system design and methodology
├── images/
│   ├── dashboard_interface.png             # Main search and filtering interface
│   ├── interactive_map.png                 # Heat vulnerability map visualization
│   ├── email_output.png                    # Generated email example
│   └── analysis_chart.png                  # Heat pump adoption correlation analysis
└── requirements.txt                        # Python dependencies

## Quick Start

1. Review the [Technical Specification](BuildingDecarbonizationAISystem.pdf) for complete methodology
2. Install dependencies: `pip install -r requirements.txt`
3. Run [`src/building_decarbonization_system.py`](src/building_decarbonization_system.py) for the main dashboard
4. Run [`src/heat_pump_adoption_analysis.py`](src/heat_pump_adoption_analysis.py) for statistical validation
5. See [images](images) below for expected output

## Features

### 🏗️ Intelligent Building Analysis
- **HVAC Replacement Window Detection**: Identifies buildings in optimal 15-20 year replacement cycles
- **Heat Pump Classification**: Uses energy consumption patterns to detect existing electrification
- **Priority Scoring**: Ranks buildings using NASA ECOSTRESS thermal vulnerability data

### 📧 Automated Outreach Generation
- **Personalized Cold Emails**: Generates location-specific marketing content
- **Social Proof Integration**: Finds nearby successful electrification examples
- **Local Context**: Incorporates neighborhood landmarks and thermal conditions

### 📊 Geospatial Visualization
- **Interactive Maps**: Real-time building data visualization with heat vulnerability overlays
- **Proximity Analysis**: Identifies relationships between nearby properties
- **Statistical Validation**: Correlation analysis between thermal stress and adoption rates

## Technical Architecture

### Core Pipeline (11-Step Process)
1. **API Data Ingestion** - Municipal building energy data (NYC Open Data)
2. **Data Cleaning** - Address deduplication and column standardization  
3. **Fuel Mix Classification** - Heat pump identification algorithm
4. **Thermal Data Integration** - NASA satellite heat island data merge
5. **Target Filtering** - HVAC age and building type criteria
6. **Priority Ranking** - Heat vulnerability and equipment age scoring
7. **Geographic Processing** - Coordinate mapping and spatial analysis
8. **Proximity Matching** - Nearby electrified building detection
9. **Email Generation** - Personalized outreach content creation
10. **Interactive Visualization** - Real-time dashboard and mapping
11. **Statistical Validation** - Adoption correlation analysis

## Code Structure

### Main System
`building_decarbonization_system.py` - Complete AI targeting and outreach platform
- Interactive dashboard with search and filtering
- Real-time map visualization with heat vulnerability layers
- Automated email generation with local context
- Proximity-based social proof identification

### Analysis Module  
`heat_pump_adoption_analysis.py` - Statistical validation and correlation analysis
- Pearson correlation between thermal stress and adoption rates
- Geospatial visualization of adoption patterns
- Building density analysis by ZIP code

## Technologies Used

- **Python** - Core data processing and analysis
- **Pandas/NumPy** - Data manipulation and numerical computing
- **GeoPandas** - Geospatial data processing
- **Folium** - Interactive mapping and visualization
- **Matplotlib/Seaborn** - Statistical plotting and analysis
- **IPython Widgets** - Interactive dashboard interface
- **Municipal APIs** - Real-time building energy data
- **NASA ECOSTRESS** - Satellite thermal vulnerability data

## Key Results

- **Personalized outreach** with neighborhood-specific landmarks and social proof
- **Heat vulnerability correlation** analysis showing relationship between thermal stress and adoption patterns
- **Real-time targeting** of optimal electrification candidates

## Installation & Usage

### Requirements
```python
numpy
pandas
geopandas
folium
matplotlib
seaborn
ipywidgets
requests
scipy
