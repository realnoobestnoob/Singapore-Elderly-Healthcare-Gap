# Singapore-Elderly-Healthcare-Infrastructure-Gap-Analysis
**Mapping eldercare demand and healthcare accessibility gaps across Singapore's subzones.**
[Link to app](https://singapore-elderly-healthcare-gap.streamlit.app/)

## Overview

SgHealth-Optimize is an interactive dashboard that combines population data, healthcare infrastructure locations, and hospital admissions records to identify where Singapore's aging population is at risk of being underserved by existing healthcare and eldercare facilities.

The dashboard uses unsupervised machine learning to classify each area into a risk tier based on how its elderly population growth compares to nearby infrastructure availability.

## Main Goal

The goal is to answer a simple but important question for each part of Singapore:

> *"Does this area have enough eldercare and healthcare infrastructure to support its elderly population?"*

To do this, the dashboard:

1. Classifies every subzone into a risk tier (e.g. **High Pressure**, **Emerging Pressure**, **Well-Served**) using clustering on demographic and infrastructure data. Outliers or anomalies are removed
2. Visualizes these risk tiers on an interactive map alongside the locations of heathcare and eldercare facilities
3. Shows historical trends in senior population and hospital admissions over time
4. Projects which areas are likely to shift into higher-risk tiers by 2030, based on population growth forecasts

## Why It Matters

Singapore's population is aging rapidly, and healthcare/eldercare infrastructure isn't always built where the need is greatest. Hence planning decisions are important

This dashboard turns scattered government datasets (population statistics, facility locations, hospital admissions) into a single, visual tool that highlights gaps before they become critical. 

## How It Works 

### 1. Data In
The project starts with three types of data:
- **Population data** (2011–2025) showing how many elderly residents live in each subzone, and how fast that number is growing
- **Infrastructure data** — the locations of eldercare centres, dementia day care/group therapy facilities, polyclinics, and hospitals
- **Hospital admissions data** — records of how many people are being admitted to hospitals over time

### 2. Risk Scoring
For each subzone, the dashboard calculates a **risk score** based on:
- How many elderly residents live there, and how fast that number is growing
- How close the nearest polyclinic and hospital are
- How dense the local eldercare/healthcare infrastructure is

### 3. Grouping Similar Areas (Clustering)
Using a machine learning clustering algrothim (Alggomerative Clustering), subzones with similar risk profiles are grouped together into three tiers:

- **High Pressure** — high elderly population, low infrastructure availability
- **Emerging Pressure** — growing elderly population, infrastructure may not keep pace
- **Well-Served** — adequate infrastructure relative to elderly population

### 4. Exploring the Dashboard
The dashboard has four pages:

| Page | What it shows |
|---|---|
| **Overview** | A summary of key statistics and what the dashboard covers |
| **Risk Map** | An interactive map of Singapore with each subzone color-coded by risk tier, plus markers for eldercare centres, dementia facilities, polyclinics, and hospitals |
| **Trends** | Charts showing senior population growth, hospital admissions over time, planning area breakdowns, and a forecast of future risk |
| **Cluster Analysis** | A closer look at what defines each risk tier |

### 5. Looking Ahead to 2030
The Trends → Forecast tab uses statistical forecasting to project how each subzone's elderly population might grow by 2030, and re-runs the clustering on these projections. This flags subzones that are likely to **move from "Emerging Pressure" into "High Pressure"**, giving planners a heads-up before problems become urgent.

## Tech Stack

- **Python** — core language
- **Streamlit** — dashboard framework
- **Folium** — interactive maps
- **Plotly** — charts and visualizations
- **scikit-learn** — clustering algorithm (Agglomerative, Ward linkage)
- **Prophet** — population forecasting
- **pandas / numpy** — data processing

## Data Sources

- SingStat population statistics
- Data.gov.sg infrastructure datasets (eldercare, dementia care, polyclinics, hospitals)
- MOH hospital admissions data
