# Wpromote_project
# Marketing KPI Monitor (Streamlit)

A lightweight Streamlit dashboard for monitoring paid media performance across channels and campaigns.  
Tracks daily trends and core KPIs like **CTR, CPC, CPA, and ROAS** using a clean dataset structure.

## Features
- KPI calculations with safe division handling (no crash on zeros)
- Filter by channel / campaign
- Daily trend charts for CTR, CPC, CPA, ROAS
- Summary metrics (impressions, clicks, spend, revenue, conversions)

## Dashboard Preview

![Marketing KPI Dashboard](assets/dashboard.png)

## Dataset
Expected columns (CSV):

- date
- channel
- campaign
- creative_id
- impressions
- clicks
- conversions
- revenue
- spend
