# 📈 End-to-End MLOps Stock Forecasting System

> A production-grade Machine Learning system that doesn't just predict stock prices — it monitors itself, detects when it becomes outdated, retrains automatically, and redeploys without human intervention.

---

## 🧭 Table of Contents

- [Overview](#overview)
- [Supported Stocks](#supported-stocks)
- [System Architecture](#system-architecture)
- [Pipeline Breakdown](#pipeline-breakdown)
  - [1. Data Collection](#1-data-collection)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Model Training](#3-model-training)
  - [4. Model Evaluation](#4-model-evaluation)
  - [5. Experiment Tracking](#5-experiment-tracking)
  - [6. Model Storage](#6-model-storage)
  - [7. Drift Detection](#7-drift-detection)
  - [8. Automated Retraining](#8-automated-retraining)
  - [9. API & Deployment](#9-api--deployment)
- [Tech Stack](#tech-stack)
- [Key Highlights](#key-highlights)

---

## Overview

This project implements a **self-maintaining MLOps pipeline** for forecasting NSE stock prices. The core objective goes beyond accuracy — it builds a system capable of detecting model degradation over time and automatically recovering from it.

Most ML projects stop at training a model. This project addresses the full lifecycle:

```
Data Collection → Training → Evaluation → Tracking → Deployment → Monitoring → Auto-Retraining
```

---

## Supported Stocks

The system currently supports **10 major NSE-listed stocks**, including:

| Company | Ticker |
|---|---|
| Tata Consultancy Services | TCS |
| Reliance Industries | RELIANCE |
| Infosys | INFY |
| HDFC Bank | HDFCBANK |
| ICICI Bank | ICICIBANK |
| State Bank of India | SBIN |
| ...and 4 more | — |

---

## System Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   GitHub Actions                    │
                    │           (Automated Retraining Trigger)            │
                    └──────────────────────┬──────────────────────────────┘
                                           │
         ┌─────────────┐        ┌──────────▼───────────┐       ┌──────────────────┐
         │  yfinance   │───────▶│   Training Pipeline   │──────▶│  Hugging Face Hub│
         │  Yahoo Fin  │        │  (Prophet Forecaster) │       │  (Model Storage) │
         └─────────────┘        └──────────┬────────────┘       └──────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │    MLflow   │
                                    │  (DagsHub)  │
                                    └──────┬──────┘
                                           │
                         ┌─────────────────▼──────────────────┐
                         │          Drift Detection            │
                         │  Layer 1: Data Drift (KS Test)      │
                         │  Layer 2: Model Drift (Error Bias)  │
                         │  AND Gate → Retrain if both true    │
                         └─────────────────┬──────────────────┘
                                           │
                              ┌────────────▼───────────┐
                              │   FastAPI Backend       │
                              │   Docker Container      │
                              │   Deployed on Render    │
                              └────────────────────────┘
```

---

## Pipeline Breakdown

### 1. Data Collection

Historical stock data is fetched using the **yfinance** library from Yahoo Finance.

- **Data span:** ~5 years of historical daily data per stock
- **Target feature:** Daily closing prices
- **Source:** Yahoo Finance via `yfinance` Python library

---

### 2. Data Preparation

The collected data is split into three purpose-specific datasets:

| Dataset | Purpose |
|---|---|
| **Full Historical Dataset** | Used for training the forecasting model |
| **Reference Dataset** | Represents past market behavior (baseline for drift detection) |
| **Current Dataset** | Contains the latest market data (compared against reference) |

These three datasets power both model training and the drift detection system.

---

### 3. Model Training

The forecasting model is built using **[Prophet](https://facebook.github.io/prophet/)**, a time-series library developed by Meta.

**Why Prophet?**

- Automatically learns long-term price trends without manual feature engineering
- Captures weekly and yearly seasonality patterns (e.g., Fridays behaving differently from Mondays)
- Handles missing data and outliers gracefully
- Produces interpretable confidence intervals out of the box

---

### 4. Model Evaluation

To prevent overfitting and ensure real-world performance, the model is evaluated on held-out data:

- The **latest 30 days** are reserved as the test set
- The model trains on all data prior to this window and predicts forward
- Performance is measured using three metrics:

| Metric | Full Name | What It Measures |
|---|---|---|
| **MAE** | Mean Absolute Error | Average prediction deviation in ₹ |
| **RMSE** | Root Mean Squared Error | Penalizes larger errors more heavily |
| **MAPE** | Mean Absolute Percentage Error | Relative accuracy as a percentage |

---

### 5. Experiment Tracking

Every training run is logged with **MLflow**, hosted remotely on **DagsHub**.

Tracked per experiment run:
- Model hyperparameters
- Evaluation metrics (MAE, RMSE, MAPE)
- Trained model artifact

This enables comparison across model versions and full reproducibility via a web interface.

---

### 6. Model Storage

After training, the model is serialized as a **pickle file** and uploaded to **Hugging Face Hub**.

- Acts as centralized cloud storage for model versioning
- Enables the API to load the latest model at inference time
- Supports rollback to previous versions if needed

---

### 7. Drift Detection

> **This is the most critical and differentiating component of the project.**

A trained model reflects patterns from historical data. Over time, market behavior shifts — and a model that was accurate six months ago may produce poor predictions today without any visible warning.

To address this, the system implements a **two-layer drift detection mechanism**:

#### Layer 1: Data Drift

Checks whether the **distribution of stock returns** has shifted significantly between the reference period and the current period.

- Uses statistical comparison of return distributions
- Detects changes in volatility, trend direction, or market regime

#### Layer 2: Model Drift

Checks whether the model is producing **systematically biased predictions**.

- Compares actual prices against model predictions
- Analyzes prediction error patterns over recent data

#### AND Gate Logic

```
Data Drift Detected?  →  YES ─┐
                               ├─▶  AND  ─▶  Trigger Retraining
Model Drift Detected? →  YES ─┘
```

Retraining is triggered **only if both conditions are true simultaneously**. This avoids unnecessary retraining due to short-term market noise and eliminates false alarms.

---

### 8. Automated Retraining

When drift is confirmed, **GitHub Actions** automatically initiates the full retraining pipeline:

```
Drift Detected
     │
     ▼
Fetch Latest Data (yfinance)
     │
     ▼
Retrain Prophet Model
     │
     ▼
Evaluate on Held-Out Window
     │
     ▼
Log to MLflow / DagsHub
     │
     ▼
Upload Updated Model to Hugging Face Hub
     │
     ▼
API Serves New Model on Next Request
```

No manual intervention is required at any stage.

---

### 9. API & Deployment

A **FastAPI** backend serves model predictions via REST API.

**Request parameters:**
- Stock ticker symbol (e.g., `TCS`, `RELIANCE`)
- Number of future days to forecast

**Response includes:**
- Forecasted prices for the requested horizon
- Upper and lower confidence intervals per day

The entire application is **containerized with Docker** and deployed on **[Render](https://render.com)**.

---

## Tech Stack

| Category | Tool / Library |
|---|---|
| Data Collection | `yfinance`, Yahoo Finance |
| Forecasting Model | `Prophet` (Meta) |
| Experiment Tracking | `MLflow` + DagsHub |
| Model Storage | Hugging Face Hub |
| Drift Detection | Custom statistical layers |
| CI/CD & Automation | GitHub Actions |
| API Framework | FastAPI |
| Containerization | Docker |
| Deployment | Render |

---

## Key Highlights

- **Production mindset:** Built as a system, not a notebook — every component is modular and automated
- **Self-healing pipeline:** Detects its own degradation and retrains without human input
- **Smart retraining logic:** AND gate prevents costly, unnecessary retraining triggered by noise
- **Full observability:** Every training run is logged, versioned, and comparable via MLflow + DagsHub
- **End-to-end coverage:** From raw data ingestion to live API serving — the entire ML lifecycle in one project
