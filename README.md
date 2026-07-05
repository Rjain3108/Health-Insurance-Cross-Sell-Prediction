# Health Insurance Cross-Sell Prediction System

> An AI-powered Machine Learning application that predicts whether an existing health insurance customer is likely to purchase vehicle insurance.

---

## Project Overview

Marketing campaigns in insurance companies are often sent to every customer, resulting in:

- Low conversion rates
- High customer acquisition costs
- Inefficient use of marketing resources

This project uses Machine Learning to predict customers who are most likely to purchase vehicle insurance, enabling targeted marketing campaigns and improving business outcomes.

---

## Domain

**Insurance**

---

## Problem Statement

An insurance company currently offers health insurance to its customers. The objective is to identify customers who are most likely to purchase vehicle insurance.

Instead of contacting every customer, the system predicts purchase probability so that marketing teams can prioritize high-potential customers.

---

## Business Objectives

- Increase policy sales
- Improve campaign conversion rate
- Reduce marketing cost
- Improve customer targeting
- Optimize resource utilization

---

## Stakeholders

| Stakeholder | Goal |
|-------------|------|
| Marketing Team | Target interested customers |
| Sales Team | Increase policy sales |
| Customers | Receive relevant offers |
| Insurance Company | Improve revenue and reduce campaign costs |

---

# Functional Requirements

- Load customer information
- Perform data preprocessing
- Feature engineering
- Predict purchase probability
- Display prediction score
- Explain predictions using SHAP
- Support batch prediction
- Store prediction history

---

# Non-Functional Requirements

- Prediction latency below **500 ms**
- Availability above **99%**
- Secure REST APIs
- Explainable AI predictions
- Scalable architecture
- Maintainable codebase
- Reproducible ML pipeline

---

# Performance Goals

| Metric | Target |
|----------|---------|
| ROC-AUC | > 0.88 |
| Recall | > 75% |
| Precision | > 40% |
| Response Time | < 500 ms |
| Uptime | 99% |
| Batch Prediction | 10,000 customers |

---

# GR4ML Views

The project includes the following architectural views:

- Business View
- Analytics Design View
- Data Preparation View

---

# Quality Requirements

## 1. Accuracy

Accurate predictions reduce unnecessary marketing costs and improve customer targeting.

**Target**

- ROC-AUC > 0.88

---

## 2. Explainability

Predictions should be transparent and understandable by business users.

**Technology**

- SHAP (SHapley Additive exPlanations)

---

## 3. Scalability

The application should support predictions for millions of customers with minimal performance degradation.

**Deployment Stack**

- FastAPI
- Docker
- Azure

---

# Machine Learning Pipeline

Historical Customer Dataset

↓

Data Validation

↓

Data Preprocessing

↓

Encoding

↓

Feature Engineering

↓

Train/Test Split

↓

Hyperparameter Optimization

↓

Model Training

↓

Evaluation

↓

SHAP Explainability

↓

Save Model

---

# Technology Stack

## Machine Learning

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- SHAP

---

## Backend

- FastAPI
- REST APIs

---

## Frontend

- Streamlit

---

## Deployment

- Docker
- Azure App Service
- Azure Container Registry

---

## CI/CD

- GitHub
- GitHub Actions

---

## Monitoring

- Application Logs
- Prediction Logs
- Model Metrics
- API Health Monitoring

---

# ML Components

- Input Validation
- Data Validation
- Preprocessing
- Encoding
- Feature Engineering
- XGBoost Model
- Prediction Service
- SHAP Explainability

---

# Non-ML Components

- Streamlit Dashboard
- FastAPI Backend
- REST APIs
- Model Repository
- Logging
- Docker
- GitHub Actions
- Azure Deployment

---

# Repository Structure
