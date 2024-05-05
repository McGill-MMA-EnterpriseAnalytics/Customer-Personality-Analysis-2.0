# Customer Personality Analysis 2.0

### Team Members
- **Xiaorong Tian**
- **Kelly Liu**
- **Ivy Lou**
- **Jiaxuan Wang**
- **Zhiming Zhang**

## Project Overview
The Customer Personality Analysis 2.0 project builds upon insights gained from version 1.0 to deliver more refined marketing strategies. By using advanced clustering and causal inference models, this project seeks to enhance customer segmentation and provide tailored marketing suggestions that directly address business needs. We focus on improving the understanding of customer behavior through data analysis to support better engagement and increased sales.

## Objectives
- Utilize advanced data insights to **refine sales operations** and tailor strategies.
- Implement soft clustering models to **improve segmentation flexibility**.
- Explore causal relationships between customer demographics and key behavioral metrics to develop **customized marketing strategies**.

## Methodology
### Version 1.0
1. **Data Sources and Preprocessing**: 
   - Kaggle's Customer Personality Analysis dataset was the primary source.
   - Rigorous data preprocessing included outlier removal, iterative imputation for missing values, feature engineering, and one-hot encoding.

2. **Clustering Analysis**:
   - Elbow method was used to determine the optimal number of clusters, and K-means clustering provided customer segmentation.
   - Four primary personas were identified based on demographic and behavioral insights.

3. **Causal Inference Models**:
   - Explored relationships between marital status, income, and complaints on customer behavior using CausalML.
   - Identified dominant segments and provided preliminary marketing recommendations.

### Version 2.0
1. **Data Refinement**:
   - Transformed the income feature into **categorical variables** rather than continuous to explore more granular impacts.
   - Addressed feature dominance issues by focusing on **Recency, Frequency, and Monetary Value (RFM)** as key segments.

2. **Soft Clustering Analysis**:
   - Employed **Fuzzy C-Means (FCM)** and **Gaussian Mixture Model (GMM)** to reveal overlapping customer segments with greater flexibility.
   - Decision tree analysis used to **increase explainability** and determine key features affecting each cluster.

3. **Causal Inference Models**:
   - Integrated **H2O AutoML** to refine causal inference models.
   - Evaluated feature impact on each part of RFM to tailor marketing strategies accordingly.
   - Utilized MLflow for tracking model performance across iterations.

3. **Prediction Models**:
   - Utilized the clusters from clustering result as target to predict new customer's segment.

## Business Value
1. **Segmentation Impact**:
   - Tailored marketing strategies for individual clusters increase promotion effectiveness and customer engagement.
   - Data-driven insights lead to more personalized marketing campaigns.

2. **Granular Strategies**:
   - By focusing on RFM and understanding the specific drivers of customer engagement, marketing efforts can be more targeted.
   - Breaking income into categories helps reveal new behavioral patterns that can guide pricing and promotion strategies.

## Solution Architecture
1. **Model Packaging & Registration**:
   - Models registered with MLflow for version control.
   - Packaged models as pickle files for efficient deployment.

2. **CI/CD & Serving**:
   - H2O AutoML and MLflow ensure streamlined CI/CD pipelines.
   - Docker containers used for seamless model deployment and scaling.
   - FastAPI and Streamlit for interactive web app deployment.

3. **User Experience**:
   - MLflow tracking and visualization provide transparency and insight.
   - Actionable recommendations delivered to stakeholders for data-driven decision-making.

## Future Enhancements
- **Azure Kubernetes Service (AKS)** for scalable production environments.
- Refine **UX/UI** via Streamlit dashboards.
- Use **Azure Databricks Delta Lake** for data orchestration.
- Optimize retraining pipelines to address data drift.

## Conclusion
Version 2.0 introduces soft clustering models and granular feature analysis, leading to refined customer segmentation and better marketing strategies. Future work will focus on expanding the data sources, improving causal inference models, and incorporating insights into business strategies.
