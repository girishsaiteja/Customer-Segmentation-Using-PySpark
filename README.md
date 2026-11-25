# Customer-Segmentation-Using-PySpark

# Customer Segmentation Using PySpark

## Project Overview

Customer Segmentation Using PySpark is a Big Data analytics and machine learning project that groups customers based on their purchasing behaviour. It uses Apache Spark's distributed data processing capabilities and the K-Means clustering algorithm to identify meaningful customer segments.

The project focuses on transforming raw customer transaction data into meaningful insights that can help businesses understand customer behaviour and develop targeted marketing strategies.

## Objectives

- Process and analyze customer transaction data using PySpark.
- Perform data cleaning and preprocessing.
- Extract meaningful customer features using RFM analysis.
- Apply feature scaling to prepare data for clustering.
- Segment customers using the K-Means clustering algorithm.
- Evaluate clustering performance using the Silhouette Score.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Big Data Framework | Apache Spark, PySpark |
| Data Processing | PySpark DataFrames |
| Machine Learning | Spark MLlib |
| Algorithms | K-Means Clustering |
| Feature Engineering | RFM Analysis |
| Feature Scaling | StandardScaler |
| Evaluation | Silhouette Score |

## Project Workflow

1. **Spark Session Initialization** – Initialize an Apache Spark session for distributed data processing.
2. **Data Loading** – Import the customer transaction dataset into a PySpark DataFrame.
3. **Data Cleaning** – Handle missing values, remove duplicate records, and prepare the dataset.
4. **Feature Engineering** – Extract Recency, Frequency, and Monetary (RFM) features to understand customer purchasing patterns.
5. **Feature Assembly** – Use VectorAssembler to combine selected features into a feature vector.
6. **Feature Scaling** – Apply StandardScaler to standardize features before clustering.
7. **Customer Clustering** – Implement K-Means to group customers based on purchasing behaviour.
8. **Model Evaluation** – Evaluate clustering quality using the Silhouette Score.
9. **Results Analysis** – Analyze the generated customer clusters and their characteristics.

## Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised machine learning algorithm that divides customers into a predefined number of clusters based on similarities in their feature values.

The algorithm assigns customers to clusters by minimizing the distance between data points and their respective cluster centroids.

### Evaluation Metric: Silhouette Score

The Silhouette Score measures how well each customer fits within its assigned cluster compared with other clusters.

- A score closer to **1** indicates well-separated clusters.
- A score around **0** indicates overlapping clusters.
- A negative score may indicate incorrect cluster assignments.

## Project Structure

```text
Customer-Segmentation-Using-PySpark/
│
├── dataset/
│   └── customer_transactions.csv
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
├── 12305666.pdf
│
├── README.md
│
└── requirements.txt
```

*Note: Adjust the folder and file names to match your actual repository structure.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/girishsaiteja/Customer-Segmentation-Using-PySpark.git
```

### 2. Navigate to the Project Directory

```bash
cd Customer-Segmentation-Using-PySpark
```

### 3. Install Dependencies

```bash
pip install pyspark pandas numpy matplotlib
```

### 4. Run the Project

Open the Jupyter Notebook or Python script and execute the cells to perform customer segmentation.

## Key Learning Outcomes

- Hands-on experience with Apache Spark and PySpark.
- Understanding distributed data processing using Spark DataFrames.
- Practical implementation of RFM-based feature engineering.
- Application of unsupervised machine learning techniques.
- Understanding feature scaling and clustering evaluation.
- Experience building a complete Big Data machine learning workflow.

## Author

**P. Girish Sai Teja**

B.Tech – Computer Science and Engineering (Data Science)

GitHub: [girishsaiteja](https://github.com/girishsaiteja)

---

*This project was developed as part of academic work in Big Data Analytics and Cluster Computing.*
