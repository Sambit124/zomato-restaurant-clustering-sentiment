# Zomato Restaurant Clustering & Sentiment Analysis

Machine Learning capstone project (Unsupervised ML + NLP) using Zomato restaurant data from Hyderabad.

## Problem statement
Zomato hosts thousands of restaurants and customer reviews, but a star rating alone does not tell customers which
restaurants are really good, or tell Zomato where to act. This project analyses review sentiment, studies how cost,
cuisine, reviewer behaviour and ratings relate, and clusters restaurants into segments so that customers can find the
best and best-value restaurants and Zomato can target each group with the right strategy.

## Data
| File | Content |
|---|---|
| `Zomato_Restaurant_names_and_Metadata.csv` | 105 restaurants: cost, cuisines, collections, timings |
| `Zomato_Restaurant_reviews.csv` | 10,000 reviews: rating, text, reviewer info, pictures, time |

After cleaning: 9,961 reviews for 100 restaurants (5 restaurants have no reviews).

## What the project does
1. Data cleaning and wrangling
2. Exploratory analysis: 15 charts following the UBM rule (univariate, bivariate, multivariate)
3. Three hypothesis tests
4. Text pre-processing (10 steps) and TF-IDF vectorisation
5. Sentiment classification: Logistic Regression, Naive Bayes, Random Forest (cross-validation + hyperparameter tuning)
6. Restaurant clustering: K-Means vs Hierarchical (silhouette, Davies-Bouldin, elbow, dendrogram, PCA)
7. Model saving and reloading with joblib

## Key results
* **Sentiment model (Logistic Regression, tuned):** accuracy 0.956, F1 0.970, ROC-AUC 0.987; catches 92.6% of negative reviews.
* **Clustering (K-Means, K = 3):** silhouette 0.238, Davies-Bouldin 1.43.

| Segment | Restaurants | Profile | Strategy |
|---|---|---|---|
| Budget Eateries | 39 | Cheap, lowest rated | Quality support |
| Premium Favourites | 15 | Expensive, best rated | Showcase and protect |
| Buzzing Mid-Range | 46 | Many photos and critics, rarely featured | Curate and promote |

## Limitations
Single city, at most 100 reviews per restaurant, moderate cluster separation (silhouette about 0.24), and a binary
sentiment model with no neutral class.

## How to run
1. Open `Zomato_Capstone_Project.ipynb` in Google Colab (or Jupyter).
2. Put both CSV files in the same folder (in Colab: the `/content` file panel).
3. Run all cells. The notebook installs missing packages itself.

Local setup: `pip install -r requirements.txt`

## Files
* `Zomato_Capstone_Project.ipynb` - full analysis with outputs
* `Zomato_Analysis_15_Charts.pdf` - the 15 charts with insights
* `requirements.txt` - Python packages
