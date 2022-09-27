#UNDER CONSTRUCTION...



# DIAGNOSIS OF BREAST CANCER
## Introduction
Breast cancer is the most common type of cancer at a staggering 12% of new cases a year, according to the World Health Organization. It is estimated that in 2022 there will be 287,850 new cases of cancer. From the moment a biopsy is done to the moment the results are given to a patient requires a lot of resources. A big team of concerned doctors, educated people in the lab, certified pathologists, and speedy transcribers are needed to produce a diagnosis. 

## Dataset
The Breast Cancer Wisconsin (Diagnostic) Data Set was taken from the UCI Machine Learning Repository. The features describe characteristics of the cell nuclei present computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The data consists of the characteristics of 569 breast mass images with thirty-three variables.

Source:  https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

## Purpose of the analysis
When I worked for a pathology lab, let us call it LadCorp, I observed how complex the process to generate biopsy results is. The purpose of this analysis is to use machine learning to classify benign or malignant breast cancer diagnosis based on a portion of the characteristics of a FNA of a breast mass. To use data science to make a complicated and lengthy process an easier one. The goal of this project is to build a model that can accurately predict the diagnosis of breast cancer tissues as either malignant or benign.

## Research Question
How do Decision Tree, Random Forest, Logistic Regression, Support Vector Machines (SVM), Naïve Bayes (NB), Stochastic Gradient Descent (SGD), and K Nearest Neighbors (KNN) compare with each other in classifying whether a mass is benign or malignant based on the FNA’s characteristics? 

## Variables
Output variable:
-	Diagnosis (M = malignant, B = benign)
357 observations in the benign class and 212 observations in the smaller malignant class. The distribution of the target variable is not the best (50-50), but it is not terrible either at a 37% malignant and 63% benign distribution. ROC, F1 score, precision and recall scores were used to evaluate the algorithms to compensate for this slight imbalance.

![alt text](https://github.com/natvalenz/breastCancer/blob/main/Picture20.png)
![alt text](https://github.com/natvalenz/breastCancer/blob/main/Picture21.png)

Input variables: 
-	ID number
-	3-32 Ten real-valued features are computed for each cell nucleus:
  -	a. radius (mean of distances from center to points on the perimeter)
  -	b. texture (standard deviation of gray-scale values)
  -	c. perimeter
  -	d. area
  -	e) smoothness (local variation in radius lengths)
  -	f) compactness (perimeter^2 / area - 1.0)
  -	g) concavity (severity of concave portions of the contour)
  -	h) concave points (number of concave portions of the contour)
  -	i) symmetry
  -	j) fractal dimension ("coastline approximation" - 1)

For each breast mass FNA image the mean, standard error, and “worst” or largest (mean of the three largest values) measure was calculated for the ten features from a) to j), resulting in thirty features. For example, columns named radius_mean, radius_se, and radius_worst.

## Data Preprocessing, Data Partitioning, and Feature Selection
The dataset was analyzed, and the data was not considered to be very dirty. Id, and Unnamed null value columns were dropped. Multicollinearity is present in this dataset. The mean, standard error and worst measures are correlated (radius_mean, radius_se, and radius_worst are correlated), and some of the variables such as radius, perimeter, and area are highly correlated. The other problem variables are compactness, concavity, concave points, and fractal_dimension. 
Data scientists are like bartenders they need to get the ingredients ready for different drinks. The dataset was scaled appropriately based on the model and the type of data (Diagnosis: M=1 or B=0 is changed using LabelEncoder() and continuous using RobustScaler()). Multicollinearity was taken into consideration for the models that are sensitive to it. The pair plots were a great visual summary showing how a number of the variables were going to be great for the classification models and others would not be. The boxplots showed skewness caused by the outliers. Outliers were not dropped because there is not enough information about them and every observation is important since there is a small number of observations, these outliers might be a representative case.

### Heatmap
![alt text](https://github.com/natvalenz/breastCancer/blob/main/Picture22.png)

### Pairplot
![alt text](https://github.com/natvalenz/breastCancer/blob/main/Picture23.png)






 

