# Salary Prediction Project

# Introduction-

This project deals with predicting the salary of an individual. Salary varies with factors such as job types, degree's, majors, industries, experience and the living cost. Any job in the market requires a set of skillset as a requirement along with educational background and experience in the field. How does all these factors influence the salary of an individual, what factors influence is the most, can we predict the salaries by knowing these factors and how well did we predict (accuracy) is the overall gist of this project.

# Data-

We have collected 1,000,000 (1 Million) observations having attributes such as

1) Job Type i.e job designation (valid values - CEO, CFO, CTO, etc)
2) Degree i.e level of education (valid values - High School, Bachelores, Masters, Doctoral, etc)
3) Major i.e stream of education (valid values - Physics, Chemistry, Computer Science, Biology, Math, etc)
4) Industry i.e type of organization (valid values - Auto, Tech, Oil, Finance, etc)
5) Years of Experience i.e level of expertise (valid values - 1,2,3,4, etc)
6) Miles From the Metropolis i.e distance away from the nearest metropolis area (valid values - 1,2,3,4, etc)
7) Company ID i.e sort of company name (valid values - COMP01, COMP02, etc)
8) Job ID i.e unique ID of the employee (valid values - JOB1362685407687, JOB1362685407690, etc)


# Data Preprocessing-
The training data comes in a two csv files, one containing all the feature variables and the other with the target variable. Since this data will be used for training and testing the model, we can merge these feature variables with the target variable. 
Hence the first step for me was 
1) to ensure for uniqueness in the merging columns of both the feature and target variables dataset before merging the into one Dataframe and made sure the shape of the resulting dataframe makesense before and after merging them
2) Checked for the duplicates or null values or nan values present in the dataset
3) Checked for invalid values present in the dataset (for eg: salary had values such as '0' which doesn't makesense) and took appropriate measures to remediate them by filling in average values wrt thier other attributes
4) Performed other various checks on the datatypes, value_counts, info, describe on the data to understand the dataset.


# Expolratory Data Analysis (EDA)-
1) Plotted few graphs to get the distributions of values - the below graphs shows the distribution of values in the feature variables for the training dataset

![image](https://user-images.githubusercontent.com/44300495/112864529-e3184200-9085-11eb-9c96-8890db4121bd.png)


2) Plotted few Box Plots to understand the outliers and pattern of data points for each cateogrical variables

![image](https://user-images.githubusercontent.com/44300495/112865045-66399800-9086-11eb-836e-a6c5bc01dbdb.png)


3) Correlation between numerical feature variables v/s target variable

![image](https://user-images.githubusercontent.com/44300495/112865667-fa0b6400-9086-11eb-8fb3-ac92e1895854.png)

   As the experience increases the salary also increases signifying the positve correlation between these two variables


![image](https://user-images.githubusercontent.com/44300495/112865788-1c9d7d00-9087-11eb-9804-4b74699fa80c.png)

   As the distance from the metropolis area increases the salary decreases signifying the negative correlation between these two variables


4) Performed ANOVA testing on the categorical feature variables v/s target variable to find the correlation between them

![image](https://user-images.githubusercontent.com/44300495/112866216-8b7ad600-9087-11eb-9f0b-f95e3a593d40.png)

   The above graph showcases which feature variable has what amount of impact on the target variable


5) Performed Lasso Regression for feature selection

![image](https://user-images.githubusercontent.com/44300495/112866586-f3c9b780-9087-11eb-82a0-b7cd7ff932f5.png)

  Above graph shows which feature variables plays important role in predicting the target variable
  
  
6) Correlational Coefficients

![image](https://user-images.githubusercontent.com/44300495/112893144-45823a00-90a8-11eb-9187-c862b84b4fdb.png)

  Above graph shows the correlational coefficients between the feature variables and the target variable


# Modeling-

 In order to develop a model, we first need to build a baseline model and calculate its MSE. I have built a baseline model which has MSE of 963.92 for jobType. Considering this model as baseline, my goal is to build a model with an MSE less than 360 and with the highest r_sqaured value. I have opted to choose Linear Regression, Random Forest Regression and Gradient Boosting Regression to build my salary prediction model and to choose the best among them after evaluating each model with cross_validation of 5 fold split.

 I have trained and tested each of the above model with train and test data split of 75% and 25% and considering feature variables such as jobType, major, degree, industry, yearsExperience and milesFromMetropolis to predict the salary(target variable). Feature selection has been inspired by the correlational heatmap and feature selection from Lasso Regression.
 
 ![image](https://user-images.githubusercontent.com/44300495/112894659-21275d00-90aa-11eb-968a-906fc22d89d6.png)


  From the above matrix, we can confirm that the Gradient Boosting Regression is the best model with MSE less than 360
  
  Below graph shows the model performance of actual v/s predicted values
  
  ![image](https://user-images.githubusercontent.com/44300495/112895131-baef0a00-90aa-11eb-97bd-3167d1d554de.png)
  
  
  Feature Importance- The Gradient Boosting Regression model considered the following feature variables as important in predicted the salaries
  
  ![image](https://user-images.githubusercontent.com/44300495/112895352-03a6c300-90ab-11eb-880d-7bbe15f38fde.png)
  
  
  # Conclusion-
  
   Gradient Boosting Regression model outperformed all the other models that was used in predicting the salary in both the MSE and R_Sqaured values.











