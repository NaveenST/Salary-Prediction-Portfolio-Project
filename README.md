# Salary Prediction Project

# Introduction-

This project deals with the predicting salaries of various job types, degree's, majors, industries, experience and the living cost. Any job in the market requires a set of skillset as a requirement along with educational criteria with years of experience in the field. How does all these factors influence the salary of the individual and can we predict the salaries by knowing these factors and how well did we predict (accuracy) is the overall gist of this project.

# Data-

We have collected 1,000,000 (1 Million) observations having attributes such as
1) Job Type i.e job designations(valid values - CEO, CFO, CTO, etc)
2) Degree (valid values - High School, Bachelores, Masters, Doctoral, etc)
3) Major (valid values - Physics, Chemistry, Computer Science, Biology, Math, etc)
4) Industry (valid values - Auto, Tech, Oil, Finance, etc)
5) Years of Experience (valid values - 1,2,3,4, etc)
6) Miles From the Metropolis i.e distance away from the nearest metropolis area (valid values - 1,2,3,4, etc)


# Data Preprocessing-
I have performed the following steps to preprocess the data:
1) The training data comes in a two csv files, hence ensured for uniqueness in the merging columns of both the training dataset before merging the into one Dataframe and made sure the shape of the resulting dataframe makesense before and after merging them
2) Checked for the duplicates or null values or nan values present in the dataset
3) Checked for invalia values present in the dataset (for eg: salary had values such as '0' which doesn't makesense) and took appropriate measures to remediate them by filling in average values wrt their other attributes
4) Performed other various checks on the datatypes, value_counts, info on the data to understand the dataset.


# Expolratory Data Analysis (EDA)-
1) Plotted few graphs to get the distributions of values
![image](https://user-images.githubusercontent.com/44300495/112863673-160e0600-9085-11eb-9a9d-e058bcb04a57.png)
![image](https://user-images.githubusercontent.com/44300495/112863692-1b6b5080-9085-11eb-98c0-f9b998acdeb4.png)
![image](https://user-images.githubusercontent.com/44300495/112863708-20300480-9085-11eb-85f3-3da4c2a7240f.png)
![image](https://user-images.githubusercontent.com/44300495/112863725-245c2200-9085-11eb-8c08-26dd75b4c855.png)





