
# coding: utf-8

# In[2]:


import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('BDAS_Project').getOrCreate()


# In[3]:


import pandas as pd


# In[4]:


#FA = pd.read_csv('Datasets/Fatalities.csv')
#ME = pd.read_csv('Datasets/Metrics.csv')


# In[5]:


#import data sets in Spark
FA = spark.read.csv('Datasets/Fatalities.csv',header=True,inferSchema=True)
ME = spark.read.csv('Datasets/Metrics.csv',header=True,inferSchema=True)


# In[6]:


# check the datatypes of FA
FA.dtypes


# In[7]:


FA.show()

# Let's see how many rows of data we originally have.
print("Total data points:", FA.count())


# In[8]:


# Import datasets in Pandas

FAP = pd.read_csv("Datasets/Fatalities.csv")
MEP = pd.read_csv("Datasets/Metrics.csv")

# Find null values to verify data quality using Pandas. 
# Count the Null Columns
# There are 583 null values in Sex column of Fatalities dataframe

null_columns=FAP.columns[FAP.isnull().any()]
FAP[null_columns].isnull().sum()


# In[9]:


#return every row that contains at least one null value in Fatalities
print(FAP[FAP.isnull().any(axis=1)][null_columns].head())


# In[10]:


# check data types of Metrics
ME.dtypes


# In[11]:


# Find null values to verify data quality using Pandas
# there are 5 null values in 3 column of Metrics dataset

#Count the Null Columns
null_columns=MEP.columns[MEP.isnull().any()]
MEP[null_columns].isnull().sum()


# In[12]:


#return every row that contains at least one null value in Metrics
print(MEP[MEP.isnull().any(axis=1)][null_columns].head())


# In[13]:


# Visualization through matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')

#FAP = pd.read_csv("Datasets/Fatalities.csv")
#MEP = pd.read_csv("Datasets/Metrics.csv")

plt.style.use('fivethirtyeight')

MEP.plot(x='Year', y="Tobacco Price Index")


# In[14]:


MEP.plot(x='Year', y="Household Expenditure on Tobacco")


# In[15]:


FAP.dtypes


# In[16]:


#Convert Value column to numeric from string
FAP.Value = pd.to_numeric(FAP.Value, errors = 'coerce').fillna(0).astype(np.int64)


# In[17]:


# Check the number of male deaths by each year
FAP2 = FAP[FAP['Metric'].str.contains("Attributable number") & FAP['Diagnosis Type'].str.match("All deaths") 
        & FAP['Sex'].str.match("Male")]
FAP2.plot(x='Year', y="Value")


# In[18]:


# Check the number of female deaths by each year
FAP2 = FAP[FAP['Metric'].str.contains("Attributable number") & FAP['Diagnosis Type'].str.match("All deaths") 
        & FAP['Sex'].str.match("Female")]
FAP2.plot(x='Year', y="Value")


# In[19]:


#Diagnosis Type
Diagnosis_Type = pd.DataFrame(FAP, columns = ['Diagnosis Type']) 
count_Diagnosis_Type = Diagnosis_Type.stack().value_counts()
ax = count_Diagnosis_Type.plot(kind = 'pie',
                              title = 'Diagnosis Type',
                              startangle = 10,
                              autopct='%.2f',
                              explode=(0,0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4))
ax.set_ylabel('')


# In[20]:


# ICD10 Diagnosis
#ICD10_Diagnosis = pd.DataFrame(FAP, columns = ['ICD10 Diagnosis']) 
#count_ICD10_Diagnosis = ICD10_Diagnosis.stack().value_counts()
#ax = count_ICD10_Diagnosis.plot(kind = 'bar',
                             # title = 'ICD10 Diagnosis',
                             # fontsize=10,  width=0.5,  figsize=(12, 10))
#ax.set_ylabel('')


# In[21]:


# Print schema allows us to visualise the data structure at a high level. 
FA.printSchema()

# We can also use head to print a specific amount of rows, so we can get a better understanding of the data points.
print(FA.head(1))


# In[22]:


# We can use the describe method to get some general statistics on our data
FA.describe().show()


# In[23]:


# Let's select the columns that are integers only for now
FA.select('Year').describe().show()


# In[24]:


# Visualize using Pandas
pd.DataFrame(FA.take(10), columns=FA.columns)


# In[25]:


#import matplotlib.pyplot as plt
#plt.hist(FA)
#plt.xlabel("Year")
#plt.ylabel("Value")
#plt.show()


# In[26]:


ME.show()

# Let's see how many rows of data we originally have.
print("Total data points:", ME.count())


# In[27]:


# check the datatypes of ME
ME.dtypes


# In[28]:


# Print schema allows us to visualise the data structure at a high level. 
ME.printSchema()

# We can also use head to print a specific amount of rows, so we can get a better understanding of the data points. 
print(ME.head(1))


# In[29]:


# We can use the describe method get some general statistics on our data too. 
ME.describe().show()


# In[30]:


# Let's select the columns that are integers, and use the describe method again.
# We see that the mean of Tobacco Price Index is 520.83, Retail Prices Index is 239.50, Tobacco Price Index Relative to Retail Price Index is 195.63,   
# Also the average Real Households Disposable Income is 154.66, Affordability of Tobacco Index is 81.97, Household Expenditure on Tobacco 13417.45, Household Expenditure Total is 652008.06 and average Expenditure on Tobacco as a Percentage of Expenditure is 2.387

ME.select('Tobacco Price Index', 'Retail Prices Index', 'Tobacco Price Index Relative to Retail Price Index','Real Households Disposable Income','Affordability of Tobacco Index','Household Expenditure on Tobacco','Household Expenditure Total','Expenditure on Tobacco as a Percentage of Expenditure').describe().show()


# In[31]:


# Using a for loop to find all columns that belong to the integer data type. 
numeric_features = [t[0] for t in ME.dtypes if t[1] == 'int']

# Selecting the numeric features, generating summary statistics, and converting to a Pandas DataFrame.
ME.select(numeric_features).describe().toPandas().transpose()


# In[32]:


# Visualize using Pandas
pd.DataFrame(ME.take(10), columns=ME.columns)


# In[33]:


# Fatality records of patients (total of 1749 records) are from 2004 to 2014, while the Metrics of smoking (total of 36 records) are from 1980 to 2015.
# Therefore, the records from only 2004 to 2014 will be selected from the ‘Metrics’ data set and assign it to a new variable called Year_fil
ME1 = ME.filter("Year >= 2004 AND Year <= 2014")
ME1.show()


# In[34]:


ME1.select('Year').show()


# In[35]:


# Let's see how many rows of data we have now. 
print("Total data points:", ME1.count())


# In[36]:


# 'na' stands for Not Available. Using na, we can then use drop. 
dropped_ME1 = ME1.na.drop()

# Let's see how many rows of data we have now after dropping rows with null values
print("Total data points:", dropped_ME1.count())


# In[37]:


# Rename column names of Fatalities (FA) to remove the spaces of the column names, as it will make the coding easier as we move on
FA1 = FA.withColumnRenamed("ICD10 Code", "ICD10_Code")
FA2 = FA1.withColumnRenamed("ICD10 Diagnosis", "ICD10_Diagnosis")
FA3 = FA2.withColumnRenamed("Diagnosis Type", "Diagnosis_Type")
       
FA3.show()


# In[38]:


# Filter Fatalities records by Diagnosis_Type = All Deaths and Metric = Attributable Number

FA4 = FA3[FA3.Metric=='Attributable number'][FA3.Diagnosis_Type=='All deaths']
FA4.show()


# In[39]:


# Count the number of null values on Sex column
from pyspark.sql.functions import isnan
FA4.filter((FA4["Sex"] == "") | FA4["Sex"].isNull() | isnan(FA4["Sex"])).count()


# In[40]:


# Select Male 
FA5 = FA4[FA4.Sex=='Male']
FA5.show()


# In[41]:


# Rename 'Value' column of FA5 to 'Number_of_Male_Deaths'
FA6 = FA5.withColumnRenamed("Value", "Number_of_Male_Deaths")

FA6.show()


# In[42]:


# Select Female 
FA7 = FA4[FA4.Sex=='Female']

FA7.show()


# In[43]:


# Rename 'Value' column of FA7 to 'Number_of_Female_Deaths'
FA8 = FA7.withColumnRenamed("Value", "Number_of_Female_Deaths")

FA8.show()


# In[44]:


# Drop ICD10_Code, ICD10_Diagnosis,Diagnosis_Type, Metric and Sex columns from FA8
FA9 = FA8.drop("ICD10_Code", "ICD10_Diagnosis","Diagnosis_Type","Metric","Sex")

FA9.show()


# In[45]:


# Rename Year column to Year2 of FA9 before merging
FA10 = FA9.withColumnRenamed("Year", "Year2")
FA10.show()


# In[46]:


# Inner Join FA6 and FA10 variables
FA11 = FA6.join(FA10, FA6.Year == FA10.Year2)

FA11.show()


# In[47]:


# Drop Year2 column from FA11
FA12 = FA11.drop("Year2")
FA12.show()


# In[48]:


# Drop Sex columns from FA12
FA13 = FA12.drop("Sex")
FA13.show()


# In[49]:


#return every row that contains at least one null value in Fatalities
#print(FA13[FAP13.isnull().any(axis=1)][null_columns].head())


# In[50]:


# Check data types on FA13
FA13.dtypes


# In[51]:


# Take the data and visualise in Pandas
pd.DataFrame(FA13.take(11), columns=FA13.columns)


# In[52]:


# Change datatype to Numerical from String on Number_of_Male_Deaths and Number_of_Female_Deaths columns of FA13

import pyspark
from pyspark.sql.types import IntegerType
FA13 = FA13.withColumn("Number_of_Male_Deaths", FA13["Number_of_Male_Deaths"].cast(IntegerType()))
FA13 = FA13.withColumn("Number_of_Female_Deaths", FA13["Number_of_Female_Deaths"].cast(IntegerType()))

# Check data types on FA13, to confirm that the above columns are changed to integers 
FA13.dtypes


# In[53]:


#Create a new column called 'Total_Number_of_Deaths' by accumlating the Number_of_Male_Deaths and Number_of_Female_Deaths 

from pyspark import SparkContext
import pyspark.sql 

FA14 = FA13.withColumn('Total_Number_of_Deaths', (FA13['Number_of_Male_Deaths'] + FA13['Number_of_Female_Deaths']))

# Take the data and visualise in Pandas
pd.DataFrame(FA14.take(11), columns=FA14.columns)


# In[54]:


# Check data types on FA14 to make sure the new column is an Integer
FA14.dtypes


# In[55]:


# Now the Fatalities dataframe (F14) is ready to be merged with Metrics dataframe. First lets visualize it again using Pandas
pd.DataFrame(ME1.take(11), columns=ME1.columns)


# In[56]:


# Check data types on ME1 to make sure all the fields are Integers
ME1.dtypes


# In[57]:


# Rename Year column of ME1 to 'Year1' before the merging takes place and also remove the spaces of other columns in ME1

MER1 = ME1.withColumnRenamed("Year", "Year1")
MER2 = MER1.withColumnRenamed("Tobacco Price Index", "Tobacco_Price_Index")
MER3 = MER2.withColumnRenamed("Retail Prices Index", "Retail_Prices_Index")
MER4 = MER3.withColumnRenamed("Tobacco Price Index Relative to Retail Price Index", "Tobacco_Price_Index_Relative_to_Retail_Price_Index")
MER5 = MER4.withColumnRenamed("Real Households Disposable Income", "Real_Households_Disposable_Income")
MER6 = MER5.withColumnRenamed("Affordability of Tobacco Index", "Affordability_of_Tobacco_Index")
MER7 = MER6.withColumnRenamed("Household Expenditure on Tobacco", "Household_Expenditure_on_Tobacco")
MER8 = MER7.withColumnRenamed("Household Expenditure Total", "Household_Expenditure_Total")
ME2 = MER8.withColumnRenamed("Expenditure on Tobacco as a Percentage of Expenditure", "Expenditure_on_Tobacco_as_a_Percentage_of_Expenditure")

#Visualize through Pandas
pd.DataFrame(ME2.take(10), columns=ME2.columns)


# In[58]:


# Now the lets merge FA14 with ME2 (Inner Join)

df = FA14.join(ME2, FA14.Year == ME2.Year1)


# In[59]:


# Visualize the output using Pandas
pd.DataFrame(df.take(11), columns=df.columns)


# In[60]:


# Drop Year1 column from df
df1 = df.drop("Year1")

# Visualize the output using Pandas
pd.DataFrame(df1.take(11), columns=df1.columns)


# In[61]:


# Let's visually inspect the data. To check whether there are any null values
df1.show()

# Let's see how many rows of data we originally have.
print("Total data points:", df1.count())


# In[62]:


# 'na' stands for Not Available. Using na, we can then use drop. 
# After using show, we will be able to find that the rows with the null values are gone.
df1.na.drop().show()

# Let's see how many rows of data we have now. If we have a count of 11, then there were no rows with null values
print("Total data points:", df1.count())


# In[63]:


# Check data types of df1
df1.dtypes


# In[64]:


# Since there is a newly formed column with Total_Number_of_Deaths 
# (‘Total_Number_of_Deaths’ was derived by adding ‘Number_of_Male_Deaths’ to ‘Number_of_Female_Deaths’), 
# number of male deaths and number of female deaths can be filtered out from df1

df2 = df1.drop("Number_of_Male_Deaths", "Number_of_Female_Deaths")

# Visualize the output using Pandas
pd.DataFrame(df2.take(11), columns=df2.columns)


# In[65]:


# Print schema allows us to visualise the data structure at a high level. 
df2.printSchema()

# We can also use head to print a specific amount of rows, so we can get a better understanding of the data points.
print(df2.head(1))


# In[66]:


# Let’s find correlation between independent variables and target variable (Total_Number_of_Deaths)
# The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation.
# When the coefficient is close to –1, it means that there is a strong negative correlation.
# Coefficients close to zero mean that there is no linear correlation.

import six
for i in df2.columns:
    if not( isinstance(df2.select(i).take(1)[0][0], six.string_types)):
        print("Correlation to Total_Number_of_Deaths for", i, df2.stat.corr('Total_Number_of_Deaths',i))


# In[67]:


# Drop ICD10_Code, ICD10_Diagnosis, Diagnosis_Type, Metric and Expenditure_on_Tobacco_as_a_Percentage_of_Expenditure

df3 = df2.drop("ICD10_Code", "ICD10_Diagnosis", "Diagnosis_Type", "Metric", "Expenditure_on_Tobacco_as_a_Percentage_of_Expenditure")

# Visualize the output using Pandas
pd.DataFrame(df3.take(11), columns=df3.columns)


# In[68]:


# Print schema allows us to visualise the data structure at a high level. 
df3.printSchema()

# We can also use head to print a specific amount of rows, so we can get a better understanding of the data points.
print(df3.head(1))


# In[69]:


# Convert the Pyspark contructed dataframe to Pandas dataframe to visualize
DFP = df3.toPandas()

# When the Retail Prices Index decrease, the number of deaths caused by smoking tend to increase.
plt.style.use('fivethirtyeight')
DFP.plot(x="Total_Number_of_Deaths" ,y='Retail_Prices_Index')


# In[70]:


# When the Tobacco Price Index Relative to Retail Price Index decrease, 
# the number of deaths caused by smoking tend to increase.
plt.style.use('fivethirtyeight')
DFP.plot(x="Total_Number_of_Deaths" ,y='Tobacco_Price_Index_Relative_to_Retail_Price_Index')


# In[71]:


# When the Tobacco Price Index decrease, 
# the number of deaths caused by smoking tend to increase.
plt.style.use('fivethirtyeight')
DFP.plot(x="Total_Number_of_Deaths" ,y='Tobacco_Price_Index')


# In[72]:


# When the Tobacco Price Index decrease, 
# the number of deaths caused by smoking tend to increase.
plt.style.use('fivethirtyeight')
DFP.plot(x="Total_Number_of_Deaths" ,y='Affordability_of_Tobacco_Index')


# In[73]:


# Linear Regression


# In[74]:


# Import LinearRegression from pyspark.ml.regression
from pyspark.ml.regression import LinearRegression


# In[75]:


# Let's explore. Here's the first row of the data.
print(df3.head())

# And the entire data structure. 
df3.printSchema()


# In[76]:


# Let's use a Python package to neatly describe the data.
df3.describe().toPandas().transpose()


# In[77]:


# Import VectorAssembler and Vectors
from pyspark.ml.feature import VectorAssembler

# The input columns are the feature column names, and the output column is what you'd like the new column to be named. 
vector_assembler = VectorAssembler(inputCols = ['Year', 'Tobacco_Price_Index', 'Retail_Prices_Index', 'Tobacco_Price_Index_Relative_to_Retail_Price_Index', 'Real_Households_Disposable_Income', 'Affordability_of_Tobacco_Index', 'Household_Expenditure_on_Tobacco', 'Household_Expenditure_Total'], outputCol = 'features')

# Now that we've created the assembler variable, let's actually transform the data.
vector_output = vector_assembler.transform(df3)

# Using print schema, you see that the features output column has been added. 
vector_output.printSchema()

# You can see that the features column is a DenseVector that combines the various features as expected.
vector_output.head(1)


# In[78]:


# Because the features have been combined into one vector, we no longer need them. Below we select the features and label.
vector_output = vector_output.select(['features', 'Total_Number_of_Deaths'])

# You can see that the dataframe now only contains two columns. 
print(vector_output.head(1))
vector_output.show(3)


# In[128]:


# Importing the LR package.
from pyspark.ml.regression import LinearRegression

# Instantiate the instance.
lr = LinearRegression(featuresCol='features', labelCol='Total_Number_of_Deaths', maxIter=100, regParam=0.0, elasticNetParam=0.0,
                      tol=1e-6, fitIntercept=True, standardization=True, solver="auto", aggregationDepth=2)
# Fit the vector_output
lr_model = lr.fit(vector_output)
# Print the coefficients.
print("Coefficients: " + str(lr_model.coefficients))
# Print the intercept.
print("Intercept: " + str(lr_model.intercept) + "\n")
# Summarise the model and print out some evaluation metrics.
vdf3_Summary = lr_model.summary
# Print MAE (Mean Absolute Error)
print("MAE: " + str(vdf3_Summary.meanAbsoluteError))
# Print RMSE (Root Mean Squared Error) 
print("RMSE: " + str(vdf3_Summary.rootMeanSquaredError))
# Print R2 (R Squarred)
print("R2: " + str(vdf3_Summary.r2))


# In[80]:


vector_output.describe().show()

# RMSE measures the differences between predicted values and actual values. 
# However, RMSE alone is meaningless until we compare with the actual "Total_Number_of_Deaths" value, such as mean, min and max. 
# After such comparison, our RMSE is not looking good. RMSE should be a low value.

# R squared (R2) at 0.97 indicates that our model can explain approximately 97% of the variability in total number of deaths value


# In[117]:


# Linear Regression model performance on the real data frame

lr_predictions = lr_model.transform(vector_output)
lr_predictions.select("prediction","Total_Number_of_Deaths","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator

lrR2_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="Total_Number_of_Deaths",metricName="r2")
print("R Squared (R2) = %g" % lrR2_evaluator.evaluate(lr_predictions))

lrMAE_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="Total_Number_of_Deaths",metricName="mae")
print("Mean Absolute Error (MAE) = %g" % lrMAE_evaluator.evaluate(lr_predictions))

lrRMSE_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="Total_Number_of_Deaths",metricName="rmse")
print("Root Mean Squared Error (RMSE) = %g" % lrRMSE_evaluator.evaluate(lr_predictions))


# In[82]:


# Convert the Pyspark contructed dataframe to Pandas dataframe to visualize
lr_predictionsPandas = lr_predictions.toPandas()

# When the total_number_of_deaths increase, the prediction tends to increase
plt.style.use('Solarize_Light2')
lr_predictionsPandas.plot(x="Total_Number_of_Deaths" ,y='prediction')


# In[83]:


# Following is the statistics comparison of the Total Number of Deaths and Prediction
lr_predictions.describe().show()


# In[84]:


lr_predictions.show()


# In[85]:


# Lets merge the lr_predictions with df3 dataframe to extract more meaningful information by visualizing
df4 = df3.join(lr_predictions, lr_predictions.Total_Number_of_Deaths == df3.Total_Number_of_Deaths)

# Order by Year
df5 = df4.orderBy('Year')

# Visualize the output using Pandas
pd.DataFrame(df5.take(11), columns=df5.columns)


# In[86]:


# Convert the Pyspark contructed dataframe to Pandas dataframe to visualize
df5Pandas = df5.toPandas()

# when the Tobacco Price Index is high, the predicted scores are low. 
# But as the Tobacco Price Index decrease, the predicted scores Increase.
plt.style.use('seaborn-pastel')
df5Pandas.plot(x="prediction" ,y='Tobacco_Price_Index')


# In[87]:


# when the ‘Tobacco Price Index Relative to Retail Price Index’ is high, the predicted values are low. 
# But as the ‘Tobacco Price Index Relative to Retail Price Index’ decrease, the predicted values Increase.
plt.style.use('seaborn-pastel')
df5Pandas.plot(x="prediction" ,y='Tobacco_Price_Index_Relative_to_Retail_Price_Index')


# In[88]:


# Real_Households_Disposable_Income vs prediction plot is very disoriented
plt.style.use('classic')
df5Pandas.plot(x="prediction" ,y='Real_Households_Disposable_Income')


# In[89]:


lr_predictions.show()


# In[124]:


import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance


xx = df5Pandas[['Year','Tobacco_Price_Index','Retail_Prices_Index','Tobacco_Price_Index_Relative_to_Retail_Price_Index',
                'Real_Households_Disposable_Income','Affordability_of_Tobacco_Index','Household_Expenditure_on_Tobacco',
                'Household_Expenditure_Total']].values
yy = df5Pandas[['Total_Number_of_Deaths']].values
reg = linear_model.LinearRegression()
model=reg.fit(xx,yy)
print('Coefficients: \n', reg.coef_)
perm = PermutationImportance(reg, random_state=1).fit(xx, yy)
eli5.show_weights(perm)


# In[91]:


################################################################################################


# In[92]:


# Linear Regression by Splitting Data 


# In[93]:


# Now lets do Linear Regression again by splitting the dataset by 60:40 
splits = vector_output.randomSplit([0.6, 0.4])
train_df = splits[0]
test_df = splits[1]

# Let's see our training data.
train_df.describe().show()

# And our testing data.
test_df.describe().show()


# In[130]:


# Linear Regression on train_df
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol='features', labelCol='Total_Number_of_Deaths', maxIter=100, regParam=0.0, elasticNetParam=0.0,
                      tol=1e-6, fitIntercept=True, standardization=True, solver="auto", aggregationDepth=2)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))


# In[131]:


lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","Total_Number_of_Deaths","features").show(5)

from pyspark.ml.evaluation import RegressionEvaluator

lrR2_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="Total_Number_of_Deaths",metricName="r2")
print("R Squared (R2) = %g" % lrR2_evaluator.evaluate(lr_predictions))

lrMAE_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="Total_Number_of_Deaths",metricName="mae")
print("Mean Absolute Error (MAE) = %g" % lrMAE_evaluator.evaluate(lr_predictions))

lrRMSE_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="Total_Number_of_Deaths",metricName="rmse")
print("Root Mean Squared Error (RMSE) = %g" % lrRMSE_evaluator.evaluate(lr_predictions))


# In[96]:


# Using our Linear Regression model to make some predictions:

predictions = lr_model.transform(test_df)
predictions.select("prediction","Total_Number_of_Deaths","features").show()


# In[97]:


################################################################################################


# In[138]:


# Decision Tree Regression

from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'Total_Number_of_Deaths', maxDepth=1, maxBins=32, 
                           minInstancesPerNode=3, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, 
                           checkpointInterval=10, impurity="variance", seed=None)
dt_model = dt.fit(vector_output)
dt_predictions = dt_model.transform(vector_output)

# RMSE
dtr_evaluator = RegressionEvaluator(
    labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="rmse")
rmse = dtr_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on vector_output = %g" % rmse)

# MAE
dtm_evaluator = RegressionEvaluator(
    labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="mae")
mae = dtm_evaluator.evaluate(dt_predictions)
print("Mean Absolute Error (MAE) on vector_output = %g" % mae)

# R2
dtr2_evaluator = RegressionEvaluator(
    labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="r2")
r2 = dtr2_evaluator.evaluate(dt_predictions)
print("R-Squared (R2) on vector_output = %g" % r2)


# In[139]:


# Feature Importance
dt_model.featureImportances


# In[140]:


df3.take(1)


# In[101]:


# Decision tree regression using train and test models

from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'Total_Number_of_Deaths', maxDepth=1, maxBins=32, 
                           minInstancesPerNode=3, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, 
                           checkpointInterval=10, impurity="variance", seed=None)
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)
dt_evaluator = RegressionEvaluator(labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[102]:


#Feature Importance
dt_model.featureImportances


# In[103]:


# Because 4: 0.0277 (as per above results), 4th column, Retail_Prices_Index is the most important feature to predict the 
# total number of deaths in our data
df3.take(1)


# In[104]:


################################################################################################


# In[144]:


# Gradient-boosted Tree Regression

from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'Total_Number_of_Deaths', maxDepth=4, maxBins=32, 
                   minInstancesPerNode=1, minInfoGain=0.0,maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10,seed=None)
gbt_model = gbt.fit(vector_output)
gbt_predictions = gbt_model.transform(vector_output)
gbt_predictions.select('prediction', 'Total_Number_of_Deaths', 'features').show(11)


# In[145]:


# Gradient-boosted tree model evaluation

# Root Mean Squared Error (RMSE)
gbt_evaluator_RMSE = RegressionEvaluator(
    labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="rmse")
grmse = gbt_evaluator_RMSE.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on vector_output = %g" % grmse)

# Mean Absolute Error (MAE)
gbt_evaluator_MAE = RegressionEvaluator(
    labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="mae")
gmae = gbt_evaluator_MAE.evaluate(gbt_predictions)
print("Mean Absolute Error (MAE) on vector_output = %g" % gmae)

# R Squarred (R2)
gbt_evaluator_R2 = RegressionEvaluator(
    labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="r2")
gr2 = gbt_evaluator_R2.evaluate(gbt_predictions)
print("R Squarred (R2) on vector_output = %g" % gr2)


# In[147]:


# Feature Importance of Gradient-boosted tree regression
gbt_model.featureImportances


# In[148]:


# Convert the Pyspark contructed dataframe to Pandas dataframe to visualize
gbt_predictionsPandas = gbt_predictions.toPandas()

# When the total_number_of_deaths increase, the prediction tends to increase
plt.style.use('Solarize_Light2')
gbt_predictionsPandas.plot(x="Total_Number_of_Deaths" ,y='prediction')


# In[149]:


# Following is the statistics comparison of the Total Number of Deaths and Prediction
gbt_predictions.describe().show()


# In[150]:


# Lets merge the gbt_predictions with df3 dataframe to extract more meaningful information by visualizing
df6 = df3.join(gbt_predictions, gbt_predictions.Total_Number_of_Deaths == df3.Total_Number_of_Deaths)

# Order by Year
df7 = df6.orderBy('Year')

# Visualize the output using Pandas
pd.DataFrame(df7.take(11), columns=df7.columns)


# In[151]:


# Convert the Pyspark contructed dataframe to Pandas dataframe to visualize
df7Pandas = df7.toPandas()

# when the Tobacco Price Index is high, the predicted scores are low. 
# But as the Tobacco Price Index decrease, the predicted scores Increase.
plt.style.use('seaborn-pastel')
df7Pandas.plot(x="prediction" ,y='Tobacco_Price_Index')


# In[152]:


# when the ‘Tobacco Price Index Relative to Retail Price Index’ is high, the predicted values are low. 
# But as the ‘Tobacco Price Index Relative to Retail Price Index’ decrease, the predicted values Increase.
plt.style.use('seaborn-pastel')
df7Pandas.plot(x="prediction" ,y='Tobacco_Price_Index_Relative_to_Retail_Price_Index')


# In[153]:


# Real_Households_Disposable_Income vs prediction plot is very disoriented
plt.style.use('classic')
df7Pandas.plot(x="prediction" ,y='Real_Households_Disposable_Income')


# In[154]:


################################################################################################


# In[155]:


# Gradient-boosted Tree Regression on train and test data (split data)

from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'Total_Number_of_Deaths', maxDepth=5, maxBins=32, 
                   minInstancesPerNode=1, minInfoGain=0.0,maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10,seed=None)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)
gbt_predictions.select('prediction', 'Total_Number_of_Deaths', 'features').show(11)


# In[156]:


# Root Mean Squared Error (RMSE)
gbt_evaluator_RMSE = RegressionEvaluator(
    labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="rmse")
grmse = gbt_evaluator_RMSE.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) = %g" % grmse)

# Mean Absolute Error (MAE)
gbt_evaluator_MAE = RegressionEvaluator(
    labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="mae")
gmae = gbt_evaluator_MAE.evaluate(gbt_predictions)
print("Mean Absolute Error (MAE) = %g" % gmae)

# R Squarred (R2)
gbt_evaluator_R2 = RegressionEvaluator(
    labelCol="Total_Number_of_Deaths", predictionCol="prediction", metricName="r2")
gr2 = gbt_evaluator_R2.evaluate(gbt_predictions)
print("R Squarred (R2) = %g" % gr2)

