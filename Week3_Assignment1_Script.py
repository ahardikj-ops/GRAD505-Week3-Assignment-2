#!/usr/bin/env python
# coding: utf-8

# # Week 3 Assignment 1 Report
# 
# GRAD 505 Foundations in Data Science  
# Hardik Amin
# 
# Tool Used: Python in Jupyter Notebook  
# 
# This notebook includes code, plots, outputs, and written interpretations for the iris and PlantGrowth datasets.

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


# ### Iris Dataset

# In[45]:


# Load Iris dataset
iris_raw = datasets.load_iris()

# Convert to DataFrame
iris = pd.DataFrame(
    iris_raw.data,
    columns=iris_raw.feature_names
)

# Rename columns to match assignment naming
iris.columns = [
    "Sepal.Length",
    "Sepal.Width",
    "Petal.Length",
    "Petal.Width"
]

# Display first few rows
print(iris.head(15))


# 1a. Make a histogram of the variable Sepal.Width.
# 
# 1b. Based on the histogram from #1a, which would you expect to be higher, the mean or the median? Why?
# 
# 1c. Confirm your answer to #1b by actually finding these values.
# 
# 1d. Only 27% of the flowers have a Sepal.Width higher than ________ cm.
# 
# 1e. Make scatterplots of each pair of the numerical variables in iris (There should be 6 pairs/plots).
# 
# 1f. Based on #1e, which two variables appear to have the strongest relationship? And which two appear to have the weakest relationship?

# #### **1a.** Histogram of the Variable Sepal.Width

# In[5]:


plt.figure(figsize=(8,5))

plt.hist(
    iris["Sepal.Width"],
    edgecolor="black"
)

plt.title("Histogram of Sepal.Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")

plt.show()


# #### **1b.** Based on histogram, which is expected to be higher, mean or median? Why?

# In first glance, the distribution seems approximately symmetric with highest bars around 3.0–3.3 cm & the frequencies gradually decreasing on both sides. But the left tail values extend to about 2.0 cm and right tail values up to ~4.4 cm. Although the distribution appears roughly symmetric overall, the right tail extends farther than the left tail, suggesting a slight right skew.
# 
# Here, because right-skewed distributions will tend to pull the mean toward the larger values, the mean should be slightly larger than the median (Mean > Median).

# #### **1c.** Find Mean and Median, to validate

# In[7]:


# Calculate the mean and median of Sepal.Width and compare them to the prediction from Question 1b.

mean_sw = iris["Sepal.Width"].mean()
median_sw = iris["Sepal.Width"].median()

print("Mean:", mean_sw)
print("Median:", median_sw)


# The calculated mean of Sepal.Width is 3.0573 and the median is 3.0. Since the mean is slightly greater than the median, this confirms the slight right skew anticipated from the histogram.

# #### **1d.** Only 27% of the flowers have a Sepal.Width higher than <u>3.3</u> cm.

# In[9]:


# Calculate the 73rd percentile of Sepal.Width values

cutoff = iris["Sepal.Width"].quantile(0.73)
print("73rd Percentile:", cutoff)


# We need to find the Sepal.Width value above which only 27% of the flowers fall in. So, we calcualte the percentile, 100-27 = 73.
# Because 73% of the flowers have a Sepal.Width at or below 3.3 cm, only 27% of the flowers have a Sepal.Width greater than 3.3 cm.

# #### **1e.** Scatterplots of each pair of the numerical variables in iris (6 pairs / plots)

# In[12]:


sns.pairplot(
    iris,
    corner=True
)

plt.show()


# #### **1f.** Based on scatterplots, which two variables appear to have the strongest relationship? And which two appear to have the weakest relationship?

# Based on the scatterplot matrix, Petal.Length and Petal.Width appear to have the strongest relationship. The points form a very tight upward linear pattern, indicating a strong positive relationship.
# 
# Sepal.Length and Sepal.Width shows a relatively dispersed cloud of points with only a weak overall trend, suggesting the weakest relationship between the variables.

# ### PlantGrowth Dataset

# In[44]:


# Load PlantGrowth dataset

data = {
    "weight": [
        4.17, 5.58, 5.18, 6.11, 4.50,
        4.61, 5.17, 4.53, 5.33, 5.14,
        4.81, 4.17, 4.41, 3.59, 5.87,
        3.83, 6.03, 4.89, 4.32, 4.69,
        6.31, 5.12, 5.54, 5.50, 5.37,
        5.29, 4.92, 6.15, 5.80, 5.26
    ],
    "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10
}

PlantGrowth = pd.DataFrame(data)

print(PlantGrowth.head(15))


# 2a. Make a histogram of the variable weight with breakpoints (bin edges) at every 0.3 units, starting at 3.3.
# 
# 2b. Make boxplots of weight separated by group in a single graph.
# 
# 2c. Based on the boxplots in #2b, approximately what percentage of the "trt1" weights are below the minimum "trt2" weight?
# 
# 2d. Find the exact percentage of the "trt1" weights that are below the minimum "trt2" weight.
# 
# 2e. Only including plants with a weight above 5.5, make a barplot of the variable group. Make the barplot colorful using some color palette (in R, try running ?heat.colors and/or check out https://www.r-bloggers.com/palettes-in-r/).

# #### **2a.** Histogram of the Variable weight with breakpoints (bin edges) at every 0.3 units from 3.3

# In[18]:


bins = np.arange(
    3.3,
    PlantGrowth["weight"].max() + 0.3,
    0.3
)

plt.figure(figsize=(8,5))

plt.hist(
    PlantGrowth["weight"],
    bins=bins,
    edgecolor="black"
)

plt.title("Histogram of Plant Weight")
plt.xlabel("Weight")
plt.ylabel("Frequency")

plt.show()


# #### **2b.** Boxplots of weight separated by group in a single graph

# In[20]:


PlantGrowth.boxplot(
    column="weight",
    by="group",
    figsize=(8,5)
)

plt.title("Weight by Group")
plt.suptitle("")
plt.xlabel("Group")
plt.ylabel("Weight")

plt.show()


# #### **2c.** Based on boxplots, what percentage of the "trt1" weights are below the minimum "trt2" weight?

# Based on the boxplot, the minimum trt2 value roughly appears to be around 4.9. The median for trt1 is approximately 4.55 and the upper quartile edge around 4.9. Also, the minimum of trt2 appears to align approximately with the top of the trt1 box (third quartile). This means that approximately 75% of the trt1 weights are below the minimum trt2 weight. 

# #### **2d.** Exact percentage of the "trt1" weights that are below the minimum "trt2" weight

# In[22]:


# Minimum trt2 weight

min_trt2 = PlantGrowth.loc[
    PlantGrowth["group"] == "trt2",
    "weight"
].min()

print("Minimum trt2 weight:", min_trt2)


# In[23]:


# All trt1 weights

trt1_weights = PlantGrowth.loc[
    PlantGrowth["group"] == "trt1",
    "weight"
]

# Percentage below minimum trt2

pct_below = (
    (trt1_weights < min_trt2).sum()
    / len(trt1_weights)
) * 100

print("Percentage:", pct_below)


# The minimum weight in the trt2 group is 4.92. Based on calculation, 80% of the trt1 weights are below the minimum trt2 weight. This is not exactly the same, but very close to 75%  estimated visually from the boxplot before. 

# #### **2e.** Barplot of the variable group, only including plants with a weight above 5.5. Note: Make it colorful using a color palette

# In[47]:


# Keep only plants with weight > 5.5

heavy_plants = PlantGrowth[
    PlantGrowth["weight"] > 5.5
]

print(heavy_plants.head(10))


# In[37]:


group_counts = heavy_plants["group"].value_counts().reset_index()
group_counts.columns = ["group", "count"]

# Create barplot
sns.set_style("whitegrid")

plt.figure(figsize=(8,5))

sns.barplot(
    data=group_counts,
    x="group",
    y="count",
    hue="group",
    palette="rocket", # Color palette
    legend=False
)

plt.title("Plants with Weight > 5.5 by Group")
plt.xlabel("Group")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

