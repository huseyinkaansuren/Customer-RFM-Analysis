import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df_ = pd.read_csv("ecommerce_customer_data_large.csv")
df = df_.copy()

###Exploratory Data Analysis
df.head()

df.info()


#Object to Datetime conversion
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"])

df["Purchase Date"] = df["Purchase Date"].dt.date

df.head()
df.info()

df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], format = "%Y-%m-%d")

df.head()

#Null values checking
df.isnull().sum()

df[df.isnull().any(axis=1)]
df.shape
df.loc[(df["Returns"].isnull()) & (df["Churn"] == 0)]

#Couldn't find related information about null values so dropping from data
df.dropna(inplace = True)


#Drop duplicate variable
df.drop("Customer Age", inplace = True, axis = 1)

df.head()

df.describe().T

df.shape

#Re-calculate total purchase amount because of wrong calculation
df["TotalPrice"] = df["Product Price"] * df["Quantity"]
df.head()
df.drop("Total Purchase Amount", inplace = True, axis = 1)

df.head()

#Column names editing
df.columns = [col.strip().replace(" ", "") if col.strip() else col for col in df.columns]
df.head()


df.groupby("PaymentMethod").agg({"CustomerID": "count",
                                 "TotalPrice": "sum"})

df.sort_values("TotalPrice", ascending=False).head(10)

#Setting Total Price 0 for returns
df.loc[df["Returns"] == 1, "TotalPrice"] = 0

#Even if our income is 0 then I will look this as not purchased so removing from data
df = df[df["Returns"] != 1]
df.drop("Returns", inplace = True, axis=1)
df.head()



###RFM Metrics
df.groupby("CustomerID").agg({"TotalPrice":"sum"})
df["PurchaseDate"].max()
today_date = dt.datetime(year=2023, month=9, day=15)
rfm = df.groupby("CustomerID").agg({"PurchaseDate": lambda date: (today_date - date.max()).days,# Recency
                                    "CustomerID": lambda order_num: order_num.value_counts(),# Frequency
                                    "TotalPrice": "sum"})# Monetary
df.head()

rfm.head()
#New RFM column names
rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()

##RF Score Calculation
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, [1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels = [1,2,3,4,5])
rfm.head()

rfm["rf_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

rfm.head()

###Segmentation for RF Scores
seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}
rfm["segment"] = rfm["rf_score"].replace(seg_map, regex = True)

rfm.head()

# Adding Churn variable to rfm data so we can analyse with segments
rfm["churn"] = df.groupby("CustomerID").agg({"Churn": "any"})

rfm["churn"] = rfm["churn"].astype(int)
rfm.head()
rfm.loc[(rfm["churn"] == 1)]


rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean",
                            "churn": "mean"})
# Most churn rate coming from champions segment with the lowest recency value.
# It should be good some discount or different opportunities.





