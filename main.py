from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, month, year, round
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder.appName("OnlineRetailAnalysis").getOrCreate()

df = spark.read.csv("Online Retail Data Set.xlsx - Online Retail.csv", header=True, inferSchema=True)


df = df.withColumn("Revenue", col("Quantity") * col("UnitPrice"))
df_clean = df.filter(col("Quantity") > 0)

country_revenue = df_clean.groupBy("Country") \
    .agg(round(sum("Revenue"), 2).alias("TotalRevenue")) \
    .orderBy(col("TotalRevenue").desc()).limit(10).toPandas()

plt.figure(figsize=(12, 6))
sns.barplot(x="TotalRevenue", y="Country", data=country_revenue, palette="viridis")
plt.title("Top 10 Countries by Total Revenue")
plt.show()


monthly_sales = df_clean.withColumn("Year", year("InvoiceDate")) \
    .withColumn("Month", month("InvoiceDate")) \
    .groupBy("Year", "Month") \
    .agg(sum("Revenue").alias("MonthlyRevenue")) \
    .orderBy("Year", "Month").toPandas()

monthly_sales['Date'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str)

plt.figure(figsize=(14, 6))
plt.plot(monthly_sales['Date'], monthly_sales['MonthlyRevenue'], marker='o', linestyle='-', color='b')
plt.xticks(rotation=45)
plt.title("Monthly Revenue Trend")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()


sample_data = df_clean.select("Quantity").sample(False, 0.1).toPandas()

plt.figure(figsize=(10, 4))
sns.boxplot(x=sample_data["Quantity"])
plt.title("Distribution of Order Quantities (Sampled)")
plt.xlim(0, 100) 
plt.show()