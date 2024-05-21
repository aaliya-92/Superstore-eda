#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd


# In[51]:


df = pd.read_csv("Superstore 2023.csv")


# In[3]:


df.info


# In[4]:


df.columns


# In[5]:


df.sample(5)


# In[3]:


df.head()


# In[4]:


import matplotlib as plt


# In[5]:


import matplotlib.pyplot as plt

# Group data by category and calculate total sales
category_sales = df.groupby('Category')['Sales'].sum()

# Plotting bar chart
plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar', color='skyblue')
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()


# In[6]:


# Convert 'Order Date' to datetime and group by month
df['Order Date'] = pd.to_datetime(df['Order Date'])
monthly_data = df.resample('M', on='Order Date').sum()

# Plotting line chart
plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Sales'], marker='o', linestyle='-')
plt.title('Trends in Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[7]:


# Pie chart for sales by product category
plt.figure(figsize=(8, 8))
plt.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', startangle=140)
plt.title('Sales Distribution by Product Category')
plt.axis('equal')
plt.show()

# Pie chart for distribution of shipping modes
ship_mode_counts = df['Ship Mode'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(ship_mode_counts, labels=ship_mode_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Shipping Modes')
plt.axis('equal')
plt.show()


# In[8]:


# Scatter plot for sales vs. profit
plt.figure(figsize=(8, 6))
plt.scatter(df['Sales'], df['Profit'], color='green', alpha=0.5)
plt.title('Sales vs. Profit')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.grid(True)
plt.show()

# Scatter plot for quantity vs. profit
plt.figure(figsize=(8, 6))
plt.scatter(df['Quantity'], df['Profit'], color='blue', alpha=0.5)
plt.title('Quantity vs. Profit')
plt.xlabel('Quantity')
plt.ylabel('Profit')
plt.grid(True)
plt.show()


# In[9]:


import seaborn as sns

# Calculate correlation matrix
correlation_matrix = df[['Sales', 'Quantity', 'Discount', 'Profit']].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[10]:


df[df['Sales']].sum()


# In[12]:


total_sales = df['Sales'].sum()
print("Total Sales:", total_sales)


# In[13]:


# Convert 'Order Date' to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract year from 'Order Date'
df['Year'] = df['Order Date'].dt.year

# Group by year and sum up the sales
yearly_sales = df.groupby('Year')['Sales'].sum()

# Print the yearly sales
print("Yearly Sales:")
print(yearly_sales)


# In[14]:


total_sales = df['Sales'].sum()
print("Total sales:", total_sales)

# Convert 'Order Date' column to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract year from 'Order Date' column and sum up sales for each year
sales_by_year = df.groupby(df['Order Date'].dt.year)['Sales'].sum()
print("Sales by year:")
print(sales_by_year)

# Extract year and month from 'Order Date' column and sum up sales for each combination
sales_by_year_month = df.groupby([df['Order Date'].dt.year, df['Order Date'].dt.month])['Sales'].sum()
print("Sales by year and month:")
print(sales_by_year_month)


# In[15]:


import matplotlib.pyplot as plt

# Convert 'Order Date' column to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract year from 'Order Date' column
df['Year'] = df['Order Date'].dt.year

# Group sales by year and calculate the sum
sales_by_year = df.groupby('Year')['Sales'].sum()

# Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(sales_by_year, labels=sales_by_year.index, autopct='%1.1f%%', startangle=140)
plt.title('Sales Distribution by Year')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Order Date' and 'Sales' columns
# Convert 'Order Date' column to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract month from 'Order Date' column
df['Month'] = df['Order Date'].dt.month

# Calculate average sales for each month
average_sales_by_month = df.groupby('Month')['Sales'].mean()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(average_sales_by_month, labels=average_sales_by_month.index, autopct='%1.1f%%', startangle=140)
plt.title('Average Sales by Month')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Order Date' and 'Sales' columns

# Convert 'Order Date' column to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract year and month from 'Order Date' column
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# Group by year and month, and calculate sum of sales
sales_by_year_month = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

# Plot pie charts for each year
years = sales_by_year_month['Year'].unique()
for year in years:
    sales_year = sales_by_year_month[sales_by_year_month['Year'] == year]
    sales = sales_year['Sales']
    months = sales_year['Month']
    
    # Create pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sales, labels=months, autopct='%1.1f%%', startangle=140)
    plt.title(f'Sales Distribution for Year {year}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.show()


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Order Date' and 'Sales' columns
# Convert 'Order Date' column to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Group by 'Order Date' and calculate total sales for each date
total_sales_by_date = df.groupby('Order Date')['Sales'].sum()

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(total_sales_by_date.index, total_sales_by_date.values, marker='o', linestyle='-')
plt.title('Total Sales Revenue Over Time')
plt.xlabel('Order Date')
plt.ylabel('Total Sales Revenue')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)  # Add grid lines
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Order Date' and 'Sales' columns

# Convert 'Order Date' column to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Group by 'Order Date' (e.g., by month or by year) and calculate total sales revenue for each group
sales_over_time = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()

# Plot the total sales revenue over time
plt.figure(figsize=(10, 6))
sales_over_time.plot(kind='line', marker='o', color='b', linestyle='-')
plt.title('Total Sales Revenue Over Time')
plt.xlabel('Order Date')
plt.ylabel('Total Sales Revenue')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[20]:


import plotly.express as px

# Assuming df is your DataFrame with 'Order Date' and 'Sales' columns

# Convert 'Order Date' column to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Group by order date and calculate sum of sales
sales_by_date = df.groupby('Order Date')['Sales'].sum().reset_index()

# Plotting the interactive line chart
fig = px.line(sales_by_date, x='Order Date', y='Sales', title='Total Sales Revenue Over Time')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Total Sales Revenue')
fig.show()


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Category' and 'Sales' columns

# Group by category and calculate sum of sales
sales_by_category = df.groupby('Category')['Sales'].sum().reset_index()

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(sales_by_category['Category'], sales_by_category['Sales'])
plt.title('Sales Revenue Across Different Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Total Sales Revenue')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[22]:


import plotly.express as px

# Assuming df is your DataFrame with 'Category' and 'Sales' columns

# Group by category and calculate sum of sales
sales_by_category = df.groupby('Category')['Sales'].sum().reset_index()

# Plotting the bar chart
fig = px.bar(sales_by_category, x='Category', y='Sales', 
             title='Sales Revenue Across Different Product Categories',
             labels={'Sales': 'Total Sales Revenue', 'Category': 'Product Category'})
fig.update_layout(xaxis_tickangle=-45, yaxis_title='Total Sales Revenue')
fig.show()


# In[23]:


import pandas as pd
import plotly.express as px

# Assuming df is your DataFrame with 'Category', 'Sub-Category', and 'Profit' columns

# Group by category and sub-category and calculate sum of profit
profit_by_category_subcategory = df.groupby(['Category', 'Sub-Category'])['Profit'].sum().reset_index()

# Plotting the horizontal bar chart
fig = px.bar(profit_by_category_subcategory, 
             x='Profit', 
             y='Sub-Category', 
             color='Category',
             title='Profitability of Each Product Category/Sub-Category',
             labels={'Profit': 'Total Profit', 'Sub-Category': 'Product Sub-Category', 'Category': 'Product Category'},
             orientation='h')
fig.update_layout(yaxis=dict(autorange="reversed"))  # Reverse the y-axis to display sub-categories from top to bottom
fig.show()


# In[25]:


import pandas as pd
import plotly.express as px

# Assuming df is your DataFrame with 'Discount' and 'Profit' columns

# Plotting the scatter plot
fig = px.scatter(df, x='Discount', y='Profit',
                 title='Relationship between Discount Offered and Profitability',
                 labels={'Discount': 'Discount Offered (%)', 'Profit': 'Profit'},
                 hover_name='Order ID', hover_data={'Profit': ':.2f', 'Discount': ':.2f'},
                 trendline='ols')  # Add a linear trendline

# Update layout for better interactivity
fig.update_layout(hovermode='closest')

# Show the plot
fig.show()


# In[26]:


import pandas as pd
import plotly.express as px

# Assuming df is your DataFrame with 'State' and 'Sales' columns

# Group by state and calculate total sales
sales_by_state = df.groupby('State')['Sales'].sum().reset_index()

# Find states with the highest sales
top_states = sales_by_state.nlargest(5, 'Sales')

# Plotting the pie chart
fig = px.pie(top_states, 
             values='Sales', 
             names='State', 
             title='States with Highest Sales',
             labels={'Sales': 'Total Sales', 'State': 'State'})
fig.show()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'City', 'State', and 'Sales' columns

# Filter data for California
ca_df = df[df['State'] == 'California']

# Group by city and calculate sum of sales
sales_by_city = ca_df.groupby('City')['Sales'].sum().reset_index()

# Sort cities by sales in descending order
sales_by_city = sales_by_city.sort_values(by='Sales', ascending=False)

# Extract top 5 cities with highest sales
top_cities = sales_by_city.head(5)

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(top_cities['Sales'], labels=top_cities['City'], autopct='%1.1f%%', startangle=140)
plt.title('Top Cities in California with Highest Sales')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[28]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Region', 'State', and 'Sales' columns

# Filter data for California
california_sales = df[df['State'] == 'California']

# Group by region and calculate total sales for each region
sales_by_region = california_sales.groupby('Region')['Sales'].sum()

# Identify the region(s) with the highest sales
highest_sales_regions = sales_by_region[sales_by_region == sales_by_region.max()]

# Plot a pie chart for regions with highest sales
plt.figure(figsize=(8, 6))
plt.pie(highest_sales_regions, labels=highest_sales_regions.index, autopct='%1.1f%%', startangle=140)
plt.title('Regions in California with Highest Sales')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[29]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Product Name' and 'Profit' columns

# Group by product and calculate total profit for each product
profit_by_product = df.groupby('Product Name')['Profit'].sum()

# Sort the products based on profitability and select the top N
N = 10  # Number of top products to display
top_profitable_products = profit_by_product.nlargest(N)

# Plot a horizontal bar chart for the top N most profitable products
plt.figure(figsize=(10, 6))
top_profitable_products.sort_values().plot(kind='barh', color='skyblue')
plt.title(f'Top {N} Most Profitable Products')
plt.xlabel('Total Profit')
plt.ylabel('Product')
plt.gca().invert_yaxis()  # Invert y-axis to display the most profitable product on top
plt.grid(axis='x')
plt.show()


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Segment' and 'Profit' columns

# Group by segment and calculate total profit for each segment
profit_by_segment = df.groupby('Segment')['Profit'].sum()

# Identify the most profitable segment
most_profitable_segment = profit_by_segment.idxmax()

# Plot a horizontal bar chart for profitability of each segment
plt.figure(figsize=(10, 6))
plt.barh(profit_by_segment.index, profit_by_segment.values, color='skyblue')
plt.xlabel('Total Profit')
plt.ylabel('Segment')
plt.title('Total Profit by Segment')
plt.axhline(y=most_profitable_segment, color='red', linestyle='--', label='Most Profitable Segment')
plt.legend()
plt.show()


# In[31]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Product Name' and 'Sales' columns

# Sort the dataset by sales in descending order
sorted_df = df.sort_values(by='Sales', ascending=False)

# Select the top N best-selling products
top_n = 10  # Define the number of top products you want to consider
top_products = sorted_df.head(top_n)

# Plot a bar chart to visualize the sales of top products
plt.figure(figsize=(12, 6))
plt.bar(top_products['Product Name'], top_products['Sales'], color='skyblue')
plt.xlabel('Product Name')
plt.ylabel('Sales')
plt.title(f'Top {top_n} Best-Selling Products')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Optional: Overlay a cumulative percentage line for Pareto chart
cumulative_percentage = (top_products['Sales'].cumsum() / top_products['Sales'].sum()) * 100
plt.twinx()
plt.plot(top_products['Product Name'], cumulative_percentage, color='red', marker='o', linestyle='-')
plt.ylabel('Cumulative Percentage (%)')
plt.grid(False)

# Show the plot
plt.show()


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Segment', 'Category', and 'Sales' columns

# Group the data by customer segment and category, and calculate total sales revenue
sales_by_segment_category = df.groupby(['Segment', 'Category'])['Sales'].sum().unstack()

# Plot a stacked bar chart to visualize the sales revenue breakdown by customer segment
sales_by_segment_category.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sales Revenue Breakdown by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Sales Revenue')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Category')
plt.show()


# In[35]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Order Date' and 'Sales' columns

# Convert 'Order Date' column to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract month and year from 'Order Date' column
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year

# Group the data by month and year, and calculate total sales for each month
monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum()

# Plot a line chart to visualize the seasonal variation in sales over time
plt.figure(figsize=(10, 6))
monthly_sales.plot(marker='o', linestyle='-')
plt.title('Seasonal Variation in Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()


# In[37]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Customer ID' and 'Order Date' columns

# Convert 'Order Date' column to datetime format if it's not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract the month and year of the first purchase for each customer
df['First Purchase Month'] = df.groupby('Customer ID')['Order Date'].transform('min').dt.to_period('M')

# Calculate the number of returning customers for each cohort in each period
cohort_data = df.groupby(['First Purchase Month', df['Order Date'].dt.to_period('M')])['Customer ID'].nunique().unstack()

# Calculate the percentage of returning customers for each cohort in each period
returning_customers_percentage = (cohort_data.divide(cohort_data.iloc[:, 0], axis=0) * 100).round(2)

# Convert 'Period' index to string for plotting
returning_customers_percentage.index = returning_customers_percentage.index.strftime('%Y-%m')

# Plot a line chart to visualize the percentage of returning customers over time for each cohort
plt.figure(figsize=(10, 6))
for cohort in returning_customers_percentage.columns:
    plt.plot(returning_customers_percentage.index, returning_customers_percentage[cohort], marker='o', label=cohort)
plt.title('Cohort Analysis: Percentage of Returning Customers Over Time')
plt.xlabel('Months Since First Purchase')
plt.ylabel('Percentage of Returning Customers')
plt.legend(title='Cohort')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# In[38]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Discount' and 'Sales' columns

# Split the data into two groups: with discounts and without discounts
with_discounts = df[df['Discount'] > 0]['Sales']
without_discounts = df[df['Discount'] == 0]['Sales']

# Plot a box plot comparing sales revenue for products with and without discounts
plt.figure(figsize=(10, 6))
sns.boxplot(x='Discount', y='Sales', data=df)
plt.title('Comparison of Sales Revenue for Products with and without Discounts')
plt.xlabel('Discount')
plt.ylabel('Sales Revenue')
plt.xticks([0, 1], ['Without Discount', 'With Discount'])
plt.grid(True)
plt.show()

# Alternatively, plot a violin plot for better visualization of the distribution
plt.figure(figsize=(10, 6))
sns.violinplot(x='Discount', y='Sales', data=df)
plt.title('Comparison of Sales Revenue for Products with and without Discounts')
plt.xlabel('Discount')
plt.ylabel('Sales Revenue')
plt.xticks([0, 1], ['Without Discount', 'With Discount'])
plt.grid(True)
plt.show()


# In[42]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Order Date' and 'Ship Date' columns

# Convert 'Order Date' and 'Ship Date' columns to datetime format with the correct date format
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')

# Calculate order processing time (in days)
df['Order Processing Time'] = (df['Ship Date'] - df['Order Date']).dt.days

# Plot a histogram to visualize the distribution of order processing time
plt.figure(figsize=(10, 6))
plt.hist(df['Order Processing Time'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Order Processing Time')
plt.xlabel('Order Processing Time (days)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Alternatively, plot a box plot to visualize the distribution
plt.figure(figsize=(8, 6))
plt.boxplot(df['Order Processing Time'], vert=False)
plt.title('Distribution of Order Processing Time')
plt.xlabel('Order Processing Time (days)')
plt.grid(True)
plt.show()


# In[45]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Ship Mode' and 'Order Processing Time' columns

# Calculate average shipping times for different shipping modes
avg_shipping_times = df.groupby('Ship Mode')['Order Processing Time'].mean().reset_index()

# Plot the grouped bar chart
plt.figure(figsize=(10, 6))
plt.bar(avg_shipping_times['Ship Mode'], avg_shipping_times['Order Processing Time'], color='skyblue')
plt.title('Average Shipping Times for Different Shipping Modes')
plt.xlabel('Shipping Mode')
plt.ylabel('Average Shipping Time (days)')

# Adjust y-axis limits to move x-axis labels below the bars
plt.ylim(0, max(avg_shipping_times['Order Processing Time']) * 1.1)

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(axis='y')  # Add gridlines only on the y-axis

plt.show()


# In[46]:


# Group the data by 'Category' and sum up the sales
category_sales = df.groupby('Category')['Sales'].sum()

# Sort the categories based on total sales in descending order
category_sales_sorted = category_sales.sort_values(ascending=False)

# Retrieve the top selling categories
top_selling_categories = category_sales_sorted.head()

print("Top Selling Product Categories:")
print(top_selling_categories)


# In[53]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Set 'Order Date' as index
df.set_index('Order Date', inplace=True)

# Group sales by month
monthly_sales = df.resample('M')['Sales'].sum()

# Plot sales trend over time
plt.figure(figsize=(10, 6))
monthly_sales.plot(color='blue', marker='o', linestyle='-')
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[54]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Group by state and calculate total profit
state_profit = df.groupby('State')['Profit'].sum().reset_index()

# Sort states by profit in descending order
state_profit_sorted = state_profit.sort_values(by='Profit', ascending=False)

# Select top performing states (e.g., top 10)
top_states = state_profit_sorted.head(10)

# Print or display the top performing states
print("Top Performing States in Terms of Profit:")
print(top_states)


# In[55]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Group by state and calculate total profit
state_profit = df.groupby('State')['Profit'].sum().reset_index()

# Sort states by profit in descending order
state_profit_sorted = state_profit.sort_values(by='Profit', ascending=False)

# Select names of states and corresponding profits
top_states = state_profit_sorted[['State', 'Profit']]

# Print or display the top performing states and their profits
print("Top Performing States in Terms of Profit:")
print(top_states)


# In[56]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Calculate order size (quantity of items per order)
order_size = df.groupby('Order ID')['Quantity'].sum()

# Calculate frequency of purchase (number of orders)
frequency_of_purchase = df['Order ID'].nunique()

# Print or display the order size and frequency of purchase
print("Average Order Size (Quantity of Items per Order):", order_size.mean())
print("Frequency of Purchase (Number of Orders):", frequency_of_purchase)


# In[58]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Calculate order size (number of items per order)
order_size = df.groupby('Order ID')['Quantity'].sum()

# Calculate frequency of purchase (number of orders per customer)
purchase_frequency = df.groupby('Customer ID')['Order ID'].nunique()

# Plotting order size distribution
plt.figure(figsize=(10, 6))
plt.hist(order_size, bins=range(1, max(order_size)+1), color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Order Sizes')
plt.xlabel('Order Size (Number of Items)')
plt.ylabel('Frequency')
plt.grid(True)
plt.xticks(range(1, max(order_size)+1))
plt.tight_layout()
plt.show()

# Print average order size and purchase frequency
print("Average Order Size:", order_size.mean())
print("Average Purchase Frequency:", purchase_frequency.mean())


# In[59]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Group by category and sub-category and calculate total sales
sales_by_category_subcategory = df.groupby(['Category', 'Sub-Category'])['Sales'].sum().unstack()

# Plotting the distribution of sales across categories and sub-categories
plt.figure(figsize=(12, 8))
sales_by_category_subcategory.plot(kind='bar', stacked=True)
plt.title('Distribution of Sales Across Categories and Sub-Categories')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sub-Category', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# In[60]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Group by category and calculate total sales
sales_by_category = df.groupby('Category')['Sales'].sum().reset_index()

# Sort categories by sales in descending order
sales_by_category_sorted = sales_by_category.sort_values(by='Sales', ascending=False)

# Print or display the highest performing categories
print("Highest Performing Categories based on Sales:")
print(sales_by_category_sorted)


# In[61]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Group by sub-category and calculate total sales
sales_by_subcategory = df.groupby('Sub-Category')['Sales'].sum()

# Sort sub-categories by sales in descending order
top_subcategories = sales_by_subcategory.sort_values(ascending=False)

# Print or display the top performing sub-categories
print("Top Performing Sub-Categories in Terms of Sales:")
print(top_subcategories.head(10))


# In[63]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract year and month from 'Order Date'
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# Group by year and month, and calculate total sales
sales_by_year_month = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

# Find the highest performing month year-wise
highest_performing_months = sales_by_year_month.loc[sales_by_year_month.groupby('Year')['Sales'].idxmax()]

# Map month numbers to month names
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

# Replace month numbers with month names
highest_performing_months['Month'] = highest_performing_months['Month'].map(month_names)

# Print or display the highest performing months year-wise
print("Highest Performing Months Year-wise:")
print(highest_performing_months)


# In[75]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Group by ship mode and calculate total sales for each mode
sales_by_ship_mode = df.groupby('Ship Mode')['Sales'].sum().reset_index()

# Find the ship mode with the highest sales
highest_sales_ship_mode = sales_by_ship_mode.loc[sales_by_ship_mode['Sales'].idxmax()]

print(f"The ship mode with the highest sales is '{highest_sales_ship_mode['Ship Mode']}' with total sales of ${highest_sales_ship_mode['Sales']:.2f}.")


# In[77]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Group by ship mode and calculate total sales for each mode
sales_by_ship_mode = df.groupby('Ship Mode')['Sales'].sum().reset_index()

print(sales_by_ship_mode)


# In[78]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Group by ship mode and calculate total sales for each mode
sales_by_ship_mode = df.groupby('Ship Mode')['Sales'].sum().reset_index()

# Set the display format to avoid scientific notation
pd.options.display.float_format = '{:.2f}'.format

print(sales_by_ship_mode)


# In[79]:


import pandas as pd

# Load the dataset
df = pd.read_csv('Superstore 2023.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

# Group by ship mode and sum the sales for each group
sales_by_ship_mode = df.groupby('Ship Mode')['Sales'].sum().reset_index()

# Sort the result in descending order of sales
sales_by_ship_mode_sorted = sales_by_ship_mode.sort_values(by='Sales', ascending=False)

# Print the result
print(sales_by_ship_mode_sorted)


# In[81]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
data = pd.read_csv("superstore 2023.csv")

# Convert 'Order Date' column to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'])

# Extract year from 'Order Date' column
data['Year'] = data['Order Date'].dt.year

# Filter data for the years 2020 to 2023
filtered_data = data[(data['Year'] >= 2020) & (data['Year'] <= 2023)]

# Group data by year and calculate total sales for each year
yearly_sales = filtered_data.groupby('Year')['Sales'].sum().reset_index()

# Prepare the data for linear regression
X = yearly_sales['Year'].values.reshape(-1, 1)  # Year
y = yearly_sales['Sales'].values.reshape(-1, 1)  # Total Sales

# Perform linear regression
regression_model = LinearRegression()
regression_model.fit(X, y)

# Predict sales for the years 2020 to 2023
future_years = np.array([[2020], [2021], [2022], [2023]])
predicted_sales = regression_model.predict(future_years)

# Output the results
for year, sales in zip(range(2020, 2024), predicted_sales):
    print(f"Predicted sales for {year}: ${sales[0]:.2f}")


# In[83]:


import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("Superstore 2023.csv")

# Extract year from 'Order Date' column
data['Year'] = pd.to_datetime(data['Order Date']).dt.year

# Calculate total sales for each year
yearly_sales = data.groupby('Year')['Sales'].sum().reset_index()

# Calculate year-on-year growth rate
yearly_sales['Yearly Growth'] = yearly_sales['Sales'].pct_change() * 100

# Drop the first row as it will have NaN for year-on-year growth
yearly_sales = yearly_sales.dropna()

# Prepare data for regression analysis
X = yearly_sales['Year'].values.reshape(-1, 1)
y = yearly_sales['Yearly Growth'].values.reshape(-1, 1)

# Perform linear regression
regression_model = LinearRegression()
regression_model.fit(X, y)

# Predict year-on-year growth for future years
future_years = [[year] for year in range(2024, 2030)]
predicted_growth = regression_model.predict(future_years)

# Display predicted year-on-year growth for future years
future_growth_df = pd.DataFrame({'Year': range(2024, 2030), 'Predicted Growth Rate': predicted_growth.flatten()})
print(future_growth_df)


# In[84]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("Superstore 2023.csv")

# Extract year from 'Order Date' column
data['Year'] = pd.to_datetime(data['Order Date']).dt.year

# Calculate total sales for each year
yearly_sales = data.groupby('Year')['Sales'].sum().reset_index()

# Calculate year-on-year growth rate
yearly_sales['Yearly Growth'] = yearly_sales['Sales'].pct_change() * 100

# Drop the first row as it will have NaN for year-on-year growth
yearly_sales = yearly_sales.dropna()

# Prepare data for regression analysis
X = yearly_sales['Year'].values.reshape(-1, 1)
y = yearly_sales['Yearly Growth'].values.reshape(-1, 1)

# Perform linear regression
regression_model = LinearRegression()
regression_model.fit(X, y)

# Predict year-on-year growth for future years
future_years = np.array(range(2024, 2030)).reshape(-1, 1)
predicted_growth = regression_model.predict(future_years)

# Plot the historical data
plt.figure(figsize=(10, 6))
plt.scatter(yearly_sales['Year'], yearly_sales['Yearly Growth'], color='blue', label='Historical Data')

# Plot the regression line
plt.plot(yearly_sales['Year'], regression_model.predict(X), color='red', label='Regression Line')

# Plot the predicted growth for future years
plt.plot(future_years, predicted_growth, color='green', linestyle='--', label='Predicted Growth')

plt.title('Year-on-Year Sales Growth')
plt.xlabel('Year')
plt.ylabel('Year-on-Year Growth Rate (%)')
plt.legend()
plt.grid(True)
plt.show()


# In[90]:


import pandas as pd

# Load the dataset
data = pd.read_csv("Superstore 2023.csv")

# Group the data by customer and calculate total sales for each customer
customer_sales = data.groupby(['Customer ID', 'Customer Name', 'State', 'City', 'Region', 'Category', 'Sub-Category', 'Quantity'])['Sales'].sum().reset_index()

# Sort customers based on total sales in descending order
customer_sales_sorted = customer_sales.sort_values(by='Sales', ascending=False)

# Retrieve top 10 highest grossing customers
top_customers = customer_sales_sorted.head(10)

# Display the highest grossing customers along with their states, cities, and regions
print("Top 10 Highest Grossing Customers:")
print(top_customers[['Customer Name', 'State', 'City', 'Region', 'Sales', 'Category', 'Sub-Category', 'Quantity']])


# In[ ]:




