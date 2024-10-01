import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# load dataset
df = pd.read_csv("bread basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")

# Extract day of week (0 = Monday, 6 = Sunday)
df['day_of_week'] = df['date_time'].dt.dayofweek

# Create a mapping dictionary
day_mapping = {
    0: "senin",
    1: "selasa",
    2: "rabu",
    3: "kamis",
    4: "jumat",
    5: "sabtu",
    6: "minggu"
}

# Replace the numeric day of week with the day name
df['day'] = df['day_of_week'].map(day_mapping)

st.title("Market Basket Analysis menggunakan algoritma Apriori")

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    
    # Extract month from date_time column
    data['month'] = data['date_time'].dt.strftime('%B')
    
    filtered = data.loc[
        (data['period_day'].str.contains(period_day, case=False)) &
        (data['weekday_weekend'].str.contains(weekday_weekend, case=False)) &
        (data['month'].str.contains(month, case=False)) &
        (data['day'].str.contains(day, case=False))
    ]
    return filtered if filtered.shape[0] > 0 else "No Result"

# Define the user_input_features function
def user_input_features():
    item = st.selectbox("Item", ["Bread", "Butter", "Jam", "Milk", "Cheese", "Meat", "Cheese Burger", "Chicken Burger", "Chicken Wings", "French Fries", "Onion Rings", "Coca Cola", "Sprite", "Milkshake", "Ice Cream"])
    period_day = st.selectbox("Period_day", ["Morning", "Afternoon", "Evening", "Night"])
    weekday_weekend = st.selectbox("Weekday / Weekend", ["Weekday", "Weekend"])
    month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    return item, period_day, weekday_weekend, month, day

# Capture the return values from user_input_features
item, period_day, weekday_weekend, month, day = user_input_features()

# Now use these variables in the get_data function call
data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

def encode_data(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1

if type(data) != type("No Result"):
    item_count = data.groupby(['Transaction', 'item'])['item'].count().reset_index(name='count')
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='item', values='count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode_data)

    support = 0.01
    frequent_itemsets = apriori(item_count_pivot, min_support=support, use_colnames=True)
    
    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()
    data['antecedents'] = data['antecedents'].apply(parse_list)
    data['consequents'] = data['consequents'].apply(parse_list)

    return list(data.loc[data['antecedents'] == item_antecedents].iloc[0, :])

# If you're trying to display a specific recommendation
if type(data) != type("No Result"):
    st.subheader("Recommendation:")
    recommendation = return_item_df(item)
    if recommendation:
        st.success(f"If a customer buys {item}, they are likely to also buy {recommendation[1]}")
    else:
        st.info("No specific recommendation found for this item.")