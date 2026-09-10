import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("cleaned_flight_data.csv")

# Drop identifiers / high-card raw cols / leakage cols
# ARR_DELAY & DEP_DELAY dropped ==> they directly give away the target (leakage)
df.drop(["FL_DATE","Aircraft Tail Number","OP_CARRIER_FL_NUM",
         "ORIGIN_CITY_NAME","ORIGIN_STATE_NM","DEST_CITY_NAME","DEST_STATE_NM",
         "ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID","ARR_DELAY","DEP_DELAY"],
        axis=1, inplace=True)

# extract hour from scheduled times
df['DEP_HOUR'] = pd.to_datetime(df['Scheduled Departure Time'], format='%H:%M').dt.hour
df['ARR_HOUR'] = pd.to_datetime(df['Scheduled Arrival Time'], format='%H:%M').dt.hour
df.drop(['Scheduled Departure Time','Scheduled Arrival Time'], axis=1, inplace=True)

# top 15 origin / dest, rest ==> Other
top_origin = df['ORIGIN'].value_counts().nlargest(15).index
top_dest = df['DEST'].value_counts().nlargest(15).index
df['ORIGIN'] = df['ORIGIN'].apply(lambda x: x if x in top_origin else 'Other')
df['DEST'] = df['DEST'].apply(lambda x: x if x in top_dest else 'Other')

X = df.drop('Arrival Delay ≥ 15 Minutes', axis=1)
Y = df['Arrival Delay ≥ 15 Minutes']

X = pd.get_dummies(X, drop_first=True, dtype=int)

from sklearn.model_selection import train_test_split
x_tarin, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_tarin_sc = sc.fit_transform(x_tarin)
x_test_sc = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=2000, random_state=42).fit(x_tarin_sc, y_train)

from sklearn.ensemble import RandomForestClassifier
# max_depth capped ==> unconstrained trees on 400k+ rows blow up the pickle size
# for deployment with barely any accuracy gain
rf = RandomForestClassifier(max_depth=12, random_state=42).fit(x_tarin, y_train)

# save everything the Streamlit app needs
# model_columns ==> exact dummy col order used at training, app must rebuild rows to match this

artifacts = {
    "lr_model": lr,
    "rf_model": rf,
    "scaler": sc,
    "model_columns": X.columns.tolist(),
    "top_origin": list(top_origin),
    "top_dest": list(top_dest)
}

with open("flight_delay_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("saved artifacts, columns:", len(X.columns))
