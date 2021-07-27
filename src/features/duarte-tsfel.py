import tsfel
import pandas as pd
import time

cfg = tsfel.get_features_by_domain()

df = pd.read_csv("C:\\Users\\marinara.marcato\\Project\\Scripts\\dog_posture\\src\\features\\sample.csv")
df = df.drop(columns=["Timestamp"])    # You should not pass the timestamp column. Time will be taken into consideration taking into account the length of your input and the sampling frequency.
t0 = time.time()
features = tsfel.time_series_features_extractor(cfg, df, 
                            fs=100, window_size=100, verbose = 0)
t1 = time.time()
print(t1 - t0)
print(features.shape)
features.to_csv('src\\features\\sample-tsfel.csv')

