#imports
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from geopy.geocoders import Nominatim 
from geopy.distance import great_circle

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.cluster import KMeans

df = pd.read_csv('locations.csv', sep=',')

# geopy in action
country = 'India'
city_names = df['location']

longitude = []
latitude = []
geolocator = Nominatim(user_agent="Trips")

for c in city_names.values:
    location = geolocator.geocode(c + ',' + country)
    latitude.append(location.latitude)
    longitude.append(location.longitude)

df['Latitude'] = latitude
df['Longtitude'] = longitude

l2 = df.iloc[:,-1:-3:-1]

print(l2)


kmeans = KMeans(5)
kmeans.fit(l2)

identified_clusters = kmeans.fit_predict(l2)
identified_clusters = list(identified_clusters)

print('identify_clusters: ',identified_clusters)

df['loc_clusters'] = identified_clusters

print('df: ',df)

input_city = input("Enter a city name:")
cluster = df.loc[df['location'] == input_city, 'loc_clusters']
print('cluster:', cluster)
cluster = cluster.iloc[0]
cities = df.loc[df['loc_clusters'] == cluster, 'location']
for c in range(len(cities)):
    if cities.iloc[c] == input_city:
        continue
    else:
        print(cities.iloc[c])