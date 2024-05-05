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

df = pd.read_csv('cities.csv', sep=';')

# print("df: ", df)


country_names = df['country_name']
city_names = df['name']
state_names = df['state_name']
city_names.to_csv('citiesExport.csv', sep=';')
latitude = []
longitude = []
latitude = df['latitude'] 
longitude = df['longitude'] 
for i in range(len(latitude)):
    # Xóa tất cả các dấu chấm từ dấu chấm thứ 2 trở đi
    latitude[i] = latitude[i].replace(".", "", latitude[i].count('.') - 1)
    # Chuyển đổi chuỗi thành số thập phân và chia cho 1.000.000 để đưa về dạng thập phân
    latitude[i] = float(latitude[i]) / 1000000

latitude.to_csv('latitude.csv', sep=';')

for i in range(len(longitude)):
    # Xóa tất cả các dấu chấm từ dấu chấm thứ 2 trở đi
    longitude[i] = longitude[i].replace(".", "", longitude[i].count('.') - 1)
    # Chuyển đổi chuỗi thành số thập phân và chia cho 1.000.000 để đưa về dạng thập phân
    longitude[i] = float(longitude[i]) / 1000000

longitude.to_csv('longitude.csv', sep=';')

l2 = pd.DataFrame({'latitude': latitude, 'longitude': longitude})

# print("l2:", l2)


kmeans = KMeans(20)
kmeans.fit(l2)

identified_clusters = kmeans.fit_predict(l2)
identified_clusters = list(identified_clusters)

# print('identify_clusters: ',identified_clusters)

training = pd.DataFrame({'city': city_names, 'state': state_names, 'country': country_names, 'latitude': latitude, 'longitude': longitude, 'loc_clusters' : identified_clusters})

# print('training: ',training)

import sys
import json

input_city = str(sys.argv[1])
cluster = training.loc[training['city'] == input_city, 'loc_clusters']
# print('cluster:', cluster)
cluster = cluster.iloc[0]
cities_and_states = training.loc[training['loc_clusters'] == cluster, ['city', 'state']]

cityList = []

for c in range(len(cities_and_states)):
    city = cities_and_states.iloc[c]['city']
    state = cities_and_states.iloc[c]['state']
    if city == input_city:
        continue
    else:
        # print(city + ',' + state)
        cityList.append(city + ',' + state)

print(json.dumps(cityList))
sys.stdout.flush()