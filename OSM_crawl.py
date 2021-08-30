import requests
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sin, cos, sqrt, atan2, radians

##############################################################
## Scraping POI locations around specific area as json file ##
##############################################################

def load_osm(lat_in, lon_in):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:500];
    node(around:3000,{lat},{lon})['amenity' = 'hospital']; out;
    node(around:3000,{lat},{lon})['leisure' = 'sports_centre']; out;
    way(around:3000,{lat},{lon})['leisure' = 'park']; out center qt;
    node(around:3000,{lat},{lon})['amenity' = 'bank']; out;
    node(around:3000,{lat},{lon})['amenity' = 'school']; out;
    node(around:3000,{lat},{lon})['amenity' = 'university']; out;
    node(around:3000,{lat},{lon})['shop' = 'mall']; out;
    node(around:3000,{lat},{lon})['shop' = 'department_store']; out;
        """
    # around is area radius in n meters
    filled_query=overpass_query.format(lat=lat_in, lon=lon_in)
    response = requests.get(overpass_url,
                            params={'data': filled_query})
    data = response.json()
    return data['elements']


###########################################
## Get the closest specific POI locations ##
###########################################

def get_closest_poi(lat_in, lon_in, poi_type):
    poi = poi_type
    lat = lat_in
    lon = lon_in
    query = poi+'+near+'+str(lat)+ '+' + str(lon)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    overpass_url = "https://nominatim.openstreetmap.org/search.php?format=json&limit=5&q=" + query

    response = requests.get(overpass_url, headers = headers)
    data = response.json()
    if(not data): return np.nan
    return data

#######################################################
## Calculate distance from the closest POI locations ##
#######################################################

def cal_distance(property_lat, property_lon, poi_lat, poi_lon):

# approximate radius of earth in km
    R = 6373.0

    lat1 = radians(property_lat)
    lon1 = radians(property_lon)
    lat2 = radians(poi_lat)
    lon2 = radians(poi_lon)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c