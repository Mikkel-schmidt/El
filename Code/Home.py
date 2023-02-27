import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import quote
import statistics
from stqdm import stqdm
from datetime import datetime

import geopy
from geopy.extra.rate_limiter import RateLimiter
import folium
from streamlit_folium import st_folium
from streamlit_functions import select_tree, getMeterPoints, getMeterReadings
from streamlit_tree_select import tree_select
from streamlit_extras.app_logo import add_logo

from pyecharts import options as opts
from pyecharts.charts import Bar, Pie, Grid, Line, Scatter, Sankey, WordCloud, HeatMap, Calendar, Sunburst, TreeMap
from streamlit_echarts import st_pyecharts


import glob

from os import listdir
from os.path import isfile, join



#nodes = select_tree()

def run_again():
    if 'df_select' in st.session_state: del st.session_state['df_select']
    if 'df_over' in st.session_state:    del st.session_state['df_over']
    if 'kunde' in st.session_state:   del st.session_state['kunde']
    if 'df_besp' in st.session_state:    del st.session_state['df_besp']
    if 'valgt_meter' in st.session_state: del st.session_state.valgt_meter
    if 'df_meter' in st.session_state:    del st.session_state['df_meter']

st.set_page_config(layout="wide", page_title="Home", page_icon="https://media.licdn.com/dms/image/C4E0BAQEwX9tzA6x8dw/company-logo_200_200/0/1642666749832?e=2147483647&v=beta&t=UiNzcE1RvJD3kHI218Al7omOzPLhHXXeE_svU4DIwEM")
st.sidebar.image('https://via.ritzau.dk/data/images/00181/e7ddd001-aee3-4801-845f-38483b42ba8b.png')

col1, col2 = st.columns([2,1])
col1.title('Forbrugsdata på erhvervsbygninger')
col1.markdown('Loading data - please wait till done.')
c = col1.container()

kunder = ['FitnessWorld', 'Syntese', 'Danskebank', 'Siemens Gamesa', 'NykreditMaegler', 'Bahne', 'Horsens Kommune', 'G4S', 'VinkPlast', 'MilestoneSystems', 'Premier Is']
if 'kunde' not in st.session_state:
    valgt = col2.multiselect('Vælg kunde (må kun være en kunde)', kunder, max_selections=1)
    st.session_state['kunde'] = valgt
else:
    valgt = col2.multiselect('Vælg kunde (må kun være en kunde)', kunder, default=st.session_state.kunde, on_change=run_again(), max_selections=1)
    st.session_state['kunde'] = valgt

if not st.session_state.kunde:
  st.warning('Vær sød at vælge en kunde i højre hjørne') 
  st.stop()

#st.write(valgt)
data, IDs = getMeterPoints(valgt[0])
data['Adresse'] = data["streetName"] + " " + data['buildingNumber'] + ", " + data["postcode"] + ", " + data["cityName"]


@st.experimental_singleton
def get_map(data):
    service = geopy.Nominatim(user_agent = "myGeocoder")
    data["coordinates"] = data["Adresse"].apply(RateLimiter(service.geocode))#,min_delay_seconds=1))
    
    data = data.dropna(subset=['coordinates'])
    #st.dataframe(data["coordinates"])
    longs = [coord.longitude for coord in data["coordinates"]]
    lats = [coord.latitude for coord in data["coordinates"]]
    #st.write(longs)
    meanLong = statistics.mean(longs)
    meanLat = statistics.mean(lats)
    #st.write(data)
    m = folium.Map(location=[meanLat, meanLong])
    sw = [np.min(lats), np.min(longs)]
    ne = [np.max(lats), np.max(longs)]

    m.fit_bounds([sw, ne]) 

    for i in stqdm(range(0,data.shape[0])): # .shape [0] for Pandas DataFrame er antallet af rækker
        # opret markør for placering i 
        markerObj = folium.CircleMarker(location = [lats[i],longs[i]], fill=True, fill_color="#3186cc")#, tooltip=data['Adresse'][i],)
        # tilføj markør til kort
        markerObj.add_to(m)
    return m
    
c = st.container()
st.write(data)

if 'IDs' not in st.session_state:
    st.session_state['IDs'] = IDs

if 'df_select' not in st.session_state:
    st.session_state['df_select'] = data[['Adresse', 'meteringPointId']].sort_values('Adresse')

if 'df_over' not in st.session_state:
    st.session_state['df_over'] = data

nodes = select_tree()

with c:
    if data.shape[0] < 50:
        figure = get_map(data)
        st_data = st_folium(figure, width=1600, height=500)
    elif st.button('Generer kort (kan tage lidt tid)'):
        figure = get_map(data)
        st_data = st_folium(figure, width=1600, height=500)



