import streamlit as st
import pandas as pd
import numpy as np
import pyodbc 
from stqdm import stqdm
from datetime import datetime, date

import folium
from streamlit_folium import st_folium
from streamlit_functions import select_tree, getMeterPoints, getMeterReadings
from streamlit_tree_select import tree_select

import plotly.graph_objects as go
from pyecharts import options as opts
from pyecharts.charts import Bar, Pie, Grid, Line, Scatter, Sankey, WordCloud, HeatMap, Calendar, Sunburst, TreeMap
from streamlit_echarts import st_pyecharts


import glob

from os import listdir
from os.path import isfile, join

st.set_page_config(layout="wide", page_title="Energimærker", page_icon="https://media.licdn.com/dms/image/C4E0BAQEwX9tzA6x8dw/company-logo_200_200/0/1642666749832?e=2147483647&v=beta&t=UiNzcE1RvJD3kHI218Al7omOzPLhHXXeE_svU4DIwEM")
st.sidebar.image('https://via.ritzau.dk/data/images/00181/e7ddd001-aee3-4801-845f-38483b42ba8b.png')

nodes = select_tree()

df_over = st.session_state.df_over
#st.write(df_over)
#st.write(df_over['Adresse'].unique())
drivers = [item for item in pyodbc.drivers()]
st.write(drivers)

SERVER = "vsqlmorb"
PORT = '1433' 
USER = "pwu-morb"
PASSWORD = "JGo9822YbQvlcLtx"
DATABASE = "EmoData"

conn = pyodbc.connect('Driver={SQL Server};'
                    'Server=' + SERVER +';'
                    'Database='+ DATABASE +';'
                    'UID=' + USER +';'
                    'PWD=' + PASSWORD + ';')
cursor = conn.cursor()

@st.experimental_memo
def sql_query():
    df_adr = df_over[['streetName', 'buildingNumber', 'postcode', 'Adresse']].drop_duplicates(ignore_index=True)
    fejl = []
    dfs = []
    for rows in stqdm(range(df_adr.shape[0])):
        query = """
        SELECT top 1000 * 
        FROM stage.EMvaliditet
        WHERE (streetName_buildings = '{}') AND (HouseNumber_buildings = '{}') AND (PostalCode_buildings = '{}')
        """.format(df_adr['streetName'].iloc[rows].replace("'", "\\'\'"), df_adr['buildingNumber'].iloc[rows], df_adr['postcode'].iloc[rows])
        chunk = pd.read_sql(query, con=conn) 
        if chunk.empty: fejl.append(df_adr['Adresse'].iloc[rows])
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    df['ValidTo_reports'] = pd.to_datetime(df['ValidTo_reports'])
    df = df[df['ValidTo_reports'].dt.date >= date.today()]
    df = df.reset_index()
    fejl = pd.DataFrame(fejl, columns=['Adresse'])
    return df, fejl

#st.button('Run again', on_click=sql_query())

df, fejl = sql_query()
df['Adresse'] = 0
for row in stqdm(range(df.shape[0])):
    df['Adresse'].iloc[row] = df["StreetName_buildings"].iloc[row] + " " + \
    str(df['HouseNumber_buildings'].iloc[row]) + ", " + \
    str(df["PostalCode_buildings"].iloc[row]) + " " + \
    df["PostalCity_buildings"].iloc[row]

st.title('Energimærker')

col1, col2 = st.columns([2,2])

col1.header('Bygninger med energimærke')
col1.write(df)

fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="count",  x=df['StatusClassification_reports'], name='Klassificering'))
fig.add_trace(go.Histogram(histfunc="count",  x=df['AllProfitableProposalsClassification_reports'], name='Alle rentable forslag'))
fig.add_trace(go.Histogram(histfunc="count",  x=df['AllProposalsClassification_reports'], name='Alle forslag'))
fig.update_xaxes(categoryorder='array', categoryarray=['F', 'E', 'D', 'C', 'B', 'A2010', 'A2015', 'A2020'])
col2.plotly_chart(fig)

st.write('---')
col1, col2 = st.columns([2,2])
col1.header('Bygninger *uden* energimærke')
col1.write(fejl.sort_values('Adresse'))


@st.experimental_singleton
def piee(df):
    data = [list(z) for z in zip(df['Energimærke'], df['Antal'])]

    p = (
        Pie()
        .add(
            series_name='Forbrug i perioder',
            data_pair=data,
            #rosetype="area",
            radius=["40%", "70%"],
            #center=["85%", "50%"],
            label_opts=opts.LabelOpts(position="outside",
        ))
        .set_global_opts(
            legend_opts=opts.LegendOpts(orient='vertical', pos_left="right", type_='scroll', is_show=True),
            title_opts=opts.TitleOpts(
                title='Fordeling med/uden energimærke', pos_left="center"
            ),
            toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
        )
    )
    return p
d = {'Energimærke': ['Med energimærke', 'Uden energimærke'], 'Antal': [df['Adresse'].nunique(), fejl['Adresse'].nunique()]}
dff = pd.DataFrame(data=d)

with col2:
    figur = piee(dff)
    st_pyecharts(figur, height='400px')

st.write('---')

query = """
SELECT top 1000 * 
FROM dbo.Zones
WHERE BuildingId IN {} 
""".format(tuple(df['Id_buildings'].unique()))
zone = pd.read_sql(query, con=conn)
zone = zone.merge(df[['Id_buildings', 'Adresse']], left_on='BuildingId', right_on='Id_buildings')
#st.write(zone)

@st.experimental_memo
def sqll(dbo, zone):
    query = """
    SELECT top 1000 * 
    FROM dbo.{}
    WHERE ZoneId IN {} 
    """.format(dbo, tuple(zone['Id'].unique()))
    chunk = pd.read_sql(query, con=conn)
    chunk = chunk.merge(zone[['Id', 'Adresse']], left_on='ZoneId', right_on='Id', suffixes=("", "_"))
    chunk.insert(0, 'Adresse', chunk.pop('Adresse'))
    return chunk

vent = sqll('Ventilations', zone)
#st.write(vent)
vent = vent.groupby('Adresse').agg({'Id': 'count', 'Area': 'sum', 'UsageFactor': 'sum', 'HasHeatCoils': 'count'}).reset_index()
#st.write(vent)

sc = sqll('SolarCells', zone)
#st.write(sc)
sc = sc.groupby('Adresse').agg({'Id': 'count', 'Area': 'sum', 'PeakPower': 'sum', 'Efficiency': 'sum'}).reset_index()
#st.write(sc)

shp = sqll('SolarHeatingPlants', zone)
#st.write(shp)
#shp
#st.write(shp.groupby('Adresse').agg({'Id': 'count', 'Area': 'sum', 'PeakPower': 'sum', 'Efficiency': 'sum'}).reset_index())

hp = sqll('HeatPumps', zone)
#st.write(hp)
hp = hp.groupby('Adresse').agg({'Id': 'count', 'ShareOfFloorArea': 'sum', 'RoomHeatingNominalEffect': 'sum', 'RoomHeatingNominalCOP': 'sum'}).reset_index()
#st.write(hp)

df_samlet = vent.merge(sc, on='Adresse', how='outer', suffixes=('_vent', '_sc'))
df_samlet = df_samlet.merge(hp, on='Adresse', how='outer', suffixes=('', '_hp'))

st.header('Overblik over væsentlig data i energimærker')
st.write(df_samlet)




