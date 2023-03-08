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
from streamlit_functions import select_tree, getMeterPoints, getMeterReadings, check_password
from streamlit_tree_select import tree_select
from streamlit_extras.app_logo import add_logo

from pyecharts import options as opts
from pyecharts.charts import Bar, Pie, Grid, Line, Scatter, Sankey, WordCloud, HeatMap, Calendar, Sunburst, TreeMap
from streamlit_echarts import st_pyecharts



def run_again():
    if 'df_select' in st.session_state: del st.session_state['df_select']
    if 'df_over' in st.session_state:    del st.session_state['df_over']
    if 'kunde' in st.session_state:   del st.session_state['kunde']
    if 'df_besp' in st.session_state:    del st.session_state['df_besp']
    if 'valgt_meter' in st.session_state: del st.session_state.valgt_meter
    if 'df_meter' in st.session_state:    del st.session_state['df_meter']

st.set_page_config(layout="wide", page_title="Forside", page_icon="https://media.licdn.com/dms/image/C4E0BAQEwX9tzA6x8dw/company-logo_200_200/0/1642666749832?e=2147483647&v=beta&t=UiNzcE1RvJD3kHI218Al7omOzPLhHXXeE_svU4DIwEM")
st.sidebar.image('https://via.ritzau.dk/data/images/00181/e7ddd001-aee3-4801-845f-38483b42ba8b.png')

col1, col2 = st.columns([2,1])
col1.title('Forbrugsdata på erhvervsbygninger')

if check_password():
    col1.success('Login success')

    c = col1.container()

    kunder = ['FitnessWorld', 'Syntese', 'DanskeBank', 'Siemens Gamesa', 'NykreditMaegler', 'Bahne', 'Horsens Kommune', 'G4S', 'VinkPlast', 'MilestoneSystems', 'Premier Is']
    if 'kunde' not in st.session_state:
        valgt = col2.multiselect('Vælg kunde (må kun være en kunde)', kunder, max_selections=1)
        st.session_state['kunde'] = valgt
    else:
        valgt = col2.multiselect('Vælg kunde (må kun være en kunde)', kunder, default=st.session_state.kunde, on_change=run_again(), max_selections=1)
        st.session_state['kunde'] = valgt

    if not st.session_state.kunde:
        st.warning('Vær sød at vælge en kunde i højre hjørne') 
        st.stop()
    else:
        col2.success(str(st.session_state.kunde[0]) + ' valgt!') 


    @st.cache_data
    def meters_overblik():
        df = pd.read_csv('https://media.githubusercontent.com/media/Mikkel-schmidt/Elforbrug/main/Data/timeforbrug/' + quote(st.session_state.kunde[0]) + '.csv?token=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', usecols=['Adresse', 'meter', 'amount', 'from'], sep=',')
        #dff = pd.read_feather('https://raw.githubusercontent.com/Mikkel-schmidt/Elforbrug/main/Data/besp/' + st.session_state.kunde[0] + '.csv')
        return df

    df = meters_overblik()

    df['meter'] = pd.to_numeric(df['meter'])
    df['from'] = pd.to_datetime(df['from'], utc=True)
    #df = df.groupby('Adresse').mean().reset_index()
    

    if 'df_select' not in st.session_state:
        st.session_state['df_select'] = df.groupby(['Adresse', 'meter']).sum().reset_index()[['Adresse', 'meter']].drop_duplicates('meter')

    nodes = select_tree()

    @st.cache_resource
    def barr(df, grader):
        df = df.sort_values('årligt forbrug')
        b1 = (
            Bar()
            .add_xaxis(list(df['Adresse']))
            .add_yaxis('Samlet forbrug', list(df['årligt forbrug']), label_opts=opts.LabelOpts(is_show=False, formatter="{b}: {c}"),)
            .reversal_axis()
            .set_global_opts(
                datazoom_opts=[opts.DataZoomOpts(type_="inside", orient="vertical"), opts.DataZoomOpts(type_="slider", orient="vertical")], 
                #legend_opts=opts.LegendOpts(orient='vertical', pos_left="left", is_show=True),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=grader), name='Intensitet [kWh/m2]'),
                title_opts=opts.TitleOpts(title='Samlet forbrug', pos_left="center"),
                toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
            )
            .set_series_opts()
        )
        return b1

    with col2:
        figur = barr(df.groupby(df['from'].dt.month), 90)
        st_pyecharts(figur, height='500px')

    st.write(df.groupby('meter').agg({'Adresse': 'first', 'amount': 'sum'}).reset_index())








