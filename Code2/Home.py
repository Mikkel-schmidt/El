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

import locale
#for lang in locale.windows_locale.values():
#    st.write(lang)

locale.setlocale(locale.LC_ALL, "da_DK") 



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

    kunder = ['FitnessWorld', 'Jeudan1', 'BallerupKommune', 'BallerupKommune1', 'Privat_elforbrug_test', 'Syntese', 'DanskeBank', 'Siemens Gamesa', 'NykreditMaegler', 'Bahne', 'Horsens Kommune', 'G4S', 'VinkPlast', 'MilestoneSystems', 'Premier Is']
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
        df = pd.read_csv('https://media.githubusercontent.com/media/Mikkel-schmidt/Elforbrug/main/Data/timeforbrug/' + quote(st.session_state.kunde[0]) + '.csv?token=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', usecols=['Adresse', 'meter', 'amount', 'from', 'bkps'], sep=',')
        #dff = pd.read_feather('https://raw.githubusercontent.com/Mikkel-schmidt/Elforbrug/main/Data/besp/' + st.session_state.kunde[0] + '.csv')
        return df

    df = meters_overblik()

    df['meter'] = pd.to_numeric(df['meter'])
    df['from'] = pd.to_datetime(df['from'], utc=True) 
    #df = df.groupby('Adresse').mean().reset_index()
    
    if 'df_select' not in st.session_state:
        st.session_state['df_select'] = df[['meter', 'Adresse', 'amount', 'bkps']].groupby(['Adresse', 'meter']).sum().reset_index()[['Adresse', 'meter']].drop_duplicates('meter')

    nodes = select_tree()

    #st.write(df.groupby(df['from'].dt.month).sum().reset_index())

    @st.cache_resource
    def barr(df, grader):
        #df = df.sort_values('årligt forbrug')
        b1 = (
            Bar()
            .add_xaxis(list(df['tid']))
            .add_yaxis('Samlet forbrug', list(df['amount']), label_opts=opts.LabelOpts(is_show=False, formatter="{b}: {c}"),)
            #.reversal_axis()
            .set_global_opts(
                #datazoom_opts=[opts.DataZoomOpts(type_="inside", orient="vertical"), opts.DataZoomOpts(type_="slider", orient="vertical")], 
                #legend_opts=opts.LegendOpts(orient='vertical', pos_left="left", is_show=True),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=grader), name='Intensitet [kWh/m2]'),
                title_opts=opts.TitleOpts(title='', pos_left="left"),
                toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
            )
            .set_series_opts()
        )
        return b1



    data = df.groupby([df['from'].dt.year, df['from'].dt.month_name(locale='da_DK'), df['from'].dt.month]).agg({'Adresse': 'first', 'amount': 'sum'})
    data = data.reset_index(level=1).rename(columns={'from':'month'}).reset_index(level=1).rename(columns={'from':'month_nr'}).reset_index().rename(columns={'from':'year'})
    data = data.sort_values(['year','month_nr'])

    data['tid'] = data['month'] + ' ' + data['year'].astype(str)
    #st.write(data)
    #st.write(data[data['year']==2022]['amount'].sum())

    #with col1:
    figur = barr(data, 30)
    st_pyecharts(figur, height='200px')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Forbrug i 2022', '{:,.0f} kWh'.format(data[data['year']==2022]['amount'].sum()) )
    col2.metric('Forbrug i sidste måned', '{:,.0f} kWh'.format(data[(data['year']==2022) & (data['month'] == 'Februar')]['amount'].sum()))
    col3.metric('Antal bygninger', df['Adresse'].nunique())
    col4.metric('Antal målere', df['meter'].nunique())


    df_besp = pd.read_csv('https://media.githubusercontent.com/media/Mikkel-schmidt/Elforbrug/main/Data/besp/' + quote(st.session_state.kunde[0]) + '.csv?token=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')
    df_besp = df_besp.sort_values(by='%', ascending=False)  
    
    #with col2:
    st.markdown('---')
    col1 , col2= st.columns([1,1])
    col1.header('Top driftsforbedringer')
    col1.dataframe(df_besp[['Adresse', 'årligt forbrug', '%']].round(1).head(10).style.background_gradient(cmap='Reds'), use_container_width=True)
    if 'df_g' in st.session_state:
        col2.header('Top standby forbedringer')
        col2.dataframe(st.session_state['df_g'].round(1).head(10).style.background_gradient(cmap='Blues'), use_container_width=True)




    #with col2:
    adr = st.selectbox('Select', df_besp['Adresse'].unique())
    dfff = df[df['Adresse']==adr].groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'bkps': 'sum'}).reset_index()
    st.write('Besparelsen er på ', str(df_besp[df_besp['Adresse']==adr]['%'].values[0].round(1)), ' %')

    @st.cache_resource()
    def linesss(df):
        b1 = (
            Line()
            .add_xaxis(list(df['from']))
            .add_yaxis('Timeforbrug', list(df['amount']), symbol='emptyCircle', symbol_size=2, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), #areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
            linestyle_opts=opts.LineStyleOpts( width=1))
            .add_yaxis('Activity', list(df['bkps']),  label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"),
            linestyle_opts=opts.LineStyleOpts( width=3),symbol='emptyCircle', symbol_size=10)
            .add_yaxis('Best', list(df['bkps'].where(dfff['bkps']==dfff['bkps'].min())),  label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"),
            linestyle_opts=opts.LineStyleOpts( width=8),symbol='emptyCircle', symbol_size=10)
            .set_global_opts(
                legend_opts=opts.LegendOpts(orient='horizontal', pos_left="center", is_show=True),
                title_opts=opts.TitleOpts(),
                toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
                yaxis_opts=opts.AxisOpts(
                    name='Forbrug [kWh]',
                    type_="value",
                    axistick_opts=opts.AxisTickOpts(is_show=True),
                    splitline_opts=opts.SplitLineOpts(is_show=True)),
                xaxis_opts=opts.AxisOpts(name='Tid'),
                datazoom_opts=[
                    opts.DataZoomOpts(range_start=0, range_end=100),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
                )
            .set_series_opts()
        )
        return b1

    #with col2:
    figur = linesss(dfff)
    st_pyecharts(figur, height='300px')
    #st.write(data)

    #st.write(df.groupby('meter').agg({'Adresse': 'first', 'amount': 'sum'}).reset_index())








