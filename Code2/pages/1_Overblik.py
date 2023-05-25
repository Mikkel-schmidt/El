import streamlit as st
import pandas as pd
import numpy as np
import ruptures as rpt
from stqdm import stqdm
from urllib.parse import quote


from streamlit_functions import select_tree, getMeterPoints, getMeterReadings
from streamlit_tree_select import tree_select

from pyecharts import options as opts
from pyecharts.charts import Bar, Pie, Grid, Line, Scatter, Sankey, WordCloud, HeatMap, Calendar, Sunburst, TreeMap
from streamlit_echarts import st_pyecharts

import matplotlib.pyplot as plt

import glob

from os import listdir
from os.path import isfile, join

import locale
#for lang in locale.windows_locale.values():
#    st.write(lang)

locale.setlocale(locale.LC_ALL, "da_DK")



st.set_page_config(layout="wide", page_title="Overblik forbrug", page_icon="https://media.licdn.com/dms/image/C4E0BAQEwX9tzA6x8dw/company-logo_200_200/0/1642666749832?e=2147483647&v=beta&t=UiNzcE1RvJD3kHI218Al7omOzPLhHXXeE_svU4DIwEM")
st.sidebar.image('https://via.ritzau.dk/data/images/00181/e7ddd001-aee3-4801-845f-38483b42ba8b.png')

df_select = st.session_state.df_select
nodes = select_tree()

st.title('Overblik over bygningsmassens forbrug')
st.markdown("""På denne side kan du se bygningernes intensitet, hvilket er et estimat af bygningernes forbrug per kvadratmeter. 
Ud fra dette vil det give en beskrivelse af hvilke bygninger der forbruger ekstra meget i forhold til deres størrelse. """)



@st.cache_data
def meters_overblik():
    df = pd.read_csv('https://media.githubusercontent.com/media/Mikkel-schmidt/Elforbrug/main/Data/timeforbrug/' + quote(st.session_state.kunde[0]) + '.csv?token=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')
    df_besp = pd.read_csv('https://media.githubusercontent.com/media/Mikkel-schmidt/Elforbrug/main/Data/besp/' + quote(st.session_state.kunde[0]) + '.csv?token=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')
    return df, df_besp

df, df_besp = meters_overblik()
df['from'] = pd.to_datetime(df['from'], utc=True)
df['meter'] = pd.to_numeric(df['meter'])
df_besp = df_besp.sort_values(by='%', ascending=False)
#st.write(df.head())
#st.write(df_besp)

col1, col2 = st.columns([3,2])
with col1:
    st.dataframe(df_besp[['Adresse', 'besparelse', 'årligt forbrug', 'mean', '%']])

with col2:
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

with col2:
    figur = linesss(dfff)
    st_pyecharts(figur, height='300px')


    # fig, ax = plt.subplots(figsize=(14,8)) 
    # ax.plot(dfff['from'], dfff['amount'], linewidth=0.3)
    # ax.plot(dfff['from'], dfff['bkps'])
    # ax.plot(dfff['from'][dfff['bkps']==dfff['bkps'].min()], dfff['bkps'][dfff['bkps']==dfff['bkps'].min()], linewidth=6)

    # st.pyplot(fig)

st.markdown("---")
col1, col2 = st.columns([1,2])
col1.subheader('Benchmark på tværs')
col1.markdown("""Det bedste overblik over bygningerne fås ved at sammenligne deres forbrug.
Ud fra dette kan det ses hvilke bygninger der er *mest energiintensive* og derfor hvilke bygninger der potentielt er noget at komme efter.""")
#df_besp['nøgletal'] = df_besp['årligt forbrug'] / df_besp['areal']
col1.write(df_besp[['Adresse', 'årligt forbrug']].sort_values('årligt forbrug', ascending=False))

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
            legend_opts=opts.LegendOpts(orient='vertical', pos_left="left", is_show=True),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=grader), name='Intensitet [kWh/m2]'),
            title_opts=opts.TitleOpts(title='Samlet forbrug', pos_left="center"),
            toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
        )
        .set_series_opts()
    )
    return b1

with col2:
    figur = barr(df_besp, 90)
    st_pyecharts(figur, height='500px')

st.markdown('---')
col1, col2 = st.columns([2,2])
abn, luk = col2.slider('Vælg bygningens åbningstider', min_value=1, max_value=24, value=(6, 18))

def get_day_moment(hour) -> str:
    if abn <= hour <= luk:
        return 'Dagsforbrug'
    return 'Standby forbrug'

df['day-moment'] = df.apply(lambda row: get_day_moment(hour = row['from'].hour), axis=1)

@st.cache_resource
def piee(df):
    hej = df[['day-moment', 'amount']].groupby('day-moment').sum()['amount'].reset_index()
    
    data = [list(z) for z in zip(hej['day-moment'], hej['amount'])]
    st.session_state['df_over_standby'] = data
    p = (
        Pie()
        .add(
            series_name='Forbrug i perioder',
            data_pair=data,
            #rosetype="area",
            radius=["40%", "70%"],
            #center=["85%", "50%"],
            label_opts=opts.LabelOpts(position="outside",
            formatter="{a|{a}}{abg|}\n{hr|}\n  {per|{d}%}  ",
            background_color="#eee",
            border_color="#aaa",
            border_width=1,
            border_radius=4,
            rich={
                "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                "abg": {
                    "backgroundColor": "#e3e3e3",
                    "width": "100%",
                    "align": "right",
                    "height": 22,
                    "borderRadius": [4, 4, 0, 0],
                },
                "hr": {
                    "borderColor": "#aaa",
                    "width": "100%",
                    "borderWidth": 0.5,
                    "height": 0,
                },
                "b": {"fontSize": 12, "lineHeight": 33},
                "per": {
                    "color": "#eee",
                    "backgroundColor": "#334455",
                    "padding": [2, 4],
                    "borderRadius": 2,
                },
            },),
            #itemstyle_opts=opts.ItemStyleOpts(color=JsCode(color_function)    )
        )
        .set_global_opts(
            legend_opts=opts.LegendOpts(orient='vertical', pos_left="right", type_='scroll', is_show=True),
            title_opts=opts.TitleOpts(
                title='Forbrug i perioder', pos_left="center"
            ),
            toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
        )
    )
    return p

@st.cache_resource
def bars(df, grader):
    df = df.sort_values('amount')
    b1 = (
        Bar()
        .add_xaxis(list(df['day-moment']))
        .add_yaxis('Gns. forbrug i timen', list(df['amount']/df['from']), label_opts=opts.LabelOpts(is_show=False, formatter="{b}: {c}"),)
        .reversal_axis()
        .set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(type_="inside", orient="vertical"), opts.DataZoomOpts(type_="slider", orient="vertical")], 
            legend_opts=opts.LegendOpts(orient='vertical', pos_left="left", is_show=True),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=grader), name='Intensitet [kWh/m2]'),
            title_opts=opts.TitleOpts(title='Gns. Forbrug pr. time i perioder', subtitle='Forbrug ml. 6 og 18 (dagsforbrug) og standbyforbrug', pos_left="center"),
            toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
        )
        .set_series_opts()
    )
    return b1

col1.subheader('Standbyanalyse')
col1.markdown("""Ved at udnytte en standbyanalyse kan man få et indblik i unødvendigt forbrug der ikke er timer på.""")
col1.markdown("""Hvis standbyforbruget er ligeså stort som forbruget indenfor åbningstid, så er det sandsynligt at f.eks. lys eller ventilation kører om natten.
Potentielt kan det også være andre ting, såsom en utæt kompressor eller andet - det vil dog først vise sig ven en besigtigelse.""")
col1.markdown('Den øverste figur til højre viser andelen af forbruget i og udenfor åbningstid. Åbningstiden kan justeres til højre. ')
col1.markdown('Den nederste figur til højre viser det gennemsnitlige forbrug i og udenfor åbningstiderne.')
col1.markdown("""I tabellen nedenunder kan du se informationer på de enkelte bygnigner om:
- Totalt dagsforbrug, standbyforbrug og det samlede forbrug
- Det gennemsnitlige forbrug i timen i og udenfor åbningstid
- Standbyforbrugets størrelse sammenlignet med dagsforbruget (vægtet) og det totale forbrug (total)
""")

@st.cache_data
def standby_df(df):
    df_g = df.groupby(['Adresse', 'day-moment']).agg({'amount': 'sum', 'from': 'count'}).reset_index()
    df_g['time gns'] = df_g.apply(lambda row: row['amount']/row['from'], axis=1)
    df_h = df_g.pivot( index='Adresse', columns=['day-moment'], values='time gns').reset_index()
    df_h = df_h.rename(columns={'Dagsforbrug': 'Time gns. dag', 'Standby forbrug': 'Time gns. standby'})

    df_g = df_g.pivot( index='Adresse', columns=['day-moment'], values='amount').reset_index()

    df_g['Totalt forbrug'] = df_g['Standby forbrug']+df_g['Dagsforbrug']
    df_g = df_g.merge(df_h, on='Adresse')

    df_g['Standby Vægtet [%]'] = df_g['Standby forbrug']/df_g['Dagsforbrug']*100
    df_g['Standby Total [%]'] = df_g['Standby forbrug']/(df_g['Standby forbrug']+df_g['Dagsforbrug'])*100
    return df_g

df_g = standby_df(df)
if 'df_g' not in st.session_state:
        st.session_state['df_g'] = df_g.sort_values(by='Standby Total [%]', ascending=False) 

col1.write(df_g.sort_values(by='Standby Total [%]', ascending=False).round(1).style.background_gradient(cmap='Blues'))


with col2:
    figur = piee(df)
    st_pyecharts(figur, height='400px')

with col2:
    figur = bars(df.groupby('day-moment').agg({'amount': 'sum', 'from': 'count'}).reset_index(), 90)
    st_pyecharts(figur, height='400px')
















#st.write(df)
#st.write(df_besp)

















