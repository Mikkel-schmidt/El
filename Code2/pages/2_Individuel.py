import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import quote

from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from stqdm import stqdm
import ruptures as rpt


from streamlit_functions import select_tree, getMeterPoints, getMeterReadings
from streamlit_tree_select import tree_select
import streamlit.components.v1 as components

import matplotlib.pyplot as plt

from pyecharts import options as opts
from pyecharts.charts import Bar, Pie, Grid, Line, Scatter, Sankey, WordCloud, HeatMap, Calendar, Sunburst, TreeMap
from streamlit_echarts import st_pyecharts

import locale
#for lang in locale.windows_locale.values():
#    st.write(lang)

locale.setlocale(locale.LC_ALL, "da_DK")

from os import listdir
from os.path import isfile, join

st.set_page_config(layout="wide", page_title="Individuel forbrug", page_icon="https://media.licdn.com/dms/image/C4E0BAQEwX9tzA6x8dw/company-logo_200_200/0/1642666749832?e=2147483647&v=beta&t=UiNzcE1RvJD3kHI218Al7omOzPLhHXXeE_svU4DIwEM")
st.sidebar.image('https://via.ritzau.dk/data/images/00181/e7ddd001-aee3-4801-845f-38483b42ba8b.png')
nodes = select_tree()
#st.write(nodes)
if not st.session_state.valgt_meter:
    st.warning('Vær sød at vælge en adresse ude i siden') 
    st.stop()
IDs = list(st.session_state.valgt_meter)
#st.write(IDs)

@st.cache_data
def meters_indi(): 
    df = pd.read_csv('https://github.com/Mikkel-schmidt/Elforbrug/raw/main/Data/timeforbrug/' + st.session_state.kunde[0] + '.csv')
    #df = df[df['meter'].isin(list(st.session_state.valgt_meter))]
    return df

df = meters_indi() 

df['meter'] = pd.to_numeric(df['meter'])
df = df[df['Adresse'].isin(IDs)]
df['from'] = pd.to_datetime(df['from'], utc=True)
df['ugedag'] = df['from'].dt.day_name(locale='da_DK')

#st.write(df['meter'].unique().isin(list(st.session_state.valgt_meter)))


col1, col2 = st.columns([1,4])

col1.header('Individuelle målere')
col1.markdown('På denne side er der analyser der beskriver de enkelte målere og giver et indblik i forskellige målere.')
col1.markdown("""Til højre kan man se timeforbruget fra målerne på bygningen. 
Ud fra dette har er der lavet et løbende gennemsnit over de sidste 24 timer.""")
col1.markdown("""
Change point detection viser forskellige perioder hvor bygningen er blevet driftet på forskellige måder. 
Det finder algoritmen ud fra ved at kigge på gennemsnittet, variationen i forbruget, peaks i forbruget og andre atypiske mønstre i forbruget.
""")
col1.markdown('Perioden kan ændres ved at trække i baren i bunden af figuren.')

@st.cache_resource()
def linesss(df):
    b1 = (
        Line()
        .add_xaxis(list(df['from']))
        .add_yaxis('Timeforbrug', list(df['amount']), symbol='emptyCircle', symbol_size=2, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), #areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts( width=1), areastyle_opts=opts.AreaStyleOpts(opacity=0.3),)
        .add_yaxis('Activity', list(df['bkps']),  label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"),
        linestyle_opts=opts.LineStyleOpts( width=3),symbol='emptyCircle', symbol_size=10)
        .add_yaxis('Best', list(df['bkps'].where(df['bkps']==df['bkps'].min())),  label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"),
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
    figur = linesss(df.groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'bkps': 'sum'}).reset_index())
    st_pyecharts(figur, height='600px')

#@st.cache_resource
def heatmapp(df):
    pivot = df.pivot_table(index=df['from'].dt.day_name(locale='da_DK'), columns=df['from'].dt.hour, aggfunc='mean', values='amount', sort=False)
    pivot['day'] = pivot.index
    pivot['day'].replace({
            "Mandag": 0,
            "Tirsdag": 1,
            "Onsdag": 2,
            "Torsdag": 3,
            "Fredag": 4,
            "Lørdag": 5,
            "Søndag": 6},
            inplace=True,)
    pivot.sort_values('day', ascending=False, inplace=True)
    pivot.drop('day', axis=1)

    x_axis = pivot.columns[:-1].tolist()
    y_axis = pivot.index.tolist()
    data = [[i, j, pivot.iloc[j,i].round(1)] for i in range(24) for j in range(7)]

    b1 = (
        HeatMap()
        .add_xaxis(x_axis)
        .add_yaxis("Forbrug [kWh]", y_axis, list(data), label_opts=opts.LabelOpts(is_show=True, position="inside"))
        .set_global_opts(
            legend_opts=opts.LegendOpts(orient='vertical', pos_left="center", is_show=False),
            visualmap_opts=opts.VisualMapOpts(min_=0, max_=pivot.max().max(), is_calculable=True, orient="horizontal", pos_left="center"),
            title_opts=opts.TitleOpts(
            ),
            toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
        )
        .set_series_opts(
        )
    )
    return b1

col1, col2 = st.columns([4,1])
col2.subheader('Heatmap og ugeprofil')
col2.markdown("""I figuren til venstre kan man se hvordan forbruget fordeler sig henover timerne på en uge.
Her burde man kunne se hvordan åbningstiderne i bygningen er og om bygningen lukkes ned udenfor åbningstid""")
col2.markdown('Tallet i midten er det gennemsnitlige forbrug i den time på den ugedag.')

if df['bkps'].iloc[-1] >= df['bkps'].max():
    df_opti = df[df['bkps']==df['bkps'].iloc[-1]].groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()
else:
    df_opti = df[df['bkps']==df['bkps'].min()].groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()

df_norm = df[df['bkps']==df['bkps'].iloc[-1]].groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()

with col1:
    figure = heatmapp(df.iloc[-2159:])
    st_pyecharts(figure, height='400px', key='hej')

#@st.cache_data
def ugeprofil(df):
    dff = df.groupby([df['from'].dt.day_name(locale='da_DK'), df['from'].dt.hour]).agg({'amount': ['mean', 'std']}).reset_index(names=['day', 'hour'])
    dff.columns = ['_'.join(tup).rstrip('_') for tup in dff.columns.values]
    #st.write(dff)
    dff['day_'] = dff['day']
    dff['day_'].replace({
            "Mandag": 0,
            "Tirsdag": 1,
            "Onsdag": 2,
            "Torsdag": 3,
            "Fredag": 4,
            "Lørdag": 5,
            "Søndag": 6},
            inplace=True,)
    dff.sort_values(['day_', 'hour'], ascending=True, inplace=True)
    #st.write(dff)
    dff['x-axis'] = dff.apply(lambda row: row['day'] + ' kl. ' + str(row['hour']), axis=1)
    return dff



uge = ugeprofil(df_opti)
uge2 = ugeprofil(df_norm)

#st.write(uge2)

#st.write(ug)
#st.write(ug2)
ugg = uge[['day', 'hour', 'amount_mean', 'x-axis']].merge(uge2[['day', 'hour', 'amount_mean']], how='outer', on=['day', 'hour'], suffixes=('_opti', '_now'))
ugg['besparelse_kwh'] = ugg['amount_mean_now'] - ugg['amount_mean_opti']
st.write('Mulig besparelse på ' + str(ugg['besparelse_kwh'].sum()*52) + ' kWh')
st.write('Årlig forbrug på ' + str(ugg['amount_mean_now'].sum()*52) + ' kWh')
st.write('Mulig besparelse på ' + str((ugg['besparelse_kwh'].sum()*52)/(ugg['amount_mean_now'].sum()*52)*100) + '%')


 
@st.cache_resource
def liness(df, df2):
    b1 = (
        Line()
        .add_xaxis(list(df['x-axis']))
        .add_yaxis('Timeforbrug', list(df['amount_mean']), symbol='emptyCircle', symbol_size=0, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), #areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts(width=2), 
        )
        .add_yaxis('Optimeret', list(df2['amount_mean']), symbol='emptyCircle', symbol_size=1, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), #areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts(width=2), 
        )
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
    )
    return b1

with col1:
    figur = liness(uge2, uge)
    st_pyecharts(figur, height='400px')

#st.write(list(uge2['hour'].unique()))
#st.write(list(uge2[uge2['day']=='Mandag']['amount_mean']))

#@st.cache_resource
def liness(df):
    b1 = (
        Line()
        .add_xaxis(list(df['hour'].unique().astype(str)))
        .add_yaxis('Mandag', list(df[df['day']=='Mandag']['amount_mean']), symbol='emptyCircle', symbol_size=0, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), color="#734848", #areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts(width=2), )
        .add_yaxis('Tirsdag', list(df[df['day']=='Tirsdag']['amount_mean']), symbol='emptyCircle', symbol_size=0, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), color="#c17150",#areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts(width=2), )
        .add_yaxis('Onsdag', list(df[df['day']=='Onsdag']['amount_mean']), symbol='emptyCircle', symbol_size=0, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), color="#c8f0e6",#areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts(width=2), )
        .add_yaxis('Torsdag', list(df[df['day']=='Torsdag']['amount_mean']), symbol='emptyCircle', symbol_size=0, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), color="#a1d9cc",#areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts(width=2), )
        .add_yaxis('Fredag', list(df[df['day']=='Fredag']['amount_mean']), symbol='emptyCircle', symbol_size=0, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), color="#1e8c82",#areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts(width=2), )
        .add_yaxis('Lørdag', list(df[df['day']=='Lørdag']['amount_mean']), symbol='emptyCircle', symbol_size=0, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), color="#006e64",#areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts(width=2), )
        .add_yaxis('Søndag', list(df[df['day']=='Søndag']['amount_mean']), symbol='emptyCircle', symbol_size=0, label_opts=opts.LabelOpts(is_show=False,formatter="{b}: {c}"), color="#004e4a", #areastyle_opts=opts.AreaStyleOpts(opacity=0.5,),# color="#546a67"),
        linestyle_opts=opts.LineStyleOpts(width=2), )
        .set_global_opts(
            legend_opts=opts.LegendOpts( pos_left="center", is_show=True),
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
    )
    return b1

with col1:
    figur = liness(uge2)
    st_pyecharts(figur, height='400px')


st.markdown('---')
col1, col2 = st.columns([3,1])
abn, luk = col2.slider('Vælg bygningens åbningstider', min_value=1, max_value=24, value=(6, 18))
col1.subheader('Standby forbrug')
col1.markdown("""Nedenunder kan man se hvor meget af forbruget der ligger udenfor åbningstiderne, som der kan defineres til højre.
I figuren til venstre kan man se hvor meget af det totale forbrug ligger i og udenfor åbningstid. I figuren til højre er det gennemsnitlige timeforbrug. """)
col1.markdown('Den inderste cirkel i figuren til venstre viser bygningsmassens fordeling til sammenligning.')

#@st.cache_resource
def piee(df):
    hej = df.groupby('day-moment').sum()['amount'].reset_index()
    data = [list(z) for z in zip(hej['day-moment'], hej['amount'])]
    #st.write(data)
    # if not st.session_state['df_over_standby']:
    p = (
        Pie()
        .add(
            series_name='Forbrug i perioder',
            data_pair=data,
            #rosetype="area",
            radius=["40%", "70%"],
            #center=["85%", "50%"],
            label_opts=opts.LabelOpts(position="outside",
            formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c} kWh\n {per|{d}%}  ",
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
    # else:
    #     data2 = st.session_state['df_over_standby']
    #     p = (
    #         Pie()
    #         .add(
    #             series_name=df['Adresse'],
    #             data_pair=data,
    #             #rosetype="area",
    #             radius=["40%", "70%"],
    #             #center=["85%", "50%"],
    #             label_opts=opts.LabelOpts(position="outside",
    #             formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c} kWh\n {per|{d}%}  ",
    #             background_color="#eee",
    #             border_color="#aaa",
    #             border_width=1,
    #             border_radius=4,
    #             rich={
    #                 "a": {"color": "#999", "lineHeight": 22, "align": "center"},
    #                 "abg": {
    #                     "backgroundColor": "#e3e3e3",
    #                     "width": "100%",
    #                     "align": "right",
    #                     "height": 22,
    #                     "borderRadius": [4, 4, 0, 0],
    #                 },
    #                 "hr": {
    #                     "borderColor": "#aaa",
    #                     "width": "100%",
    #                     "borderWidth": 0.5,
    #                     "height": 0,
    #                 },
    #                 "b": {"fontSize": 12, "lineHeight": 33},
    #                 "per": {
    #                     "color": "#eee",
    #                     "backgroundColor": "#334455",
    #                     "padding": [2, 4],
    #                     "borderRadius": 2,
    #                 },
    #             },),
    #             #itemstyle_opts=opts.ItemStyleOpts(color=JsCode(color_function)    )
    #         )
    #         .add(
    #             series_name='Bygningsmassens forbrug',
    #             data_pair=data2,
    #             #rosetype="area",
    #             radius=["20%", "30%"],
    #             #center=["85%", "50%"],
    #             #label_opts=opts.LabelOpts(position="inside",)
    #         )
    #         .set_global_opts(
    #             legend_opts=opts.LegendOpts(orient='vertical', pos_left="right", type_='scroll', is_show=True),
    #             title_opts=opts.TitleOpts(
    #                 title='Forbrug i perioder', pos_left="center"
    #             ),
    #             toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
    #         )
    #     )

    return p

#st.write(df.groupby('day-moment').agg({'amount': 'sum', 'from': 'count'}).reset_index())
def bars(df, grader):
    df = df.sort_values('amount')
    b1 = (
        Bar()
        .add_xaxis(list(df['day-moment']))
        .add_yaxis('Intensitet per kvadratmeter', list(df['amount']), label_opts=opts.LabelOpts(is_show=False, formatter="{b}: {c}"),)
        .reversal_axis()
        .set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(type_="inside", orient="vertical"), opts.DataZoomOpts(type_="slider", orient="vertical")], 
            legend_opts=opts.LegendOpts(orient='vertical', pos_left="right", is_show=True),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=grader), name='Intensitet [kWh/m2]'),
            title_opts=opts.TitleOpts(title='Forbrug mellem 6 og 18 og standby', pos_left="center"),
            toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
        )
        .set_series_opts()
    )
    return b1

col1, col2 = st.columns([2,3])

# with col1:
#     figur = piee(df)
#     st_pyecharts(figur, height='400px')

figur = piee(df)
st_pyecharts(figur, height='400px')

with col2:
    figur = bars(df.groupby('day-moment').sum()['amount'].reset_index(), 90)
    st_pyecharts(figur, height='400px')


st.markdown('---')

col1, col2 = st.columns([1,1])
col1.subheader('Kalender oversigt')
col1.markdown("""I den nedenstående grafik kan man se hvordan forbruget fordeler sig henover året. 
""")


maxx = df.groupby(df['from'].dt.date).sum()['amount'].max()
minn = df.groupby(df['from'].dt.date).sum()['amount'].min()
aar = col2.multiselect('Vælg år', df['from'].dt.year.unique(), default=2022)

 
#@st.cache_resource
def Calendarr(df, maxx, aar):
    data = [[df.loc[i, 'from'], df.loc[i, 'amount']] for i in range(len(df['from']))]
    b1 = (
        Calendar()
        .add(series_name=str(aar),
    yaxis_data=data,
    calendar_opts=opts.CalendarOpts(
    pos_left="30" ,
    pos_right="30",
    range_=str(aar), 
    yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
    daylabel_opts=opts.CalendarDayLabelOpts(name_map="en"),
    #monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="en"),
    ))
        .set_global_opts(
            legend_opts=opts.LegendOpts(orient='vertical', pos_left="center", is_show=True),
            title_opts=opts.TitleOpts(
            ),
            toolbox_opts=opts.ToolboxOpts(orient='vertical', is_show=False),
            visualmap_opts=opts.VisualMapOpts(
            max_= maxx, min_=minn, orient="horizontal", is_piecewise=False 
            ),
        )
        .set_series_opts(
        )
    ) 
    return b1
#st.write(df)
#st.write(df.groupby('from').sum().reset_index())
for i in aar:
    figure = Calendarr(df.groupby(df['from'].dt.date).sum()['amount'].reset_index(), maxx, i)
    st_pyecharts(figure, height='400px', key='hejsa'+str(i))






