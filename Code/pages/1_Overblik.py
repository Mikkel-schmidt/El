import streamlit as st
import pandas as pd
import numpy as np
import ruptures as rpt
from stqdm import stqdm

import folium
from streamlit_folium import st_folium
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


IDs = st.session_state['IDs']

@st.cache_data
def meters_overblik(IDs):
    df = getMeterReadings(IDs)
    return df

df = meters_overblik(IDs)
#st.write(df.head(100))
st.session_state['df_overblik'] = df 
df['from'] = pd.to_datetime(df['from'], utc=True)
df['meter'] = pd.to_numeric(df['meter'])
df_orig = df
#df.dtypes
df_ignore = pd.read_csv('Code/ignore_periods.txt', sep=',', header=0, parse_dates=[1,2])
df_ignore =pd.DataFrame(df_ignore)

#st.dataframe(df_ignore)
#st.write(df_ignore['to'].iloc[0].date())
#st.write(df['from'].dt.date)
#st.write(df_ignore.dtypes)
#df_ignore['from'] = pd.to_datetime(df_ignore['from'], utc=True)
#st.write(df_ignore['from'].dt.date)

#st.write(df.head(1000))
# for i in range(len(df_ignore)):
#     df = df.loc[~((df.meter == df_ignore['meter'].iloc[i]) &
#                   (df['from'].dt.date >= df_ignore['from'].iloc[i].date()) &
#                   (df['from'].dt.date <= df_ignore['to'].iloc[i].date()))]

#st.write(df.head())
 
def get_day_moment(hour) -> str: 
    if 6 <= hour <= 18:
        return 'day'
    return 'night'
df['day-moment'] = df['from'].dt.hour.map(get_day_moment)
#st.write(df.head(1000))

def ugeprofil(df):
            dff = df.groupby([df['from'].dt.day_name(locale='da_DK'), df['from'].dt.hour]).mean().reset_index(names=['day', 'hour'])
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
            dff['x-axis'] = dff.apply(lambda row: row['day'] + ' kl. ' + str(row['hour']), axis=1)
            return dff






def rupt(tid, df):
    nbkps = 12
    #mavg = moving_avg(df.groupby('from').agg({'meter': 'mean', 'amount': 'sum'}).reset_index(), 24)
    #st.write(mavg)
    #df['day-moment'] = df['from'].dt.hour.map(get_day_moment)
    #test = df.groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()
    #test1 = test[test['day-moment']==tid]
    #st.write(test1)
    test = df['amount']
    points=np.array(test)
    n= len(points)
    if nbkps == 1:
        nbkps = np.floor(np.round(n/365*4))
        #st.write(nbkps)
    #Changepoint detection with dynamic programming search method
    model = "l1"  
    #penalty_value = 800
    algo = rpt.Window(width=24*30, model=model, min_size=3, jump=5).fit(points)
    #algo = rpt.KernelCPD(kernel="linear", min_size=3).fit(points)
    my_bkps = algo.predict(n_bkps=nbkps)
    return my_bkps


#@st.experimental_memo
def besp():
    df_besp = pd.DataFrame(columns=['Adresse', 'besparelse', 'årligt forbrug', 'last', 'best', 'mean'])
    
    for adr in stqdm(df['Adresse'].unique()):
        dff = df[df['Adresse']==adr]
        dff['day-moment'] = dff['from'].dt.hour.map(get_day_moment)
        dff = dff.groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()
        my_bkps = rupt('day', dff)

        value_avg_day = np.zeros(len(dff['amount']))
        #stdd_avg_day  = np.zeros(len(dff['amount']))
        vvvv = np.zeros((len(my_bkps),2),)
        k=0
        j=0
        for i in my_bkps:
            value_avg_day[j:i] = np.mean(dff['amount'][j:i])
            #stdd_avg_day[j:i]  = np.std(dff['amount'][j:i])
            vvvv[k, 0] = np.mean(dff['amount'][j:i])
            vvvv[k, 1] = i
            k += 1
            j=i
        dff['bkps'] = value_avg_day

        if dff['bkps'].iloc[-1] >= dff['bkps'].max():
            df_opti = dff[dff['bkps']==dff['bkps'].iloc[-1]].groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()
        else:
            df_opti = dff[dff['bkps']==dff['bkps'].min()].groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()

        df_norm = dff[dff['bkps']==dff['bkps'].iloc[-1]].groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()

        uge = ugeprofil(df_opti)
        uge2 = ugeprofil(df_norm)
        

        ugg = uge[['day', 'hour', 'amount', 'x-axis']].merge(uge2[['day', 'hour', 'amount']], how='outer', on=['day', 'hour'], suffixes=('_opti', '_now'))
        ugg['besparelse_kwh'] = ugg['amount_now'] - ugg['amount_opti']
        ttt = pd.DataFrame(data={'Adresse': [adr], 
                                 'besparelse': [ugg['besparelse_kwh'].sum()*52], 
                                 'årligt forbrug': [ugg['amount_now'].sum()*52],
                                 'last': df_norm['amount'].mean(),
                                 'best': df_opti['amount'].mean(), 
                                 'mean': dff['amount'].mean()} )
        df_besp = df_besp.append(ttt)
        
    
    df_besp = df_besp[df_besp['årligt forbrug'] != 0.0]
    df_besp['%'] = df_besp.apply(lambda row: row['besparelse']/row['årligt forbrug']*100, axis=1)
    return df_besp

col1, col2 = st.columns([3,2])
with col1:
    if 'df_besp' not in st.session_state:
        df_besp = besp().sort_values(by='%', ascending=False)
        st.session_state['df_besp'] = df_besp
    df_besp = st.session_state['df_besp']
    st.write(st.session_state['df_besp'])
 
with col2:
    adr = st.selectbox('Select', df_besp['Adresse'].unique())
    dff = df[df['Adresse']==adr].groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()
    st.write('Besparelsen er på ', str(df_besp[df_besp['Adresse']==adr]['%'].values[0].round(1)), ' %')
    
    dff['day-moment'] = dff['from'].dt.hour.map(get_day_moment)
    dff = dff.groupby('from').agg({'meter': 'mean', 'amount': 'sum', 'day-moment': 'first'}).reset_index()
    my_bkps = rupt('day', dff)

    value_avg_day = np.zeros(len(dff['amount']))
    #stdd_avg_day  = np.zeros(len(dff['amount']))
    vvvv = np.zeros((len(my_bkps),2),)
    k=0
    j=0
    for i in my_bkps:
        value_avg_day[j:i] = np.mean(dff['amount'][j:i])
        #stdd_avg_day[j:i]  = np.std(dff['amount'][j:i])
        vvvv[k, 0] = np.mean(dff['amount'][j:i])
        vvvv[k, 1] = i
        k += 1
        j=i 
    dff['bkps'] = value_avg_day
    dfff = df_orig[df_orig['Adresse']==adr].groupby('from').agg({'meter': 'mean', 'amount': 'sum'}).reset_index()
    dfff['from'] = pd.to_datetime(dfff['from'])
    fig, ax = plt.subplots(figsize=(14,8)) 
    ax.plot(dfff['from'], dfff['amount'], linewidth=0.3)
    #st.write(dff['meter'].isin(df_ignore['meter']).any())
    # if dff['meter'].isin(df_ignore['meter']).any():
        
    #     for i in range(len(df_ignore[df_ignore['meter'].isin(dff['meter'].unique())])):
    #         ax.axvspan(df_ignore['from'].iloc[i], df_ignore['to'].iloc[i], facecolor='0.2', alpha=0.1)
    
    ax.plot(dff['from'], dff['bkps'])
    ax.plot(dff['from'][dff['bkps']==dff['bkps'].min()], dff['bkps'][dff['bkps']==dff['bkps'].min()], linewidth=6)

    st.pyplot(fig)
    #st.write(dff.mean())
    #st.write(dff)
    def exclude_dates(valg1, valg2):
        df_ex = pd.DataFrame(data={'meter': dfff.meter.unique(), 'from': valg1, 'to': valg2})
        df_ex.to_csv('ignore_periods.txt', mode='a', index=False, header=False)


    valg1, valg2 = st.slider('Vælg datoer der skal ekskluderes', dfff['from'].dt.date.min(), dfff['from'].dt.date.max(), (dfff['from'].dt.date.min(), (dfff['from'].dt.date.min() + pd.Timedelta(1, unit='d'))))
    #st.button('Ekskluder datoer fra analysen', on_click=exclude_dates(valg1, valg2))

    st.button('Ekskluder datoer fra analysen', on_click=exclude_dates(valg1, valg2))













st.markdown('---')
dff = df.groupby('Adresse').sum('amount').reset_index()
col1, col2 = st.columns([1,2])

@st.cache_resource
def barr(df, grader):
    df = df.sort_values('amount')
    b1 = (
        Bar()
        .add_xaxis(list(df['Adresse']))
        .add_yaxis('Samlet forbrug', list(df['amount']), label_opts=opts.LabelOpts(is_show=False, formatter="{b}: {c}"),)
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
    figur = barr(dff, 90)
    st_pyecharts(figur, height='500px')

col1.subheader('Benchmark på tværs')
col1.markdown("""Det bedste overblik over bygningerne fås ved at sammenligne deres forbrug.
Ud fra dette kan det ses hvilke bygninger der er *mest energiintensive* og derfor hvilke bygninger der potentielt er noget at komme efter.""")
col1.write(dff.sort_values('amount', ascending=False))

st.markdown('---')








col1, col2 = st.columns([2,2])
abn, luk = col2.slider('Vælg bygningens åbningstider', min_value=1, max_value=24, value=(6, 18))


def get_day_moment(hour) -> str:
    if abn <= hour <= luk:
        return 'Dagsforbrug'
    return 'Standby forbrug'

df['day-moment'] = df.apply(lambda row: get_day_moment(hour = row['from'].hour), axis=1)

def piee(df):
    hej = df.groupby('day-moment').sum()['amount'].reset_index()
    
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

df_g = df.groupby(['Adresse', 'day-moment']).agg({'amount': 'sum', 'from': 'count'}).reset_index()
df_g['time gns'] = df_g.apply(lambda row: row['amount']/row['from'], axis=1)
df_h = df_g.pivot( index='Adresse', columns=['day-moment'], values='time gns').reset_index()
df_h = df_h.rename(columns={'Dagsforbrug': 'Time gns. dag', 'Standby forbrug': 'Time gns. standby'})
df_g = df_g.pivot( index='Adresse', columns=['day-moment'], values='amount').reset_index()
df_g['Totalt forbrug'] = df_g['Standby forbrug']+df_g['Dagsforbrug']
df_g = df_g.merge(df_h, on='Adresse')
df_g['Standby Vægtet [%]'] = df_g['Standby forbrug']/df_g['Dagsforbrug']*100
df_g['Standby Total [%]'] = df_g['Standby forbrug']/(df_g['Standby forbrug']+df_g['Dagsforbrug'])*100

col1.write(df_g.style.background_gradient(cmap='Blues').set_precision(1))
#col1.write(dff) 

with col2:
    figur = piee(df)
    st_pyecharts(figur, height='400px')

with col2:
    figur = bars(df.groupby('day-moment').agg({'amount': 'sum', 'from': 'count'}).reset_index(), 90)
    st_pyecharts(figur, height='400px')

st.markdown('---')














