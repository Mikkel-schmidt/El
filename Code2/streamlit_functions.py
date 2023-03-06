from streamlit_tree_select import tree_select
import streamlit as st
import pandas as pd
import requests
import urllib.request as request
from urllib.parse import quote

from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from stqdm import stqdm


def getMeterPoints(Customer):
    ApiKey = "3bb0bb5c-392d-4daa-8b0e-6a675387d08b"
    url = "https://superhub.dk/api/nrgiraadgivning/v2/meteringPoints"
    url = url + "?" + "apikey=" + ApiKey + "&customerKey=" + quote(Customer)
    #print(url)
    response = requests.get(url)
    df = pd.json_normalize(response.json()) 
    df['meteringPointId'] = pd.to_numeric(df['meteringPointId'])
    IDs = list(df['meteringPointId'])
    return df, IDs

def getMeterReadings(meteringPoints):
    ApiKey = "3bb0bb5c-392d-4daa-8b0e-6a675387d08b"
    From = '31122021'#(datetime.today() - timedelta(days = 1000)).strftime('%d%m%Y')
    To = datetime.today().strftime('%d%m%Y')
    df = pd.DataFrame()
    Fails = 0
    for meter in stqdm(meteringPoints):
        url = "https://superhub.dk/api/nrgiraadgivning/v2/meterreadings"
        url = url + "?" + "apikey=" + ApiKey + "&meteringpointId=" + str(meter) + "&from=" + From + "&to=" + To
        #print(url)
        response = requests.get(url)
        if response.status_code != 200:
            Fails += 1
            continue
        jso = response.json()

        dff = pd.json_normalize(jso)
        
        if (dff.columns == 'meteringPoints.Production').any():
            df_meter = pd.json_normalize(jso['meteringPoints'], 'Production')
        else:
            df_meter = pd.json_normalize(jso['meteringPoints'], 'Consumption') 
        df_meter['meter'] = jso['meteringPointId']
        df_meter['Adresse'] = jso['streetName'] + ' ' + jso['buildingNumber'] + ', ' + jso['postcode'] + ' ' + jso['cityName']
        df = pd.concat([df, df_meter], ignore_index=True)
    print('Amount of fails: ' + str(Fails))
    return df

def select_tree():

    # initializing key list 
    key_list = ["label", "value", 'children']
    key_list1 = ['label', 'value']

    df_select = st.session_state.df_select
    res = []
    for i in df_select['Adresse'].unique():
        ress = []
        for j in pd.to_numeric(df_select['meter'][df_select['Adresse']==i].unique()):
            ress.append({key_list1[0]: '⚡️' + str(j), key_list1[1] : str(j)})
        res.append({key_list[0]: i, key_list[1] : i, key_list[2]: ress})
    nodes = res
    
    with st.sidebar:
        if 'valgt_meter' not in st.session_state:
            return_select = tree_select(nodes)
        else:
            return_select = tree_select(nodes, checked=list(st.session_state.valgt_meter))
            #st.write(return_select)
    st.session_state.valgt_meter = pd.json_normalize(return_select)['checked'][0]
    return res

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(http://placekitten.com/200/200);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True