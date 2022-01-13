import streamlit as st
import pandas as pd
import re
import copy
import numpy as np
import hashlib
import requests
import json
from itertools import chain
import base64
import io
import xlsxwriter
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
# import pymongo
# from pymongo import MongoClient
from datetime import datetime
from google.cloud import firestore
# import boto
# import boto.s3
import sys
# from boto.s3.key import Key
# from github import Github
# import dropbox
import os
# import pymongo
# from pymongo import MongoClient
# client = MongoClient("mongodb+srv://test:test@bram.nybip.mongodb.net/test")
# database
# db = client["Nokia_database"]
# collection
# company = db["LTE"]


# import boto3
# from botocore.exceptions import NoCredentialsError
pd.options.display.precision = 1
# from streamlit_disqus import st_disqus


# def navigation():
#     try:
#         path = st.experimental_get_query_params()['p'][0]
#     except Exception as e:
#         st.error('Please use the main app.')
#         return None
#     return path


# if navigation() == "home":
#     st.title('Home')
#     st.write('This is the home page.')

# elif navigation() == "results":
#     st.title('Results List')
#     for item in range(25):
#         st.write(f'Results {item}')

# elif navigation() == "analysis":
#     st.title('Analysis')
#     x, y = st.number_input('Input X'), st.number_input('Input Y')
#     st.write('Result: ' + str(x+y))

# elif navigation() == "examples":
#     st.title('Examples Menu')
#     st.write('Select an example.')


# elif navigation() == "logs":
#     st.title('View all of the logs')
#     st.write('Here you may view all of the logs.')


# elif navigation() == "verify":
#     st.title('Data verification is started...')
#     st.write('Please stand by....')


# elif navigation() == "config":
#     st.title('Configuration of the app.')
#     st.write('Here you can configure the application')

class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.radio(
            'Go To',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()


# new_app1.py


def foo():
    st.write("Hello Foo")


def bar():
    st.write("Hello Bar")


app = MultiApp()
app.add_app("Foo", foo)
app.add_app("Bar", bar)
# app.run()


# st_disqus("streamlit-disqus-demo")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

st.header('Nokia Health Check')
st.markdown('Please upload  **only excel files**.')
# st.subheader('Cool App')


# def create_git_file(content, filename):
# Write a file
# github = Github('')
# repository = github.get_user().get_repo('IRAN-React')  # path in the repository
# create with commit message
# content = '{\"name\":\"beppe\",\"city\":\"amsterdam\"}'
# f = repository.create_file(filename, "create_file via PyGithub", content)

# # read a file
# github = Github('personal_access_token)
# repository = github.get_user().get_repo('my_repo')  # path in the repository
# filename = 'files/file.json'
# file = repository.get_contents(filename)
# print(file.decoded_content.decode())


# client = dropbox.client.DropboxClient( < auth_token > )
# print 'linked account: ', client.account_info()

# f = open('working-draft.txt', 'rb')
# response = client.put_file('/magnum-opus.txt', f)
# print 'uploaded: ', response

# folder_metadata = client.metadata('/')
# print 'metadata: ', folder_metadata

# f, metadata = client.get_file_and_metadata('/magnum-opus.txt')
# out = open('magnum-opus.txt', 'wb')
# out.write(f.read())
# out.close()
# print metadata
# github username
# username = "BramheshV"
# url to request
# url = f"https://api.github.com/users/{username}"
# make the request and return the json
# user_data = requests.get(url).json()
# pretty print JSON data
# st.sidebar.write(user_data)

# Get the current working
# directory (CWD)
cwd = os.getcwd()
db = firestore.Client.from_service_account_json("firestore-key.json")
# import json
# key_dict = json.loads(st.secrets["textkey"])
# st.sidebar.write(key_dict)
# creds = service_account.Credentials.from_service_account_info(key_dict)
# db = firestore.Client(credentials=creds, project="streamlit-reddit")

# Print the current working
# directory (CWD)
# st.sidebar.write("Current working directory:", cwd)

# Get the list of all files and directories
# in the root directory
# dir_list = os.listdir(cwd)

# st.sidebar.write("Files and directories in '", cwd, "' :")

# print the list
# st.sidebar.write(dir_list)

# fd = "GFG.txt"

# popen() is similar to open()
# file = open(fd, 'w')
# file.write("Hello")
# file.close()
# file = open(fd, 'r')
# text = file.read()
# st.sidebar.write(text)


# directory = r"c:\temp\uploads"
# data = {'grant_type':"client_credentials",
#         'resource':"https://graph.microsoft.com",
#         'client_id':'XXXXX',
#         'client_secret':'XXXXX'}
# URL = "https://login.windows.net/YOURTENANTDOMAINNAME/oauth2/token?api-version=1.0"
# r = requests.post(url = URL, data = data)
# j = json.loads(r.text)
# TOKEN = j["access_token"]
# URL = "https://graph.microsoft.com/v1.0/users/YOURONEDRIVEUSERNAME/drive/root:/fotos/HouseHistory"
# headers={'Authorization': "Bearer " + TOKEN}
# r = requests.get(URL, headers=headers)
# j = json.loads(r.text)
# print("Uploading file(s) to "+URL)
# for root, dirs, files in os.walk(directory):
#     for filename in files:
#         filepath = os.path.join(root,filename)
#         print("Uploading "+filename+"....")
#         fileHandle = open(filepath, 'rb')
#         r = requests.put(URL+"/"+filename+":/content", data=fileHandle, headers=headers)
#         fileHandle.close()
#         if r.status_code == 200 or r.status_code == 201:
#             #remove folder contents
#             print("succeeded, removing original file...")
#             os.remove(os.path.join(root, filename))
# print("Script completed")
# raise SystemExit


# ACCESS_KEY = ''
# SECRET_KEY = ''


# def upload_to_aws(local_file, bucket, s3_file):
#     s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
#                       aws_secret_access_key=SECRET_KEY)

#     try:
#         s3.upload_file(local_file, bucket, s3_file)
#         print("Upload Successful")
#         return True
#     except FileNotFoundError:
#         print("The file was not found")
#         return False
#     except NoCredentialsError:
#         print("Credentials not available")
#         return False


# uploaded = upload_to_aws('local_file', 'bucket_name', 's3_file_name')


def color_negative_red(value):

    if value > 3:
        color = 'red'
    else:
        color = 'lightgreen'

    return 'background-color: %s' % color


def value_loc(value, df):
    for col in list(df):
        if value in df[col].values.astype(str):
            return (list(df).index(col), df[col][df[col] == value].index[0])


def process_vswr(uploaded_vswr_file):
    if 'vswr' not in st.session_state:
        st.session_state['vswr'] = 1.4
    if uploaded_vswr_file is not None:
        # To read file as bytes:
        #  bytes_data = uploaded_file.getvalue()
        #  st.write(bytes_data)# To convert to a string based IO:
        #  stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        #  st.write(stringio)# To read file as string:
        #  string_data = stringio.read()
        #  st.write(string_data)# Can be used wherever a "file-like" object is accepted:
        # excel_read = pd.read_csv(uploaded_file)
        # uploadedfn = uploaded_vswr_file.name
        # siteid = uploadedfn.split('_')[2][1:]
        # st.sidebar.write(f"Site Id: :point_right: *{siteid}*")
        # excel_read = pd.read_csv(
        # uploaded_vswr_file, skiprows=1, header=None)
        excel_read = uploaded_vswr_file
        myfiler = excel_read

        myfile1 = excel_read[excel_read.isin(['VSWR']).any(axis=1)].index
        print(excel_read.head())

        print(f"myfile1 is.. {len(myfile1)}")
        if len(myfile1) > 0:
            # print(myfile1)
            # print(hashfunc('-'))
            print(f"one--------{len(myfile1)}")
            indexval = myfile1.values[0]
            print(indexval)
            vswr = excel_read.drop(excel_read.index[0:indexval+1])
            myfilevswr = vswr
            # print(rtwp)
        # print(myfile1)
        # print(hashfunc('-'))

        # print(f"vswr file is..{vswr}")
        else:
            vswr = excel_read
            myfilevswr = vswr

        myfile4 = vswr[vswr.isin(['RTWP LTE']).any(axis=1)].index
        # print(myfile2)
        if len(myfile4) > 0:
            indexval2 = myfile4.values[0]
            print(f"RTWP LTE index ..{indexval2}")
            vswr2 = vswr.drop(excel_read.index[indexval2:])
        #     print(f"vswr2..is {vswr}")
            print(vswr2.shape)
            list(vswr2.columns)
            myfiler_vswr = vswr2
            print(f"vswr clean file.. {myfiler_vswr}")
        else:
            myfiler_vswr = vswr

        # myfiler = myfile.replace({'-': '0'}, regex=True)
        myfilervswr = myfiler_vswr.applymap(
            lambda x: '0' if (x == '-') else x)

        # myfiler.applymap(lambda x: x**2)
        # myfile = myfile[~myfile.eq('-').any(1)]

        myfiler = myfilervswr
        # st.sidebar.table(myfiler)

        filt = myfiler.iloc[:, 0].str.contains(
            'Radio module') & myfiler.iloc[:, 2].str.contains('Supported TX bands')
        mylist = (myfiler.loc[filt]).values.flatten().tolist()
        newlist = list(map(str, mylist))
        print(type(newlist))
        newlistt = [x for x in newlist if 'nan' not in x.lower()]
        lennewlist = len(newlist)
        lennewlistt = len(newlistt)
        N = lennewlist - lennewlistt
        if N > 0:
            myfiler = myfiler.iloc[:, :-N]
        myfiler.columns = newlistt

        starttime = newlistt[3]
        endtime = newlistt[-1]

        myfiler.reset_index(drop=True)

        minusfilt = ~myfiler.eq('Radio module').any(
            1) & ~myfiler.eq('Supported TX bands').any(1)
        myfiler = myfiler[minusfilt]

        myfiler.dropna()
        # print(myfilevswr.columns.values.tolist())
        radiofile = myfiler.set_index('Radio module')
        radioofile = radiofile.dropna()
        print(f"VSWR data....")
        # radioofile.head(100)
        my_list = radioofile.columns.values.tolist()
        print(f"List of columns.{my_list}")
        print(f"Index column is.. {radioofile.index.name}")

        radioofile['combined'] = radioofile.apply(
            lambda x: '%s_%s' % (x.name, x['Antenna/Port']), axis=1)

        grouped = radioofile.groupby(['combined'])

        rxlist = radioofile['combined'].unique().tolist()
        print(f"List of unique combinations.. {rxlist}")

        greeks = ['Alpha', 'Beta', 'Gamma', 'Delta']

        num_of_columns = radioofile.shape[1]
        no_of_readings = (num_of_columns - 3)
        double_no_of_readings = 2*no_of_readings
        print(f"No. of readings..{no_of_readings}")
        rmodlist = []
        avg_ant1 = []
        avg_ant2 = []
        avg_ant3 = []
        avg_ant4 = []
        rmod_flag = ""
        afhig_ahloa_flag = ""
        current_greek = "Alpha"
        readingslist = []
        ant1_percent = []
        ant2_percent = []
        ant3_percent = []
        ant4_percent = []
        sector_radio = []

        def hashfunc(s):
            return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

        def lastpercent(s):
            if type(s) is str and len(s) > 2:
                return s[-2] == '%'
            else:
                return 0

        def lastpercent_yellow(s):
            if type(s) is str and len(s) > 2:
                return s[-3] == '%'
            else:
                return 0

        def bg_color(v):
            if (lastpercent(v) and hashfunc(v) != 62282978):
                return "red"
            elif (lastpercent_yellow(v) and hashfunc(v) != 62282978):
                return "yellow"
            else:
                return "lightgreen"

        def ant_count(antenna_num, count):
            if (count/double_no_of_readings) > 0.49:
                if(int(antenna_num) == 1):
                    ant1_percent.append(
                        " ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                if(int(antenna_num) == 2):
                    ant2_percent.append(
                        " ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                if(int(antenna_num) == 3):
                    ant3_percent.append(
                        " ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                if(int(antenna_num) == 4):
                    ant4_percent.append(
                        " ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
            elif (count/double_no_of_readings) < 0.49 and (count/double_no_of_readings) > 0:
                if(int(antenna_num) == 1):
                    ant1_percent.append(
                        "  ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                if(int(antenna_num) == 2):
                    ant2_percent.append(
                        "  ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                if(int(antenna_num) == 3):
                    ant3_percent.append(
                        "  ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                if(int(antenna_num) == 4):
                    ant4_percent.append(
                        "  ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
            else:
                if(int(antenna_num) == 1):
                    ant1_percent.append(
                        f"{count}|{np.around((count/double_no_of_readings)*100,0)}%")
                elif(int(antenna_num) == 2):
                    ant2_percent.append(
                        f"{count}|{np.around((count/double_no_of_readings)*100,0)}%")
                elif(int(antenna_num) == 3):
                    ant3_percent.append(
                        f"{count}|{np.around((count/double_no_of_readings)*100,0)}%")
                elif(int(antenna_num) == 4):
                    ant4_percent.append(
                        f"{count}|{np.around((count/double_no_of_readings)*100,0)}%")

        limitdb = np.around(np.around(st.session_state.vswr, 2), 2) - 0.01
        cleanrxlist = [x for x in rxlist if str(x) != 'nan']
        print(f"cleanrxlist..{cleanrxlist}")
        for i in cleanrxlist:

            dff = grouped.get_group(i)
            current_rmod = f"{dff['combined'].tolist()[0].split('_')[0]}{dff['combined'].tolist()[0].split('_')[1]}"
            current_afhig_ahloa = current_rmod.split('(')[1][:-1]
            print(f"current afhig_ahloa is..{current_afhig_ahloa}")
            print(f"current_rmod is .. {current_rmod}")
            print(f"rmod_flag is .. {rmod_flag}")
            print(f"afhig_ahloa_flag is..{afhig_ahloa_flag}")

            if rmod_flag != current_rmod:
                rmodlist.append(current_rmod)
                readingslist.append(str(double_no_of_readings))
                if afhig_ahloa_flag != current_afhig_ahloa:
                    current_greek = "Alpha"
                else:
                    current_greek = greeks[greeks.index(
                        f"{current_greek}") + 1]
                sector_radio.append(
                    f"{current_greek}-{current_afhig_ahloa}")

            print(f"current_greek is ..{current_greek}")
            print(f"rmodlist inside the for loop is..{rmodlist}")
            rmod_flag = current_rmod
            afhig_ahloa_flag = current_afhig_ahloa
            combined_value = dff['combined'].iloc[0]
            print(f"combined value is.. {combined_value}")
            antenna_num = combined_value.split('_')[2][-1]
        #     RMOD-1/RMOD_R-1(AHFIG)_ANT1
            print(f"dff[combined]is .. {dff['combined'].iloc[0]}")
            print(f"current rmod is. {current_rmod}")
            dff = grouped.get_group(i)
            dff['Avg'] = np.around(
                dff.iloc[:, 2: -1].astype(float).sum(axis=1)/no_of_readings, 2)
            dff.drop(['Supported TX bands'], axis=1, inplace=True)
        #     dff.drop(['combined'], axis=1, inplace=True)
            N = len(dff.index)
            dff.set_index('combined')
            print(
                f"combined value is..{dff['combined'].tolist()[0].split('_')[0]}{dff['combined'].tolist()[0].split('_')[1]}")
            print(f"Antenna value is..{antenna_num}")
            mean_df = dff['Avg'].mean()

            print(f"Antenna Number is..{antenna_num}")
            if(int(antenna_num) == 1):
                avg_ant1.append(np.around(mean_df, 2))
            elif(int(antenna_num) == 2):
                avg_ant2.append(np.around(mean_df, 2))
            elif(int(antenna_num) == 3):
                avg_ant3.append(np.around(mean_df, 2))
            elif(int(antenna_num) == 4):
                avg_ant4.append(np.around(mean_df, 2))

            print(f"Avg of avg for {i} is..{mean_df}")
        #     dff['grp'] = list(chain.from_iterable([x]*2 for x in range(0, N//2)))
        #     dff.groupby('grp').mean()
            print(f"dff is..{dff}")
            dffreadings = dff.iloc[:, 1:-1]

            print(f"shape of dffreadings.. {dffreadings.shape}")
        #     print(f"list of column names for dffreadings..{dffreadings.columns.values.tolist()}")
            dffreadings = dffreadings.drop('combined', axis=1)
            print(f" dffreadings is..{dffreadings}")
            count = (dffreadings.iloc[:, :].astype(float) > limitdb).sum()
            print(type(count))
            print(
                f"count is ..{count.agg(sum)} for Antenna value ..{antenna_num}")
            count = count.agg(sum)
            ant_count(antenna_num, count)

        print(f"rmodlist is..{rmodlist}")
        poprmodlist = rmodlist[1:]
        print(f"poprmodlist is..{poprmodlist}")
        print(f"readingslist..{readingslist}")
        print(f"avg_ant1..{avg_ant1}")
        print(f"avg_ant1..{avg_ant2}")
        print(f"avg_ant1..{avg_ant3}")
        print(f"avg_ant1..{avg_ant4}")
        print(f"ant1_percent..{ant1_percent}")
        print(f"ant2_percent..{ant2_percent}")
        print(f"ant3_percent..{ant3_percent}")
        print(f"ant4_percent..{ant4_percent}")
        print(f"Sector Radio List is.. {sector_radio}")

        data = {'Sector-RadioType': sector_radio, 'Readings Analyzed (10 second intervals)': readingslist,
                f'ANT1 VSWR >={np.around(st.session_state.vswr, 2)}': ant1_percent, f'ANT2 VSWR >={np.around(st.session_state.vswr, 2)}': ant2_percent,
                f'ANT3 VSWR >={np.around(st.session_state.vswr, 2)}': ant3_percent, f'ANT4 VSWR >={np.around(st.session_state.vswr, 2)}': ant4_percent,
                'Average VSWR ANT1': avg_ant1, 'Average VSWR ANT2': avg_ant2,
                'Average VSWR ANT3': avg_ant3, 'Average VSWR ANT4': avg_ant4,
                'RMOD [logical number]':  rmodlist
                }

        # data= {'Sector-Radio Type': [], 'Band': [],'Readings Analyzed (10 second intervals)': [], 'Average DI': [] }

        # Create DataFrame
        df = pd.DataFrame(data)

        return df


def app():
    with st.container():

        st.header('Nokia Health Check (VSWR)')
        st.markdown('Please upload  **only excel files**.')
        hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
        st.markdown(hide_st_style, unsafe_allow_html=True)

        uploaded_vswr_file = st.file_uploader("Choose a file")
        if uploaded_vswr_file is not None:
            # To read file as bytes:
            #  bytes_data = uploaded_file.getvalue()
            #  st.write(bytes_data)# To convert to a string based IO:
            #  stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            #  st.write(stringio)# To read file as string:
            #  string_data = stringio.read()
            #  st.write(string_data)# Can be used wherever a "file-like" object is accepted:
            # excel_read = pd.read_csv(uploaded_file)
            uploadedfn = uploaded_vswr_file.name
            siteid = uploadedfn.split('_')[2][1:]
            st.sidebar.write(f"Site Id: :point_right: *{siteid}*")
            excel_read = pd.read_csv(
                uploaded_vswr_file, skiprows=1, header=None)
            myfiler = excel_read

            myfile1 = excel_read[excel_read.isin(['VSWR']).any(axis=1)].index
            print(excel_read.head())

            print(f"myfile1 is.. {len(myfile1)}")
            if len(myfile1) > 0:
                # print(myfile1)
                # print(hashfunc('-'))
                print(f"one--------{len(myfile1)}")
                indexval = myfile1.values[0]
                print(indexval)
                vswr = excel_read.drop(excel_read.index[0:indexval+1])
                myfilevswr = vswr
                # print(rtwp)
            # print(myfile1)
            # print(hashfunc('-'))

            # print(f"vswr file is..{vswr}")
            else:
                vswr = excel_read
                myfilevswr = vswr

            myfile4 = vswr[vswr.isin(['RTWP LTE']).any(axis=1)].index
            # print(myfile2)
            if len(myfile4) > 0:
                indexval2 = myfile4.values[0]
                print(f"RTWP LTE index ..{indexval2}")
                vswr2 = vswr.drop(excel_read.index[indexval2:])
            #     print(f"vswr2..is {vswr}")
                print(vswr2.shape)
                list(vswr2.columns)
                myfiler_vswr = vswr2
            print(f"vswr clean file.. {myfiler_vswr}")

            # myfiler = myfile.replace({'-': '0'}, regex=True)
            myfilervswr = myfiler_vswr.applymap(
                lambda x: '0' if (x == '-') else x)

            # myfiler.applymap(lambda x: x**2)
            # myfile = myfile[~myfile.eq('-').any(1)]

            myfiler = myfilervswr

            filt = myfiler.iloc[:, 0].str.contains(
                'Radio module') & myfiler.iloc[:, 2].str.contains('Supported TX bands')
            mylist = (myfiler.loc[filt]).values.flatten().tolist()
            newlist = list(map(str, mylist))
            print(type(newlist))
            newlistt = [x for x in newlist if 'nan' not in x.lower()]
            lennewlist = len(newlist)
            lennewlistt = len(newlistt)
            N = lennewlist - lennewlistt
            if N > 0:
                myfiler = myfiler.iloc[:, :-N]
            myfiler.columns = newlistt

            starttime = newlistt[3]
            endtime = newlistt[-1]

            myfiler.reset_index(drop=True)

            minusfilt = ~myfiler.eq('Radio module').any(
                1) & ~myfiler.eq('Supported TX bands').any(1)
            myfiler = myfiler[minusfilt]

            myfiler.dropna()
            # print(myfilevswr.columns.values.tolist())
            radiofile = myfiler.set_index('Radio module')
            radioofile = radiofile.dropna()
            print(f"VSWR data....")
            # radioofile.head(100)
            my_list = radioofile.columns.values.tolist()
            print(f"List of columns.{my_list}")
            print(f"Index column is.. {radioofile.index.name}")

            radioofile['combined'] = radioofile.apply(
                lambda x: '%s_%s' % (x.name, x['Antenna/Port']), axis=1)

            grouped = radioofile.groupby(['combined'])

            rxlist = radioofile['combined'].unique().tolist()
            print(f"List of unique combinations.. {rxlist}")

            greeks = ['Alpha', 'Beta', 'Gamma', 'Delta']

            num_of_columns = radioofile.shape[1]
            no_of_readings = (num_of_columns - 3)
            double_no_of_readings = 2*no_of_readings
            print(f"No. of readings..{no_of_readings}")
            rmodlist = []
            avg_ant1 = []
            avg_ant2 = []
            avg_ant3 = []
            avg_ant4 = []
            rmod_flag = ""
            afhig_ahloa_flag = ""
            current_greek = "Alpha"
            readingslist = []
            ant1_percent = []
            ant2_percent = []
            ant3_percent = []
            ant4_percent = []
            sector_radio = []

            def hashfunc(s):
                return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

            def lastpercent(s):
                if type(s) is str and len(s) > 2:
                    return s[-2] == '%'
                else:
                    return 0

            def lastpercent_yellow(s):
                if type(s) is str and len(s) > 2:
                    return s[-3] == '%'
                else:
                    return 0

            def bg_color(v):
                if (lastpercent(v) and hashfunc(v) != 62282978):
                    return "red"
                elif (lastpercent_yellow(v) and hashfunc(v) != 62282978):
                    return "yellow"
                else:
                    return "lightgreen"

            def ant_count(antenna_num, count):
                if (count/double_no_of_readings) > 0.49:
                    if(int(antenna_num) == 1):
                        ant1_percent.append(
                            " ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                    if(int(antenna_num) == 2):
                        ant2_percent.append(
                            " ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                    if(int(antenna_num) == 3):
                        ant3_percent.append(
                            " ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                    if(int(antenna_num) == 4):
                        ant4_percent.append(
                            " ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                elif (count/double_no_of_readings) < 0.49 and (count/double_no_of_readings) > 0:
                    if(int(antenna_num) == 1):
                        ant1_percent.append(
                            "  ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                    if(int(antenna_num) == 2):
                        ant2_percent.append(
                            "  ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                    if(int(antenna_num) == 3):
                        ant3_percent.append(
                            "  ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                    if(int(antenna_num) == 4):
                        ant4_percent.append(
                            "  ".join([f"{count}|{np.around((count/double_no_of_readings)*100,0)}%", ""]))
                else:
                    if(int(antenna_num) == 1):
                        ant1_percent.append(
                            f"{count}|{np.around((count/double_no_of_readings)*100,0)}%")
                    elif(int(antenna_num) == 2):
                        ant2_percent.append(
                            f"{count}|{np.around((count/double_no_of_readings)*100,0)}%")
                    elif(int(antenna_num) == 3):
                        ant3_percent.append(
                            f"{count}|{np.around((count/double_no_of_readings)*100,0)}%")
                    elif(int(antenna_num) == 4):
                        ant4_percent.append(
                            f"{count}|{np.around((count/double_no_of_readings)*100,0)}%")

            limitdb = 1.39
            cleanrxlist = [x for x in rxlist if str(x) != 'nan']
            print(f"cleanrxlist..{cleanrxlist}")
            for i in cleanrxlist:

                dff = grouped.get_group(i)
                current_rmod = f"{dff['combined'].tolist()[0].split('_')[0]}{dff['combined'].tolist()[0].split('_')[1]}"
                current_afhig_ahloa = current_rmod.split('(')[1][:-1]
                print(f"current afhig_ahloa is..{current_afhig_ahloa}")
                print(f"current_rmod is .. {current_rmod}")
                print(f"rmod_flag is .. {rmod_flag}")
                print(f"afhig_ahloa_flag is..{afhig_ahloa_flag}")

                if rmod_flag != current_rmod:
                    rmodlist.append(current_rmod)
                    readingslist.append(str(double_no_of_readings))
                    if afhig_ahloa_flag != current_afhig_ahloa:
                        current_greek = "Alpha"
                    else:
                        current_greek = greeks[greeks.index(
                            f"{current_greek}") + 1]
                    sector_radio.append(
                        f"{current_greek}-{current_afhig_ahloa}")

                print(f"current_greek is ..{current_greek}")
                print(f"rmodlist inside the for loop is..{rmodlist}")
                rmod_flag = current_rmod
                afhig_ahloa_flag = current_afhig_ahloa
                combined_value = dff['combined'].iloc[0]
                print(f"combined value is.. {combined_value}")
                antenna_num = combined_value.split('_')[2][-1]
            #     RMOD-1/RMOD_R-1(AHFIG)_ANT1
                print(f"dff[combined]is .. {dff['combined'].iloc[0]}")
                print(f"current rmod is. {current_rmod}")
                dff = grouped.get_group(i)
                dff['Avg'] = np.around(
                    dff.iloc[:, 2: -1].astype(float).sum(axis=1)/no_of_readings, 2)
                dff.drop(['Supported TX bands'], axis=1, inplace=True)
            #     dff.drop(['combined'], axis=1, inplace=True)
                N = len(dff.index)
                dff.set_index('combined')
                print(
                    f"combined value is..{dff['combined'].tolist()[0].split('_')[0]}{dff['combined'].tolist()[0].split('_')[1]}")
                print(f"Antenna value is..{antenna_num}")
                mean_df = dff['Avg'].mean()

                print(f"Antenna Number is..{antenna_num}")
                if(int(antenna_num) == 1):
                    avg_ant1.append(np.around(mean_df, 2))
                elif(int(antenna_num) == 2):
                    avg_ant2.append(np.around(mean_df, 2))
                elif(int(antenna_num) == 3):
                    avg_ant3.append(np.around(mean_df, 2))
                elif(int(antenna_num) == 4):
                    avg_ant4.append(np.around(mean_df, 2))

                print(f"Avg of avg for {i} is..{mean_df}")
            #     dff['grp'] = list(chain.from_iterable([x]*2 for x in range(0, N//2)))
            #     dff.groupby('grp').mean()
                print(f"dff is..{dff}")
                dffreadings = dff.iloc[:, 1:-1]

                print(f"shape of dffreadings.. {dffreadings.shape}")
            #     print(f"list of column names for dffreadings..{dffreadings.columns.values.tolist()}")
                dffreadings = dffreadings.drop('combined', axis=1)
                print(f" dffreadings is..{dffreadings}")
                count = (dffreadings.iloc[:, :].astype(float) > limitdb).sum()
                print(type(count))
                print(
                    f"count is ..{count.agg(sum)} for Antenna value ..{antenna_num}")
                count = count.agg(sum)
                ant_count(antenna_num, count)

            print(f"rmodlist is..{rmodlist}")
            poprmodlist = rmodlist[1:]
            print(f"poprmodlist is..{poprmodlist}")
            print(f"readingslist..{readingslist}")
            print(f"avg_ant1..{avg_ant1}")
            print(f"avg_ant1..{avg_ant2}")
            print(f"avg_ant1..{avg_ant3}")
            print(f"avg_ant1..{avg_ant4}")
            print(f"ant1_percent..{ant1_percent}")
            print(f"ant2_percent..{ant2_percent}")
            print(f"ant3_percent..{ant3_percent}")
            print(f"ant4_percent..{ant4_percent}")
            print(f"Sector Radio List is.. {sector_radio}")

            data = {'Sector-RadioType': sector_radio, 'Readings Analyzed (10 second intervals)': readingslist,
                    'ANT1 VSWR >=1.4': ant1_percent, 'ANT2 VSWR >=1.4': ant2_percent,
                    'ANT3 VSWR >=1.4': ant3_percent, 'ANT4 VSWR >=1.4': ant4_percent,
                    'Average VSWR ANT1': avg_ant1, 'Average VSWR ANT2': avg_ant2,
                    'Average VSWR ANT3': avg_ant3, 'Average VSWR ANT4': avg_ant4,
                    'RMOD [logical number]':  rmodlist
                    }

            # data= {'Sector-Radio Type': [], 'Band': [],'Readings Analyzed (10 second intervals)': [], 'Average DI': [] }

            # Create DataFrame
            df = pd.DataFrame(data)
            # df = df.round(1)

            # bool_findings = df.loc[:, ['ANT1 DI Cause', 'ANT2 DI Cause',
            #    'ANT3 DI Cause', 'ANT4 DI Cause']].str.contains('0|0')
            # st.sidebar.write(bool_findings)
            # mask = df['AT4 DI Cause'].str.contains(r'16|64', na=True)
            # df.reset_index(inplace=True)
            # df.reset_index(drop=True, inplace=True)
            # df.set_index('Sector-Radio Type')
            # st.write(df)
            # set first td and first th of every table to not display
            st.markdown("""
            <style>
            table td:nth-child(1) {
                display: none
            }
            table th:nth-child(1) {
                display: none
            }
            </style>
            """, unsafe_allow_html=True)

            dffstyle = df.style.apply(lambda x: [f"background-color: {bg_color(v)}" for v in x], subset=[
                "ANT1 VSWR >=1.4", "ANT2 VSWR >=1.4", "ANT3 VSWR >=1.4", "ANT4 VSWR >=1.4"], axis=1)

            st.write(
                f"*Capture Time Range*: :point_right: [{starttime}] to [{endtime}]")
            st.table(df.style.set_caption("Summary for VSWR (Copyright \
                    Integer Telecom)").apply(lambda x: [f"background-color: {bg_color(v)}" for v in x], subset=["ANT1 VSWR >=1.4", "ANT2 VSWR >=1.4", "ANT3 VSWR >=1.4", "ANT4 VSWR >=1.4"], axis=1))

            styles = [
                dict(selector="tr:hover",
                     props=[("background-color", "#f4f4f4")]),
                dict(selector="th", props=[("color", "#fada5e"),
                                           ("border", "1px solid #eee"),
                                           ("padding", "12px 35px"),
                                           ("border-collapse", "collapse"),
                                           ("background-color", "#00cccc"),
                                           ("text-transform", "uppercase"),
                                           ("font-size", "18px")
                                           ]),
                dict(selector="td", props=[("color", "#999"),
                                           ("border", "1px solid #eee"),
                                           ("padding", "12px 35px"),
                                           ("border-collapse", "collapse"),
                                           ("font-size", "15px")
                                           ]),
                dict(selector="table", props=[
                    ("font-family", 'Arial'),
                    ("margin", "25px auto"),
                    ("border-collapse", "collapse"),
                    ("border", "1px solid #eee"),
                    ("border-bottom", "2px solid #00cccc"),
                ]),
                dict(selector="caption", props=[("caption-side", "bottom")])
            ]

            dffstyle = df.style.set_properties(
                **{'text-align': 'left'}).hide_index()\
                .apply(lambda x: [f"background-color: {bg_color(v)}" for v in x], subset=["ANT1 VSWR >=1.4", "ANT2 VSWR >=1.4", "ANT3 VSWR >=1.4", "ANT4 VSWR >=1.4"], axis=1)
            # st.table(df.style.set_caption("Image by Author (Made \
            # in Pandas)").applymap(color_negative_red, subset=["Average DI"]).apply(lambda x: [f"background-color: {bg_color(v, flat_list)}" for v in x],
            #    subset=["ANT2 DI Cause", "ANT1 DI Cause", "ANT3 DI Cause", "ANT4 DI Cause", ">3db Failures"], axis=1))
            # dffstyle.hide_index().to_excel(
            # f"C:\\Users\\bramh\\Documents\\Downloads\\{siteid}_Output_summary.xlsx", engine='xlsxwriter')
            # read_dff = pd.read_excel(
            # f"C:\\Users\\bramh\\Documents\\Downloads\\{siteid}_Output_summary.xlsx", index_col=0)

            # st.table(read_dff)

            @ st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                writer = pd.ExcelWriter(
                    f"C:\\Users\\bramh\\Documents\\Downloads\\{siteid}_Output_summary.xlsx", engine='xlsxwriter')
                dff = df.to_excel(writer)
                return dff

            # create_git_file(dffstyle.hide_index().to_excel(f"{siteid}_Output_summary.xlsx"),
                # f"{siteid}_Output_summary.xlsx")
            # with pd.ExcelWriter(f"{cwd}\\{siteid}_Output_summary.xlsx") as xlwriter:

                # dffstyle.hide_index().to_excel(
                # xlwriter, engine='xlsxwriter', index=False)

            # csv = convert_df(read_dff)

            # with st.form("my_form"):
            #     st.write("Inside the form")
            #     slider_val = st.slider("Form slider")
            #     checkbox_val = st.checkbox("Form checkbox")

            #     # Every form must have a submit button.
            #     submitted = st.form_submit_button("Submit")
            #     if submitted:
            #         st.write("slider", slider_val, "checkbox", checkbox_val)

            # st.write("Outside the form")

            # st.download_button(
            # label="Download data as XLS",
            # data=pd.read_excel(
            # f"C:\\Users\\bramh\\Documents\\Downloads\\{siteid}_Output_summary.xlsx", index_col=0),
            # file_name=f"{siteid}_Output_summary.xls",
            # mime='application/vnd.ms-excel',
            # writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
            # )

            # st.download_button(
                # label="Download data as Excel",
                # data=dffstyle.hide_index().to_excel(
                # f"{cwd}\\{siteid}_Output_summary.xlsx", engine='xlsxwriter').encode('utf-8'),
                # file_name=f"C:\\Users\\bramh\\Documents\\{siteid}_Output_summary.xlsx",
                # mime='application/vnd.ms-excel',
            # )

            # Using the "with" syntax
            # form = st.form(key='my-form')
            # filename = st.file_picker(
            # "Pick a file", folder=f"{cwd}", type=("xls", "xlsx"))
            # name = form.text_input(
                # f"**File Name**", value=f"{cwd}\\{siteid}_Output_summary.xlsx")
            # submit = form.form_submit_button('Download as Excel')

            # st.write('Press button to download file as Excel')

            # if submit:
                # st.write(f"Downloaded  :point_right: **{name}**")
                # st.write(f'Downloaded {name}')
                # dffstyle.hide_index().to_excel(
                # name, engine='xlsxwriter')

            # with open(f"{cwd}\\{siteid}_Output_summary.xlsx", 'rb') as my_file:
                # st.download_button(label='Download as Excel', data=my_file, file_name=f"{siteid}_Output_summary.xlsx",
                #    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            # col1, col2 = st.columns(2)

            # with col1:
            #     with st.form('Form1'):
            #         st.selectbox('Select Technology', ['LTE', 'NR'], key=1)
            #         st.slider(label='Select intensity',
            #                   min_value=0, max_value=100, key=4)
            #         submitted1 = st.form_submit_button('Submit 1')

            # with col2:
            #     with st.form('Form2'):
            #         st.selectbox('Select Type', ['VSWR', 'RTWP'], key=2)
            #         st.slider(label='Select Intensity',
            #                   min_value=0, max_value=100, key=3)
            #         submitted2 = st.form_submit_button('Submit 2')

            # towrite = io.BytesIO()
            # downloaded_file = dffstyle.to_excel(
                # towrite, encoding='utf-8', index=False, header=True)
            # towrite.seek(0)  # reset pointer
            # b64 = base64.b64encode(towrite.read()).decode()  # some strings
            # linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="myfilename.xlsx">Download excel file</a>'
            # st.markdown(linko, unsafe_allow_html=True)

            def get_col_widths(dataframe):
                # First we find the maximum length of the index column
                idx_max = max(
                    [len(str(s)) for s in dataframe.index.values] + [len(str(dataframe.index.name))])
                # st.sidebar.write(dataframe.index.name)
                len_index = [[s for s in dataframe[col].values]
                             for col in dataframe.columns]
                # st.sidebar.write(len_index)
                # Then, we concatenate this to the max of the lengths of column name and its values for each column, left to right
                return_list = [idx_max] + [max([len(str(s)) for s in dataframe[col].values] + [
                    len(col)]) for col in dataframe.columns]
                # st.sidebar.write(return_list)
                return [idx_max] + [max([len(str(s)) for s in dataframe[col].values] + [len(col)]) for col in dataframe.columns]

                # st.sidebar.write(get_col_widths(df1))

            def to_excel(df, df1):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df = df.set_properties(**{'text-align': 'left'})
                # df.set_properties(subset=['Average DI'], **{'width': '300px'})
                # st.table(df)
                df.to_excel(writer, index=False)
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                # Format all the columns.
                my_format = workbook.add_format(
                    {'align': 'left'})
                my_format.set_align('left')
                # my_format.set_text_wrap()
                cell_format = workbook.add_format(
                    {'bold': True, 'font_color': 'red'})

                # add format for headers
                header_format = workbook.add_format()
                # header_format.set_font_name('Bodoni MT Black')
                # header_format.set_font_color('green')
                # header_format.set_font_size(24)
                header_format.set_align('left')
                header_format.set_text_wrap()
                header_format.set_bold()
                # header_format.set_bg_color('yellow')

                # Write the column headers with the defined format.
                for col_num, value in enumerate(df1.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # Set the default height of all the rows, efficiently.
                worksheet.set_default_row(23)
                # Set the default height of all the columns, efficiently.
                # worksheet.set_default_column(45)

                # align left
                format3 = workbook.add_format({'align': 'left'})

                # worksheet.conditional_format('D1:D100', {'type':     'cell',
                #  'criteria': 'between',
                #  'minimum':  0,
                #  'maximum':  30,
                #  'format':   format3})
                col_width_list = get_col_widths(df1)
                col_width_list[0] = 15  # Sector Radio Type
                col_width_list[2] = 10  # ANT1 VSWR >=1.4
                col_width_list[10] = 30  # RMOD [logical number]
                for i, width in enumerate(col_width_list):
                    worksheet.set_column(i, i, width)

                worksheet.set_row(0, 30)  # Set the height of Row 1 to 30.
                # worksheet.set_column('A:A', None, format1)
                border_fmt = workbook.add_format(
                    {'bottom': 5, 'top': 5, 'left': 5, 'right': 5})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    0, 0, len(df1), len(df1.columns) - 1), {'type': 'no_errors', 'format': border_fmt})
                # worksheet.conditional_format(xlsxwriter.utility.xl_range(
                # 0, 0, 1, len(df1.columns)), {'type': 'no_errors', 'format': my_format})
                writer.save()
                processed_data = output.getvalue()
                return processed_data

            df_xlsx = to_excel(dffstyle, df)
            st.download_button(label='📥 Download As Excel',
                               data=df_xlsx,
                               file_name=f'{siteid}_Output_summary_vswr.xlsx')


app()
