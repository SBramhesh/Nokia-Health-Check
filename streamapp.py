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
# import pymongo
# from pymongo import MongoClient
from datetime import datetime
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
add_selectbox = st.sidebar.selectbox(
    "Technology",
    ("LTE", "NR")
)

add_selectbox = st.sidebar.selectbox(
    "Type",
    ("VSWR", "RTWP")
)


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
    elif value > 0:
        color = 'lightgreen'
    else:
        color = 'black'

    return 'background-color: %s' % color


def value_loc(value, df):
    for col in list(df):
        if value in df[col].values.astype(str):
            return (list(df).index(col), df[col][df[col] == value].index[0])


with st.container():

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        #  bytes_data = uploaded_file.getvalue()
        #  st.write(bytes_data)# To convert to a string based IO:
        #  stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        #  st.write(stringio)# To read file as string:
        #  string_data = stringio.read()
        #  st.write(string_data)# Can be used wherever a "file-like" object is accepted:
        # excel_read = pd.read_csv(uploaded_file)
        uploadedfn = uploaded_file.name
        siteid = uploadedfn.split('_')[2][1:]
        st.sidebar.write(f"Site Id: :point_right: *{siteid}*")
        excel_read = pd.read_csv(uploaded_file, skiprows=1, header=None)
        myfiler = excel_read

        myfile1 = excel_read[excel_read.isin(['RTWP LTE']).any(axis=1)].index

        if len(myfile1) > 0:
            # print(myfile1)
            # print(hashfunc('-'))
            print(f"one--------{len(myfile1)}")
            indexval = myfile1.values[0]
            print(indexval)
            rtwp = excel_read.drop(excel_read.index[0:indexval+1])
            myfiler = rtwp
            # print(rtwp)
        # print(myfile1)
        # print(hashfunc('-'))

        # print(rtwp)
        myfile2 = rtwp[rtwp.isin(['RTWP WCDMA']).any(axis=1)].index
        # print(myfile2)
        if len(myfile2) > 0:
            indexval2 = myfile2.values[0]
            rtwp2 = rtwp.drop(excel_read.index[indexval2:])
            print(rtwp2)
            print(rtwp2.shape)
            list(rtwp2.columns)
            myfiler = rtwp2

        # myfiler = myfile.replace({'-': '0'}, regex=True)
        myfiler = myfiler.applymap(lambda x: '0' if (x == '-') else x)
        # myfiler.applymap(lambda x: x**2)
        # myfile = myfile[~myfile.eq('-').any(1)]

        filt = myfiler.iloc[:, 0].str.contains(
            'Radio module') & myfiler.iloc[:, 2].str.contains('RX carrier')
        mylist = (myfiler.loc[filt]).values.flatten().tolist()
        # print(mylist)
        # myfiler.columns = mylist
        # myfiler.reset_index(drop=True)
        newlist = list(map(str, mylist))
        print(type(newlist))
        newlistt = [x for x in newlist if 'nan' not in x.lower()]
        lennewlist = len(newlist)
        lennewlistt = len(newlistt)
        N = lennewlist - lennewlistt
        if N > 0:
            myfiler = myfiler.iloc[:, :-N]
        myfiler.columns = newlistt
        print(newlistt)
        starttime = newlistt[3]
        endtime = newlistt[-1]

        myfiler.reset_index(drop=True)

        # myfiler = myfiler[myfiler.isin(['RMOD-']).any(axis=1)]
        minusfilt = ~myfiler.eq('Radio module').any(
            1) & ~myfiler.eq('RX carrier').any(1)
        myfiler = myfiler[minusfilt]
        # radioofile = myfiler.set_index('Radio module')

        # filt = excel_read.iloc[:, 0].str.contains(
        #     'Radio module') & excel_read.iloc[:, 2].str.contains('RX carrier')
        # mylist = (excel_read.loc[filt]).values.flatten().tolist()
        # excel_read.columns = mylist
        # # excel_read

        # myfile = excel_read.dropna()
        # # myfile[myfile["team"].str.contains("-")==False]
        # myfile = myfile[~myfile.eq('-').any(1)]
        # filt2 = myfiler.iloc[:, 0].str.contains('RMOD')
        # filt2
        # myfile = myfiler.loc[filt2]
        radiofile = myfiler.set_index('Radio module')

        # # Mongo DB#### 389 - 436

        # now = datetime.now()
        # # dd/mm/YY H:M:S
        # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        # first_col = radiofile.iloc[:, :1]
        # first_json = first_col.to_json(orient="records")

        # print(f"----TO JSON---")
        # radiojson = radiofile.to_json(orient="records")
        # # print(radiojson)
        # # data_dict = radioofile.to_dict("records")
        # company.create_index(keys=[("Technology", pymongo.ASCENDING),
        #                            ("timestamp", pymongo.ASCENDING)],
        #                      unique=True,
        #                      name="new_key")
        # company.insert_one({"Technology": "Nokia LTE", "firstcol": first_json,
        #                    "data": radiojson, "timestamp": dt_string})

        # data_from_db = company.find_one({"Technology": "Nokia LTE"})
        # inradiojson = data_from_db["data"]
        # firstradiojson = data_from_db["firstcol"]
        # # df.set_index("Date",inplace=True)
        # print(f"--IN JSON--")
        # fjson = pd.read_json(inradiojson, orient='records')
        # colonejson = pd.read_json(firstradiojson, orient='records')
        # radiodbfile = fjson.set_index('Radio module')
        # radiodbfile = radiodbfile.applymap(
        #     lambda x: '0.0' if (x == '0') else x)
        # # fjson.head(100)
        # # colonejson
        # # print(inradiojson)
        # # myfiler.head(100)
        # # radiodbfile.head(100)

        # print(f"DIFFERECE..")

        # differencedf = radiofile[~radiofile.apply(
        #     tuple, 1).isin(radiodbfile.apply(tuple, 1))]
        # # print(radioofile)
        # # radiodbfile.head(100)
        # print(differencedf.shape)
        # print("Collection:", company)
        # items = company.find()
        # items = list(items)

        # # END of MongoDB stuff
        # #############
        grouped = radiofile.groupby('RX carrier')
        # st.write(radiofile)
        rxlist = radiofile['RX carrier'].unique().tolist()
        num_of_columns = excel_read.shape[1]
        no_of_readings = num_of_columns - 3
        percentage = np.around(100/no_of_readings, 3)

        ant1_di_cause = []
        ant2_di_cause = []
        ant3_di_cause = []
        ant4_di_cause = []

        def hashfunc(s):

            return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

        def lastpercent(s):
            if type(s) is str:
                return s[-1] == '%'
            else:
                return 0

        def bg_color(v, threedlist):
            #     print(threedlist)
            if (lastpercent(v) and hashfunc(v) != 62282978 and v in threedlist):
                return "red"
        #     elif (lastpercent(v) and hashfunc(v) == 62282978):
        #         return "lightgreen"
            else:
                return "lightgreen"

        def highlight_max(x, color):
            return np.where(x == np.nanmax(x.to_numpy()), f"color: {color};", None)

        def color_negative(v, color):
            return f"background: {color};" if v > 3 else f"background: lightgreen"

        # create a function with a dictionary

        def radiotype(char):

            # definition of the dictionary
            prefix = {
                # case 1
                "1": "Alpha",

                # case 2
                "2": "Beta",

                # case 3
                "3": "Gamma",

                # case 4
                "4": "Delta",

                # case 5
                "5": "Epsilon",
                # case 6
                "6": "zeta"

            }
            return prefix.get(char)

        # create a function with a dictionary
        def antnum(char):

            # definition of the dictionary
            prefix = {
                # case 1
                "1": ant1_di_cause,

                # case 2
                "2": ant2_di_cause,

                # case 3
                "3": ant3_di_cause,

                # case 4
                "4": ant4_di_cause,


            }
            return prefix

        # create a function with a dictionary
        def bandtype(char):

            # definition of the dictionary
            prefix = {
                # case 1
                "L1": "L2100-1",

                # case 2
                "B1": "L1900-1",

                # case 3
                "B2": "L1900-2",

                # case 4
                "F1": "LAWS3",

                # case 5
                "E1": "L600",

                # case 6
                "D1": "L700"

            }
            return prefix.get(char)

        def mapradio(cell, str):

            char = cell[-2]
            prefix = radiotype(char)
            return f'{prefix}-{str}'

        def mapband(cell):

            char = f'{cell[0]}{cell[-1]}'
            prefix = bandtype(char)
            return prefix

        def antlistnum(countzero, countone, counttwo, countthree):

            ant1_di_cause.append(
                f"{countzero}|{np.around((countzero/no_of_readings)*100,0)}%")
            ant2_di_cause.append(
                f"{countone}|{np.around((countone/no_of_readings)*100,0)}%")
            ant3_di_cause.append(
                f"{counttwo}|{np.around((counttwo/no_of_readings)*100,0)}%")
            ant4_di_cause.append(
                f"{countthree}|{np.around((countthree/no_of_readings)*100,0)}%")

        absmin = -1000
        sectorradiolist = []
        bandlist = []
        intervallist = []
        avgdilist = []
        rmodlist = []
        readingslist = []
        avg_ant1 = []
        avg_ant2 = []
        avg_ant3 = []
        avg_ant4 = []
        grtthreelist = []

        # no_of_readings = 25
        # st.write(rxlist)
        limitdb = 2.99

        cleanrxlist = [x for x in rxlist if str(x) != 'nan']
        print(cleanrxlist)
        rxlistminusY = [x for x in cleanrxlist if "YPH" not in x]
        print(rxlistminusY)
        rxlistminusYZ = [x for x in rxlistminusY if "ZPH" not in x]
        print(rxlistminusYZ)

        for i in rxlistminusYZ:

            dff = grouped.get_group(i)
            # print(f'value of i is..{i}')
        #     print(dff.index[0])

            # print('index value is..')
            postfix = dff.loc[dff['RX carrier'] == i].index[0].split('(', 1)[
                1][:-1]
            sectorradiolist.append(mapradio(i, postfix))
            bandlist.append(mapband(i))
            rmodlist.append(dff.index[0])
            readingslist.append(str(no_of_readings))

            # dff['Avg'] = dff.iloc[:, 2:].sum(axis=1)/no_of_readings
            dff['Avg'] = np.around(dff.iloc[:, 2:].astype(
                float).sum(axis=1)/no_of_readings, 1)
            # print(f"dff average.. {dff.iloc[1,-1]}")
            avg_ant1.append(np.around(dff.iloc[0, -1].astype(float), 1))
            avg_ant2.append(np.around(dff.iloc[1, -1].astype(float), 1))
            avg_ant3.append(np.around(dff.iloc[2, -1].astype(float), 1))
            avg_ant4.append(np.around(dff.iloc[3, -1].astype(float), 1))

        #     tdff = dff.T
        #     tdff.drop_duplicates()

        #     odff = pd.concat([odff, dff])

        #     filt33 =  tdff.iloc[:,1].astype(str).str.contains(dff['RX carrier'])
        #     lphfile = radiofile.loc[filt33]
        #     tdff['DIF'] = tdff.iloc[:, -25:].agg(['max', 'min'])
            dffdi = dff.iloc[:, 2:-1].astype(float).agg(['max', 'min'])
            dffmax = dff.iloc[:, 2:-1].astype(float).agg(['min'])
            dffapp = dff.append(dffmax, ignore_index=True)
            print(dffapp)
            rangee = dffapp.shape[1]-3
            dffdif = dffapp
            for index in range(rangee):
                dffdif.iloc[:, index+2] = dffapp.iloc[:, index+2].astype(
                    float).apply(lambda x: x - float(dffapp.iloc[4, index+2]))
                # dffdif.iloc[:, index+2] = dffapp.iloc[:, index +
                #   2].apply(lambda x: x - dffapp.iloc[4, index+2])
            #     dffapp.iloc[:,index+2].apply(lambda x: x - dffapp.iloc[4,index+2] )
            dffdif = dffdif.iloc[:-1, :]

            print(dffdif.T)
            dffdift = dffdif.T
            dffdift = dffdift.iloc[2: -1, :]

            print('count 0 ')
            countzero = (dffdift[0].astype(float) > limitdb).sum()
            print(countzero)
            print('---------')
            print('count 1 ')
            countone = (dffdift[1].astype(float) > limitdb).sum()
            print(countone)
            print('---------')
            print('count 2')
            counttwo = (dffdift[2].astype(float) > limitdb).sum()
            print(counttwo)
            print('---------')
            print('count 3 ')
            countthree = (dffdift[3].astype(float) > limitdb).sum()
            print(countthree)
            print('---------')

            antlistnum(countzero, countone, counttwo, countthree)
            dffdit = dffdi.T
        #     dffd3 =  dffdi.loc['max'] - dffdi.loc['min']
        #     dffnew = dffdit[dffdit.columns.difference(['max', 'min'])]
            # dffdit['DI'] = dffdit['max'] - dffdit['min']
            dffdit['DI'] = dffdit['max'].astype(
                float) - dffdit['min'].astype(float)
            meandif = dffdit['DI'].mean()
        #     dffdit['DI'] =  (dffdit.iloc[0]) - Int(dffdit.iloc[1])
        #     dffd3['mean'] = dffd3['DI'].mean(axis=0)
            grtfilt = dffdit['DI'].astype(float) > 2.99
            # grtfilt = dffdit['DI'] > 3
        #     dffgrt = dffdit[grtfilt]
            count = len(dffdit.loc[grtfilt])
            # minlist = max(list(dff.iloc[:, 2:-1].max()))
            minlist = max(list(dff.iloc[:, 2:-1].astype(float).max()))
            avgdilist.append(np.around((meandif), 1))

        #     dffd3['Avg']= 0
        #     shape = dffdit.shape
        #     print('DataFrame Shape :', shape)
        #     print('Number of rows :', shape[0])
        #     print('Number of columns :', shape[1])
        #     print(dffd3)
        #     print(dffd3['Avg'], dff3['DI'])

        #      df[np.isin(df, ['pear','apple']).any(axis=1)] <-- query using numpy

        #         print(dffq)
        #     print(dff)

        #     print(dffdit)
        #     print(meandif)
            # print(count)
            # print(f'{count*4}%')
            grtthreelist.append(
                f"{count}|{np.around((count/no_of_readings)*100,0)}%")
        #     print(dffd3)
            # print('----------')
        # odff.head(30)
        # print('Absolute Minimum')
        # print(absmin)

        # initialise data of lists.
        #     data = {'Name':['Tom', 'nick', 'krish', 'jack'], 'Age':[20, 21, 19, 18]}
        data = {'Sector-Radio Type': sectorradiolist,   'Band': bandlist,
                'Readings Analyzed (10 second intervals)': readingslist,
                'Average DI': avgdilist, '>3db Failures': grtthreelist,
                'ANT1 DI Cause': ant1_di_cause, 'ANT2 DI Cause': ant2_di_cause, 'ANT3 DI Cause': ant3_di_cause,
                'ANT4 DI Cause': ant4_di_cause, 'Average RTWP ANT1': avg_ant1, 'Average RTWP ANT2': avg_ant2,
                'Average RTWP ANT3': avg_ant3, 'Average RTWP ANT4': avg_ant4,
                'RMOD [logical number]':  rmodlist,  'CELL [logical name]': rxlistminusYZ,
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

        dfcause = df.loc[df['Average DI'] > 3, ['ANT1 DI Cause',
                                                'ANT2 DI Cause', 'ANT3 DI Cause', 'ANT4 DI Cause']]
        threedilist = dfcause.values.tolist()
        flat_list = list(chain(*threedilist))
        flat_list[:] = [x for x in flat_list if "0|0" not in x]
        print(flat_list)

        dffstyle = df.style.apply(lambda x: [f"background-color: {bg_color(v, flat_list)}" for v in x],
                                  subset=["ANT2 DI Cause", "ANT1 DI Cause", "ANT3 DI Cause", "ANT4 DI Cause", ">3db Failures"], axis=1)\
            .applymap(color_negative, color='red', subset="Average DI")

        st.write(
            f"*Capture Time Range*: :point_right: [{starttime}] to [{endtime}]")
        st.table(df.style.set_caption("Summary for RTWP LTE (Copyright \
                Integer Telecom)").apply(lambda x: [f"background-color: {bg_color(v, flat_list)}" for v in x],
                                         subset=["ANT2 DI Cause", "ANT1 DI Cause", "ANT3 DI Cause", "ANT4 DI Cause", ">3db Failures"], axis=1)
                 .applymap(color_negative, color='red', subset="Average DI"))

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

        dffstyle = df.style.set_caption("Nokia LTE (Made \
                in Pandas)").applymap(color_negative_red, subset=["Average DI"]).apply(lambda x: [f"background-color: {bg_color(v, flat_list)}" for v in x],
                                                                                       subset=["ANT2 DI Cause", "ANT1 DI Cause", "ANT3 DI Cause", "ANT4 DI Cause", ">3db Failures"], axis=1)
        # st.table(df.style.set_caption("Image by Author (Made \
        # in Pandas)").applymap(color_negative_red, subset=["Average DI"]).apply(lambda x: [f"background-color: {bg_color(v, flat_list)}" for v in x],
        #    subset=["ANT2 DI Cause", "ANT1 DI Cause", "ANT3 DI Cause", "ANT4 DI Cause", ">3db Failures"], axis=1))
        # dffstyle.hide_index().to_excel(
        # f"C:\\Users\\bramh\\Documents\\Downloads\\{siteid}_Output_summary.xlsx", engine='xlsxwriter')
        # read_dff = pd.read_excel(
        # f"C:\\Users\\bramh\\Documents\\Downloads\\{siteid}_Output_summary.xlsx", index_col=0)

        # st.table(read_dff)

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            writer = pd.ExcelWriter(
                f"C:\\Users\\bramh\\Documents\\Downloads\\{siteid}_Output_summary.xlsx", engine='xlsxwriter')
            dff = df.to_excel(writer)
            return dff

        # create_git_file(dffstyle.hide_index().to_excel(f"{siteid}_Output_summary.xlsx"),
            # f"{siteid}_Output_summary.xlsx")
        with pd.ExcelWriter(f"{cwd}\\{siteid}_Output_summary.xlsx") as xlwriter:

            dffstyle.hide_index().to_excel(
                xlwriter, engine='xlsxwriter', index=False)

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
        form = st.form(key='my-form')
        # filename = st.file_picker(
        # "Pick a file", folder=f"{cwd}", type=("xls", "xlsx"))
        name = form.text_input(
            f"**File Name**", value=f"{cwd}\\{siteid}_Output_summary.xlsx")
        submit = form.form_submit_button('Download as Excel')

        # st.write('Press button to download file as Excel')

        if submit:
            st.write(f"Downloaded  :point_right: **{name}**")
            # st.write(f'Downloaded {name}')
            dffstyle.hide_index().to_excel(
                name, engine='xlsxwriter')

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
