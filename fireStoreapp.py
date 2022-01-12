import streamlit as st
from google.cloud import firestore
from datetime import datetime
import pandas as pd
import re
import copy
import numpy as np
import hashlib
from itertools import chain
from vswrapp import process_vswr
import io
import xlsxwriter
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

# Authenticate to Firestore with the JSON account key.
db = firestore.Client.from_service_account_json("firestore-key.json")
dbname = 'Nokiadbprod'
if 'vswr' not in st.session_state:
    st.session_state['vswr'] = 1.4
# import json
# key_dict = json.loads(st.secrets["textkey"])
# st.sidebar.write(key_dict)
# creds = service_account.Credentials.from_service_account_info(key_dict)
# db = firestore.Client(credentials=creds, project="streamlit-reddit")


# # Mongo DB#### 389 - 436

# now = datetime.now()
# # dd/mm/YY H:M:S
# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

# first_col = radiofile.iloc[:, :1]
# first_json = first_col.to_json(orient="records")

# print(f"----TO JSON---")
# radiojson = radiofile.to_json(orient="records")
# print(radiojson)
# data_dict = radioofile.to_dict("records")
# doc_ref = db.collection(u'{dbname}').document(u'LTE')
# doc_ref.set({
#     u'Technology': u'Nokia LTE',
#     u'firstcol': first_json,
#     u'data': radiojson,
#     "timestamp": dt_string
# })

nokia_ref = db.collection(dbname)

nokiadoc = nokia_ref.get()


def color_negative_red(value):

    if value > 3:
        color = 'red'
    else:
        color = 'lightgreen'

    return 'background-color: %s' % color


# genre = st.radio(
#     "What's your favorite movie genre",
#     ('Comedy', 'Drama', 'Documentary'))

# if genre == 'Comedy':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn't select comedy.")

def app():
    hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    nokiaoptions = []
    for doc in nokia_ref.stream():
        nokia = doc.to_dict()
        # st.sidebar.write(nokia["site_id"])
        timestamp = nokia["timestamp"]
        technology = nokia["Technology"]
        data = nokia["data"]
        siteid = nokia["site_id"]
        nokiaoptions.append(f"Site Id: {siteid}- Processed on: {timestamp}")

        # st.subheader(f"Data: {data}")
        # st.write(f":timestamp: [{timestamp}]")
        # st.sidebar.write("The Time Stamp is: ", timestamp)
        # st.sidebar.write("The site ID   is: ", siteid)
        # inradiojson = nokiadoc.data
    option = st.selectbox(
        'Nokia RTWP Data Processed to Date',
        nokiaoptions)

    # st.write('You selected:', option)
    siteselected = option.split('-')[0].split(':')[1]
    timestampp1 = option.split('-')[1].split(':')[1].rstrip().lstrip()
    timestampp2 = option.split('-')[1].split(':')[2].rstrip().lstrip()
    timestampp3 = option.split('-')[1].split(':')[3].rstrip().lstrip()
    timestampp = timestampp1 + ':' + timestampp2 + ':' + timestampp3
    # st.write('You selected Site:', siteselected.rstrip().lstrip())
    # st.write('You selected Time Stamp:', timestampp)

    doc_ref = db.collection(dbname).document(
        u'PH23909B->2021-12-31 19:19:49.007444')

    doc = doc_ref.get()
    # if doc.exists:
    # st.write(f'Document data: {doc.to_dict()}')
    # else:
    # st.write(u'No such document!')

    # Note: Use of CollectionRef stream() is prefered to get()
    # st.sidebar.write(siteselected)
    doks = db.collection(dbname).where(
        u'site_id', u'==', siteselected.rstrip().lstrip()).where(u'timestamp', u'==', timestampp).stream()
    selectedsite = ""
    for doc in doks:
        # st.write(f'{doc.id} => {doc.to_dict()}')
        selectedsite = doc.to_dict()["site_id"]
        inradiojson = doc.to_dict()["data"]
        if "vswr" in doc.to_dict():
            vswrjson = doc.to_dict()["vswr"]
            vswr_json = pd.read_json(vswrjson, orient='records')
            df_vswr = process_vswr(vswr_json)
        fradiojson = doc.to_dict()["firstcol"]
        fjson = pd.read_json(inradiojson, orient='records')
        firstcol = pd.read_json(fradiojson, orient='records')
        # radiodbfile = fjson.set_index('Radio module')
        radiodbfile = fjson.applymap(
            lambda x: '0.0' if (x == '0') else x)
        # vswrdbfile = vswr_json.applymap(
        # lambda x: '0.0' if (x == '0') else x)
        # st.sidebar.table(vswrdbfile)

        st.write(
            f"*Raw Data*: :point_down:")
        st.table(radiodbfile.head(100))
        # st.table(firstcol.head(100))
        radiofile = radiodbfile.set_index('Radio module')
        grouped = radiofile.groupby('RX carrier')
        # st.write(radiofile)
        mylist = list(radiodbfile.columns.values)
        # print(mylist)
        # myfiler.columns = mylist
        # myfiler.reset_index(drop=True)
        # st.write(mylist)

        starttime = mylist[3]
        endtime = mylist[-1]
        # st.table(radiofile)
        rxlist = radiofile['RX carrier'].unique().tolist()
        # st.write(rxlist)
        num_of_columns = radiodbfile.shape[1]
        no_of_readings = num_of_columns - 3
        percentage = np.around(100/no_of_readings, 3)

        ant1_di_cause = []
        ant2_di_cause = []
        ant3_di_cause = []
        ant4_di_cause = []

        def hashfunc(s):

            return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

        def lastpercent_vswr(s):
            if type(s) is str and len(s) > 2:
                return s[-2] == '%'
            else:
                return 0

        def lastpercent(s):
            if type(s) is str:
                return s[-1] == '%' or s[-2] == '%'
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

        def lastpercent_yellow(s):
            if type(s) is str and len(s) > 2:
                return s[-3] == '%'
            else:
                return 0

        def bg_vswr_color(v):
            if (lastpercent_vswr(v) and hashfunc(v) != 62282978):
                return "red"
            elif (lastpercent_yellow(v) and hashfunc(v) != 62282978):
                return "yellow"
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

        def antlistnum(meandif, countzero, countone, counttwo, countthree):
            zerostring = f"{countzero}|{np.around((countzero/no_of_readings)*100,0)}%"
            zstring = " ".join([zerostring, ""])

            if meandif > 2.99:
                ant1_di_cause.append(
                    " ".join([f"{countzero}|{np.around((countzero/no_of_readings)*100,0)}%", ""]))
                ant2_di_cause.append(
                    " ".join([f"{countone}|{np.around((countone/no_of_readings)*100,0)}%", ""]))
                ant3_di_cause.append(
                    " ".join([f"{counttwo}|{np.around((counttwo/no_of_readings)*100,0)}%", ""]))
                ant4_di_cause.append(
                    " ".join([f"{countthree}|{np.around((countthree/no_of_readings)*100,0)}%", ""]))
            else:
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

        cleanrxlist = [x for x in rxlist if x != 'null' and x != None]
        print(cleanrxlist)
        rxlistminusY = [x for x in cleanrxlist if "YPH" not in x]
        print(rxlistminusY)
        rxlistminusYZ = [x for x in rxlistminusY if "ZPH" not in x]
        print(rxlistminusYZ)

        for i in rxlistminusYZ:

            dff = grouped.get_group(i)
            print(f'value of i is..{i}')
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
            print(f"Dash value is...{dff.iloc[0, -1]}")
            if dff.iloc[0, -1].astype(float) == 0.0:
                avg_ant1.append(0.0)
            else:
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
            antlistnum(meandif, countzero, countone, counttwo, countthree)

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
            if meandif > 2.99:
                grtthreelist.append(
                    " ".join([f"{count}|{np.around((count/no_of_readings)*100,0)}%", ""]))
            else:
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
        if 'df_vswr' in locals():
            # df_vswr exists.
            dffstyle_vswr = df_vswr.style.apply(lambda x: [f"background-color: {bg_vswr_color(v)}" for v in x], subset=[
                f"ANT1 VSWR >={st.session_state.vswr}", f"ANT2 VSWR >={st.session_state.vswr}", f"ANT3 VSWR >={st.session_state.vswr}", f"ANT4 VSWR >={st.session_state.vswr}"], axis=1)
        st.write(
            f"*Output Summary*: :point_down:")
        st.write(
            f"*Capture Time Range*: :point_right: [{starttime}] to [{endtime}]")
        st.table(df.style.set_caption("Summary for RTWP LTE (Copyright \
                Integer Telecom)").apply(lambda x: [f"background-color: {bg_color(v, flat_list)}" for v in x],
                                         subset=["ANT2 DI Cause", "ANT1 DI Cause", "ANT3 DI Cause", "ANT4 DI Cause", ">3db Failures"], axis=1)
                 .applymap(color_negative, color='red', subset="Average DI"))
        if 'df_vswr' in locals():
            st.write(
                f"*VSWR*: :point_down:")
            st.table(df_vswr.style.set_caption("Summary for VSWR (Copyright \
                Integer Telecom)").apply(lambda x: [f"background-color: {bg_vswr_color(v)}" for v in x], subset=[f"ANT1 VSWR >={st.session_state.vswr}", f"ANT2 VSWR >={st.session_state.vswr}", f"ANT3 VSWR >={st.session_state.vswr}", f"ANT4 VSWR >={st.session_state.vswr}"], axis=1))

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

        def to_excel(df, df1, df_vswr=None, df2=None):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df = df.set_properties(**{'text-align': 'left'})
            # df.set_properties(subset=['Average DI'], **{'width': '300px'})
            # st.table(df)
            df.to_excel(writer, index=False)
            # df1.to_excel(writer, sheet_name='Result',
            #              startrow=1, startcol=0)

            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            vswr_format = workbook.add_format()
            vswr_format.set_bold()
            if df_vswr is not None:
                worksheet.write_string(
                    df1.shape[0] + 4, 0, 'VSWR', vswr_format)
                df_vswr.to_excel(writer, sheet_name='Sheet1',
                                 startrow=df1.shape[0] + 5, startcol=0, index=False)
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

            # add format for vswr headers
            header_vswr_format = workbook.add_format()
            # header_format.set_font_name('Bodoni MT Black')
            # header_format.set_font_color('green')
            # header_format.set_font_size(24)
            header_vswr_format.set_align('left')
            header_vswr_format.set_text_wrap()
            header_vswr_format.set_bold()

            if df_vswr is not None:
                # Write the column headers with the defined format.
                for col_num, value in enumerate(df2.columns.values):
                    worksheet.write(len(df1) + 5, col_num,
                                    value, header_vswr_format)

            # Set the default height of all the rows, efficiently.
            worksheet.set_default_row(30)
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
            col_width_list[2] = 10  # Readings Analyzed
            col_width_list[3] = 10  # Average DI
            for i, width in enumerate(col_width_list):
                worksheet.set_column(i, i, width)

            worksheet.set_row(0, 30)  # Set the height of Row 1 to 30.
            # worksheet.set_column('A:A', None, format1)
            border_fmt = workbook.add_format(
                {'bottom': 5, 'top': 5, 'left': 5, 'right': 5})
            worksheet.conditional_format(xlsxwriter.utility.xl_range(
                0, 0, len(df1), len(df1.columns) - 1), {'type': 'no_errors', 'format': border_fmt})
            if df_vswr is not None:
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    len(df1) + 5, 0, len(df2) + len(df1) + 5, len(df2.columns)), {'type': 'no_errors', 'format': border_fmt})
            # worksheet.conditional_format(xlsxwriter.utility.xl_range(
            # 0, 0, 1, len(df1.columns)), {'type': 'no_errors', 'format': my_format})
            writer.save()
            processed_data = output.getvalue()
            return processed_data

        if 'df_vswr' in locals() and 'dffstyle_vswr' in locals():
            # df_vswr exists.
            df_xlsx = to_excel(dffstyle, df, dffstyle_vswr, df_vswr)
        else:
            df_xlsx = to_excel(dffstyle, df)

        st.download_button(label='📥 Download As Excel',
                           data=df_xlsx,
                           file_name=f'{siteid}_Output_summary.xlsx')

    # doc_reffer = db.collection(u'{dbname}')

    # sites = doc_reffer.where(
        # u'site_id', u'==', siteselected).stream()
    # st.write(f"site  id is..{option.split('-')[0].split(':')[1]}")
    # st.write(f"Sites..{list(sites)}")

    # for doc in sites:
        # st.write(u'{} => {}'.format(doc.id, doc.to_dict()))

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
