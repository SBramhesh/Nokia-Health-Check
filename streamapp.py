import streamlit as st
import pandas as pd
import re
import copy
import json
import numpy as np
# from streamlit_disqus import st_disqus


# st_disqus("streamlit-disqus-demo")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

st.header('Nokia Health Check')
st.markdown('Please upload  **only excel files**.')
# st.subheader('Cool App')
add_selectbox = st.sidebar.selectbox(
    "OEM Name",
    ("Nokia", "Ericsson")
)
with st.container():

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:  # To read file as bytes:
        #  bytes_data = uploaded_file.getvalue()
        #  st.write(bytes_data)# To convert to a string based IO:
        #  stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        #  st.write(stringio)# To read file as string:
        #  string_data = stringio.read()
        #  st.write(string_data)# Can be used wherever a "file-like" object is accepted:
        primaryexcelsheet = 'Raw_Data'
        excel_read = pd.read_excel(uploaded_file, sheet_name=primaryexcelsheet)
        filt = excel_read.iloc[:, 0].str.contains(
            'Radio module') & excel_read.iloc[:, 2].str.contains('RX carrier')
        mylist = (excel_read.loc[filt]).values.flatten().tolist()
        excel_read.columns = mylist
        # excel_read

        myfile = excel_read.dropna()
        filt2 = myfile.iloc[:, 0].str.contains('RMOD')
        # filt2
        myfile = myfile.loc[filt2]
        radiofile = myfile.set_index('Radio module')
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

        def antlistnum(char, count):
            if char == "1":
                ant1_di_cause.append(
                    f"{count}|{np.around((count/no_of_readings)*100,0)}%")
                ant2_di_cause.append(f"{0}|{0*4}%")
                ant3_di_cause.append(f"{0}|{0*4}%")
                ant4_di_cause.append(f"{0}|{0*4}%")
            if char == "2":
                ant2_di_cause.append(
                    f"{count}|{np.around((count/no_of_readings)*100,0)}%")
                ant1_di_cause.append(f"{0}|{0*4}%")
                ant3_di_cause.append(f"{0}|{0*4}%")
                ant4_di_cause.append(f"{0}|{0*4}%")
            if char == "3":
                ant3_di_cause.append(
                    f"{count}|{np.around((count/no_of_readings)*100,0)}%")
                ant2_di_cause.append(f"{0}|{0*4}%")
                ant1_di_cause.append(f"{0}|{0*4}%")
                ant4_di_cause.append(f"{0}|{0*4}%")
            if char == "4":
                ant4_di_cause.append(
                    f"{count}|{np.around((count/no_of_readings)*100,0)}%")
                ant2_di_cause.append(f"{0}|{0*4}%")
                ant3_di_cause.append(f"{0}|{0*4}%")
                ant1_di_cause.append(f"{0}|{0*4}%")

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

        for i in rxlist:

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

            dff['Avg'] = dff.iloc[:, 2:].sum(axis=1)/no_of_readings
            # print(f"dff average.. {dff.iloc[1,-1]}")
            avg_ant1.append(np.around(dff.iloc[0, -1], 1))
            avg_ant2.append(np.around(dff.iloc[1, -1], 1))
            avg_ant3.append(np.around(dff.iloc[2, -1], 1))
            avg_ant4.append(np.around(dff.iloc[3, -1], 1))

        #     tdff = dff.T
        #     tdff.drop_duplicates()

        #     odff = pd.concat([odff, dff])

        #     filt33 =  tdff.iloc[:,1].astype(str).str.contains(dff['RX carrier'])
        #     lphfile = radiofile.loc[filt33]
        #     tdff['DIF'] = tdff.iloc[:, -25:].agg(['max', 'min'])
            dffdi = dff.iloc[:, 2:-1].agg(['max', 'min'])
            dffdit = dffdi.T
        #     dffd3 =  dffdi.loc['max'] - dffdi.loc['min']
        #     dffnew = dffdit[dffdit.columns.difference(['max', 'min'])]
            dffdit['DI'] = dffdit['max'] - dffdit['min']
            meandif = dffdit['DI'].mean()
        #     dffdit['DI'] =  (dffdit.iloc[0]) - Int(dffdit.iloc[1])
        #     dffd3['mean'] = dffd3['DI'].mean(axis=0)
            grtfilt = dffdit['DI'] > 3
        #     dffgrt = dffdit[grtfilt]
            count = len(dffdit.loc[grtfilt])
            minlist = max(list(dff.iloc[:, 2:-1].max()))
            avgdilist.append(np.around((meandif), 1))

        #     dffd3['Avg']= 0
        #     shape = dffdit.shape
        #     print('DataFrame Shape :', shape)
        #     print('Number of rows :', shape[0])
        #     print('Number of columns :', shape[1])
        #     print(dffd3)
        #     print(dffd3['Avg'], dff3['DI'])

            if count > 0:
                # print('Minimum value where DI > 3')
                #         ant1_di_cause.append(f"{count}|{count*4}%")
                # print(minlist)
                #         print(dff[dff == minlist].stack().index)
                antstr = dff[(dff == minlist).any(axis=1)
                             ]['Antenna/Port'].astype(str)
                # print(type(antstr))
                antindex = antstr[0][-1]
                antlistnum(antindex, count)
            else:
                ant4_di_cause.append(f"{0}|{0*4}%")
                ant2_di_cause.append(f"{0}|{0*4}%")
                ant3_di_cause.append(f"{0}|{0*4}%")
                ant1_di_cause.append(f"{0}|{0*4}%")

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
        data = {'Sector-Radio Type': sectorradiolist, 'CELL [logical name]': rxlist,  'Band': bandlist,
                'RMOD [logical number]':  rmodlist, 'Readings Analyzed (10 second intervals)': readingslist,
                'Average DI': avgdilist, 'Average RTWP ANT1': avg_ant1, 'Average RTWP ANT2': avg_ant2,
                'Average RTWP ANT3': avg_ant3, 'Average RTWP ANT4': avg_ant4, '>3db Failures': grtthreelist,
                'ANT1 DI Cause': ant1_di_cause, 'ANT2 DI Cause': ant2_di_cause, 'ANT3 DI Cause': ant3_di_cause,
                'ANT4 DI Cause': ant4_di_cause
                }

        # data= {'Sector-Radio Type': [], 'Band': [],'Readings Analyzed (10 second intervals)': [], 'Average DI': [] }

        # Create DataFrame
        df = pd.DataFrame(data)
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

        st.table(df)

        # st.map(df)

        # st.write(df)
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        # with st.form("my_form"):
        #     st.write("Inside the form")
        #     slider_val = st.slider("Form slider")
        #     checkbox_val = st.checkbox("Form checkbox")

        #     # Every form must have a submit button.
        #     submitted = st.form_submit_button("Submit")
        #     if submitted:
        #         st.write("slider", slider_val, "checkbox", checkbox_val)

        # st.write("Outside the form")

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='output_summary.csv',
            mime='text/csv',
        )
