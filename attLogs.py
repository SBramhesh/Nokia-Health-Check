from io import StringIO
from io import BytesIO
import streamlit as st
import xlsxwriter
import pandas as pd
import re
import copy
import numpy as np
pd.options.display.precision = 2


def get_first_table(string_data):
    print(string_data.split(r'EARFCNDL')
          [1].split(r'GLOBALCELLID')[0])
    textt = string_data.split(r'EARFCNDL')[
        1].split(r'GLOBALCELLID')[0]
    ddfn = pd.read_csv(StringIO(textt.strip()),
                       sep='|', skiprows=2, header=None)
    return textt


def get_vswr_table(string_data):
    regexp_rtwp = re.compile(r'RTWP')
    regexp_vswr = re.compile(r'VSWR')
    text_vswr = ""
    if string_data.find('VSWR') != -1:
        if regexp_rtwp.search(string_data):
            print(string_data.split(r'VSWR TABLE:')[1].split(r'RTWP TABLE')[0])
            text_vswr = string_data.split(r'VSWR TABLE:')[
                1].split(r'RTWP TABLE')[0]
        else:
            print(string_data.split(r'VSWR TABLE:')
                  [1].split(r'CLI PARSING')[0])
            text_vswr = string_data.split(r'VSWR TABLE:')[
                1].split(r'CLI PARSING')[0]
    return text_vswr


def get_no_vswr_table(string_data):
    regexp_1 = re.compile(r'CLI PARSING: STARTED:')
    regexp_2 = re.compile(r'LNCEL_ID')

    text_novswr = ""

    if regexp_1.search(string_data):
        print(string_data.split(r'CLI PARSING: STARTED:')[
              0].split(r'LNCEL_ID')[1].split(r'RM_EXID_CONF ')[0])
        text_novswr = string_data.split(r'CLI PARSING: STARTED:')[0].split(
            r'LNCEL_ID')[1].split(r'RM_EXID_CONF ')[0]

    return text_novswr


def get_first_table_df(textt):
    ddfn = pd.read_csv(StringIO(textt.strip()),
                       sep='|', skiprows=2, header=None)
    # dffs2 = pd.read_csv(StringIO(d.strip()), delim_whitespace=True, header=None)
    dffn = ddfn.iloc[:, 1:].dropna(thresh=3)
    dffn = dffn.iloc[:, :-1]
    # st.sidebar.write(dffn)
    # st.sidebar.write(len(dffn.index))
    # dffn= dffn.iloc[:,4:9]
    dffn = dffn.iloc[:, [4, 5, 7, 8]].copy()
    dffn.columns = ['Bandwidth', 'BBMOD', 'RMOD_ID', 'RMOD']
    return dffn, len(dffn.index)


def get_no_vswr_df(text_novswr):
    ddfnvswr = pd.read_csv(StringIO(text_novswr.strip()),
                           sep='|', skiprows=2, header=None)
    ddfnvswr = ddfnvswr.iloc[:, 1: -1]
    ddfnvswr = ddfnvswr.dropna(thresh=3)
    columns = [2, 7]
    ddfnn1 = pd.DataFrame(ddfnvswr, columns=columns)
    ddfnn1.columns = ['CELL', 'LNCEL_ID']
    ddfnn1['LNCEL_ID'] = ddfnn1['LNCEL_ID'].astype(int)
    lncel_list = list(ddfnn1['LNCEL_ID'].astype(int))
    lnid_list = [f"LNCEL-{str(x)}" for x in lncel_list if not str(x) == "nan"]
    print(lnid_list)
    ddfnn1['LNCEL_ID'] = lnid_list
    ddfnn1['VSWR_BRANCH_1'] = np.nan
    ddfnn1['VSWR_BRANCH_2'] = np.nan
    ddfnn1['VSWR_BRANCH_3'] = np.nan
    ddfnn1['VSWR_BRANCH_4'] = np.nan
    return ddfnn1


def get_vswr_df(textv):
    ddfnv = pd.read_csv(StringIO(textv.strip()),   sep='|', header=1)
    # dffs2 = pd.read_csv(StringIO(d.strip()), delim_whitespace=True, header=None)
    ddfnv = ddfnv.iloc[:, 1:].dropna(thresh=3)
    ddfnv = ddfnv.iloc[:, :-1]
    print(ddfnv.shape)
    ddfnv.head(50)
    print(ddfnv.columns)
    ddfnv.columns = ddfnv.columns.str.strip()
    cxlist = ddfnv['LNCEL'].unique().tolist()
    print(cxlist)
    cleancxlist = [x for x in cxlist if str(x) != 'LNCEL']
    # print(f"cleancxlist..{cleancxlist}")
    grouped = ddfnv.groupby('LNCEL')
    vswr_branch_1 = []
    vswr_branch_2 = []
    vswr_branch_3 = []
    vswr_branch_4 = []
    lncel_list = []
    cell_list = []
    for i in cleancxlist:
        dff = grouped.get_group(i)
        print(f"dff is..{dff}")
        print(dff.columns)
        print(f"CELL ID is..{dff['LNCEL']}")
        vswr_list = dff['VSWR'].to_list()
        lncell_list = dff['LNCELID'].to_list()
        celll_list = dff['LNCEL'].to_list()
        print(f"vswr list is..{vswr_list}")
        print(f"lncl list is..{lncell_list}")
        print(f"celll_list is.. {celll_list}")
        vswr_branch_1.append(vswr_list[0])
        vswr_branch_2.append(vswr_list[1])
        if len(vswr_list) > 2:
            vswr_branch_3.append(vswr_list[2])
            vswr_branch_4.append(vswr_list[3])
        else:
            vswr_branch_3.append(0)
            vswr_branch_4.append(0)

        lncel_list.append(lncell_list[0])
        cell_list.append(celll_list[0])

    print(f"vswr_branch_1 is.. {vswr_branch_1}")
    print(f"vswr_branch_2 is.. {vswr_branch_2}")
    print(f"vswr_branch_3 is.. {vswr_branch_3}")
    print(f"vswr_branch_4 is.. {vswr_branch_4}")
    print(f"lncel_list is.. {lncel_list}")
    print(f"cell_list is.. {cell_list}")

    data = {'CELL': cell_list, 'LNCEL_ID': lncel_list,   'VSWR_BRANCH_1': vswr_branch_1,
            'VSWR_BRANCH_2': vswr_branch_2, 'VSWR_BRANCH_3': vswr_branch_3, 'VSWR_BRANCH_4': vswr_branch_4}
    df_vswr = pd.DataFrame(data)
    # st.sidebar.write(df_vswr)
    return df_vswr


def get_rssi_table1(string_data):

    regexp_v1 = re.compile(r'CLI PARSING: COMPLETED:')
    regexp_v2 = re.compile(r'CLI PARSING: STARTED:')
    if regexp_v1.search(string_data):
        print(string_data.split(r'CLI PARSING: COMPLETED:')[
            1].split(r'RSSI_ANT_1')[1].split(r'DL_Vol(MBs)')[0])
        textr1 = string_data.split(r'CLI PARSING: COMPLETED:')[1].split(
            r'RSSI_ANT_1')[1].split(r'DL_Vol(MBs)')[0]
    elif regexp_v2.search(string_data):
        print(string_data.split(r'CLI PARSING: STARTED:')[
            1].split(r'RSSI_ANT_1')[1].split(r'DL_Vol(MBs)')[0])
        textr1 = string_data.split(r'CLI PARSING: STARTED:')[1].split(
            r'RSSI_ANT_1')[1].split(r'DL_Vol(MBs)')[0]
    return textr1


def get_rssi_table2(string_data):

    regexp_v1 = re.compile(r'CLI PARSING: COMPLETED:')
    regexp_v2 = re.compile(r'CLI PARSING: STARTED:')
    if regexp_v1.search(string_data):
        print(string_data.split(r'CLI PARSING: COMPLETED:')[
            1].split(r'RSSI_ANT_1')[2])
        textr2 = string_data.split(r'CLI PARSING: COMPLETED:')[
            1].split(r'RSSI_ANT_1')[2]
    elif regexp_v2.search(string_data):
        print(string_data.split(r'CLI PARSING: STARTED:')[
            1].split(r'RSSI_ANT_1')[2])
        textr2 = string_data.split(r'CLI PARSING: STARTED:')[
            1].split(r'RSSI_ANT_1')[2]
    return textr2


def get_rrsi_df1(textr):
    ddf_nrr = pd.read_csv(StringIO(textr.strip()),
                          sep='|', skiprows=2,  header=None)
    # dffs2 = pd.read_csv(StringIO(d.strip()), delim_whitespace=True, header=None)
    ddf_nrr = ddf_nrr.iloc[:, 1:].dropna(thresh=3)
    ddf_nrr = ddf_nrr.iloc[: -1, :-5]
    for i in range(5, 9):
        ddf_nrr.iloc[:, i] = pd.to_numeric(
            ddf_nrr.iloc[:, i], errors='coerce').fillna(0).astype('float')
    ddf_nrr = ddf_nrr.iloc[:, 4:]
    return ddf_nrr


rssi_time = ""
rssi_date = ""


def get_combined_rssi_df(textr2, ddf_nrr):
    ddfnr1 = pd.read_csv(StringIO(textr2.strip()),
                         sep='|', skiprows=2,  header=None)
    # dffs2 = pd.read_csv(StringIO(d.strip()), delim_whitespace=True, header=None)
    ddfnr1 = ddfnr1.iloc[:, 1:].dropna(thresh=3)
    ddfnr1 = ddfnr1.iloc[:, :-5]
    for i in range(5, 9):
        ddfnr1.iloc[:, i] = pd.to_numeric(
            ddfnr1.iloc[:, i], errors='coerce').fillna(0).astype('float')
    print(ddfnr1.shape)
    rssi_time = ddfnr1.iloc[0, 1]
    # st.sidebar.write(f"time is ..{rssi_time}")
    rssi_date = ddfnr1.iloc[0, 0]
    # st.sidebar.write(f"Date is ..{rssi_date}")
    ddfnr1 = ddfnr1.iloc[:, 4:]
    # ddfnr1

    deff = pd.concat([ddfnr1, ddf_nrr])

    # deff
    print(f"{deff.columns.tolist()}")

    # ddfnr1.combine(ddf_nrr, np.sum)
    # ddf_comb = pd.concat([ddfnr1, ddf_nrr]).groupby(ddf_comb.columns.tolist()[:5]).mean()
    # df = pd.concat([df1, df2]).groupby(df.columns.tolist()[1:4]).mean()
    # print(deff.loc['5'])
    deff.columns = ['CELL', 'RSSI_BRANCH_1',
                    'RSSI_BRANCH_2', 'RSSI_BRANCH_3', 'RSSI_BRANCH_4']
    # deff['CELL'] = deff['CELL1']
    deff = deff.groupby('CELL').mean()
    deff.reset_index(inplace=True)
    deff = deff.rename(columns={'index': 'CELL'})
    # deff['DiversityImbalance'] = (deff.max(axis=1) - deff.min(axis=1))
    Row_list = []
    # Iterate over each row
    for index, rows in deff.iterrows():
        # Create list for the current row
        my_list = [rows.RSSI_BRANCH_1, rows.RSSI_BRANCH_2,
                   rows.RSSI_BRANCH_3, rows.RSSI_BRANCH_4]
    #     my_list = np.min(my_list[np.nonzero(my_list)])
        # st.sidebar.write(deff.columns)
        my_list = [i for i in my_list if i != 0]
        if not my_list:
            diff_my_list = 0
        else:
            print(f"my_list,, {my_list}")
            diff_my_list = np.around(min(my_list) - max(my_list), 2)
            print(f"diff is ..{diff_my_list}")
            # append the list to the final list
        Row_list.append(diff_my_list)
    # Print the list
    print(Row_list)

    deff['DiversityImbalance'] = Row_list
    print(deff.columns)
    print(f"max ")
    return deff, rssi_date, rssi_time


def rssi_na(ddfnr):
    rssi_list_4 = []
    rssi_list_3 = []
    rssi_list_2 = []
    rssi_list_1 = []
    for index, rows in ddfnr.iterrows():
        # st.sidebar.write(rows.Bandwidth)
        if not rows.Bandwidth.__contains__("MHz"):
            rssi_list_4.append(np.nan)
            rssi_list_3.append(np.nan)
            rssi_list_2.append(np.nan)
            rssi_list_1.append(np.nan)
        else:
            rssi_list_4.append(rows.RSSI_BRANCH_4)
            rssi_list_3.append(rows.RSSI_BRANCH_3)
            rssi_list_2.append(rows.RSSI_BRANCH_2)
            rssi_list_1.append(rows.RSSI_BRANCH_1)
        # st.sidebar.write(rssi_list_4)
    rssi_list_4 = [np.nan if x == 0 else x for x in rssi_list_4]
    rssi_list_3 = [np.nan if x == 0 else x for x in rssi_list_3]
    rssi_list_2 = [np.nan if x == 0 else x for x in rssi_list_2]
    rssi_list_1 = [np.nan if x == 0 else x for x in rssi_list_1]
    ddfnr["RSSI_BRANCH_4"] = rssi_list_4
    ddfnr["RSSI_BRANCH_3"] = rssi_list_3
    ddfnr["RSSI_BRANCH_2"] = rssi_list_2
    ddfnr["RSSI_BRANCH_1"] = rssi_list_1
    # st.sidebar.table(ddfnr)

    return ddfnr


def get_final_df(deff, dffn, df_vswr):
    ddfnr = deff.join(dffn)
    ddfnr = ddfnr.merge(df_vswr, on='CELL', how='inner')
    print(ddfnr.columns)
    ddfnr = ddfnr.reindex(columns=['CELL', 'LNCEL_ID',  'BBMOD', 'Bandwidth', 'RMOD', 'RMOD_ID',  'RSSI_BRANCH_1', 'RSSI_BRANCH_2', 'RSSI_BRANCH_3',
                                   'RSSI_BRANCH_4',  'DiversityImbalance', 'VSWR_BRANCH_1', 'VSWR_BRANCH_2', 'VSWR_BRANCH_3',
                                   'VSWR_BRANCH_4'])
    print(ddfnr.shape)
    # df_style = ddfnr.style.hide_index()
    # deff['DiversityImbalance'] = (deff.max(axis=1) - deff.min(axis=1))
    return rssi_na(ddfnr)


def get_rssi_bandwidth(deff):
    Five_list = []
    Ten_list = []
    Fifteen_list = []
    Twenty_list = []
    Nan_list = []
    for index, rows in deff.iterrows():
        # Create list for the current row
        my_list = rows.Bandwidth
        #     my_list = np.min(my_list[np.nonzero(my_list)])
        # st.sidebar.write(f"my_list is.. {my_list}")
        # min = a if a < b else b
        five_my_list = index if str(
            my_list).__contains__('5') else 1000
        ten_my_list = index if str(
            my_list).__contains__('10') else 1000
        fifteen_my_list = index if str(
            my_list).__contains__('15') else 1000
        twenty_my_list = index if str(
            my_list).__contains__('20') else 1000
        nan_my_list = index if not str(
            my_list).__contains__('MHz') else 1000

        # append the list to the final list
        Five_list.append(five_my_list)
        Ten_list.append(ten_my_list)
        Fifteen_list.append(fifteen_my_list)
        Twenty_list.append(twenty_my_list)
        Nan_list.append(nan_my_list)
        # st.sidebar.write(f"5MHz list is ..{Five_list}")
        # st.sidebar.write(f"10MHz list is ..{Ten_list}")
        # st.sidebar.write(f"NaN list is ..{Nan_list}")

    # Five_listt = list(filter((np.NaN).__ne__, Five_list))
    # res = [i for i in test_list if i]
    Five_listt = [i for i in Five_list if i != 1000]
    Ten_listt = [i for i in Ten_list if i != 1000]
    Fifteen_listt = [i for i in Fifteen_list if i != 1000]
    Twenty_listt = [i for i in Twenty_list if i != 1000]
    Nan_listt = [i for i in Nan_list if i != 1000]
    return Five_listt, Ten_listt, Fifteen_listt, Twenty_listt, Nan_listt


def bg_color_di(v):
    if (v < -2.99):
        return "red"
    else:
        return "lightgreen"


def bg_color_vswr(v):
    if (v > 1.49):
        return "red"
    else:
        return "#75c609"


def bg_color_AtoF(v):
    return "#79cbf7"


rssi_dict = {'Five': (-110, -98), 'Ten': (-107, -95),
             'Fifteen': (-105.2, -93.2), 'Twenty': (-104, -92)}


def bg_color_five(v):
    if (v < rssi_dict['Five'][0] and v > rssi_dict['Five'][1]):
        return "red"
    else:
        return "lightgreen"


def bg_color_ten(v):
    if (v < rssi_dict['Ten'][0] and v > rssi_dict['Ten'][1]):
        return "red"
    else:
        return "lightgreen"


def bg_color_fifteen(v):
    if (v < rssi_dict['Fifteen'][0] and v > rssi_dict['Fifteen'][1]):
        return "red"
    else:
        return "lightgreen"


def bg_color_twenty(v):
    if (v < rssi_dict['Twenty'][0] and v > rssi_dict['Twenty'][1]):
        return "red"
    else:
        return "lightgreen"


def bg_color_nan(v):
    return "yellow"


def app():
    with st.container():

        st.header('Process AT&T Log Files (RSSI)')
        st.markdown('Please upload  **only log files**.')
        hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
        st.markdown(hide_st_style, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a file", key="att")

        # def form_callback():
        #     st.sidebar.write(st.session_state.my_slider)
        #     st.sidebar.write(st.session_state.my_checkbox)

        if uploaded_file is not None:

            uploadedfn = uploaded_file.name
            siteid = uploadedfn.split('.')[0][3:]

            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            # st.write(bytes_data)

            # To convert to a string based() IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # st.write(stringio)

            # To read file as string:
            string_data = stringio.read()
            # st.write(string_data)

            textt = get_first_table(string_data)

            first_table_df, len_first_df = get_first_table_df(textt)

            # st.table(first_table_df)
            # st.sidebar.write(string_data)

            vswr_table = get_vswr_table(string_data)

            if len(vswr_table) > 1:
                vswr_df = get_vswr_df(vswr_table)
            else:
                no_vswr_table = get_no_vswr_table(string_data)
                vswr_df = get_no_vswr_df(no_vswr_table)

            rssi_table_1 = get_rssi_table1(string_data)

            rssi_df1 = get_rrsi_df1(rssi_table_1)

            # st.table(rssi_df1)

            rssi_table_2 = get_rssi_table2(string_data)

            rssi_combined_df, rssi_date, rssi_time = get_combined_rssi_df(
                rssi_table_2, rssi_df1)

            # st.table(rssi_combined_df)

            df_final = get_final_df(rssi_combined_df, first_table_df, vswr_df)

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

            five_list, ten_list, fifteen_list, twenty_list, nan_list = get_rssi_bandwidth(
                df_final)
            # df_final.apply(lambda x: [
            # "N/A" for v in x],  subset=(twenty_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))
            # st.sidebar.write(df_final)

            st.write(
                f"*Measured Time*: :point_right: {rssi_date} {rssi_time}")
            st.table(df_final.style.apply(lambda x: [
                f"background-color: {bg_color_di(v)}" for v in x], subset=["DiversityImbalance"], axis=1)
                .apply(lambda x: [
                    f"background-color: {bg_color_vswr(v)}" for v in x],  subset=(["VSWR_BRANCH_1", "VSWR_BRANCH_2", "VSWR_BRANCH_3", "VSWR_BRANCH_4"]))
                .apply(lambda x: [
                    f"background-color: {bg_color_five(v)}" for v in x],  subset=(five_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))
                .apply(lambda x: [
                    f"background-color: {bg_color_ten(v)}" for v in x],  subset=(ten_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))
                .apply(lambda x: [
                    f"background-color: {bg_color_fifteen(v)}" for v in x],  subset=(fifteen_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))
                .apply(lambda x: [
                    f"background-color: {bg_color_twenty(v)}" for v in x],  subset=(twenty_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))
                .apply(lambda x: [
                    f"background-color: {bg_color_nan(v)}" for v in x],  subset=(nan_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))
                .apply(lambda x: [
                    f"background-color: {bg_color_AtoF(v)}" for v in x],  subset=(["CELL", "LNCEL_ID", "BBMOD", "Bandwidth", "RMOD", "RMOD_ID"])))

            # df_final.loc[df_final['Bandwidth'] == '10 MHz']
            # my_list = df_final.index[df_final['Bandwidth'].__contains__(
            # 'MHz')].tolist()

            # st.sidebar.write(my_list)
            # st.sidebar.write(five_list)
            # st.sidebar.write(ten_list)
            # st.sidebar.write(fifteen_list)
            # st.sidebar.write(twenty_list)
            df_with_style = df_final.style.apply(lambda x: [
                f"background-color: {bg_color_di(v)}" for v in x], subset=["DiversityImbalance"], axis=1)\
                .apply(lambda x: [
                    f"background-color: {bg_color_vswr(v)}" for v in x],  subset=(["VSWR_BRANCH_1", "VSWR_BRANCH_2", "VSWR_BRANCH_3", "VSWR_BRANCH_4"]))\
                .apply(lambda x: [
                    f"background-color: {bg_color_five(v)}" for v in x],  subset=(five_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))\
                .apply(lambda x: [
                    f"background-color: {bg_color_ten(v)}" for v in x],  subset=(ten_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))\
                .apply(lambda x: [
                    f"background-color: {bg_color_fifteen(v)}" for v in x],  subset=(fifteen_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))\
                .apply(lambda x: [
                    f"background-color: {bg_color_twenty(v)}" for v in x],  subset=(twenty_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))\
                .apply(lambda x: [
                    f"background-color: {bg_color_nan(v)}" for v in x],  subset=(nan_list, ["RSSI_BRANCH_1", "RSSI_BRANCH_2", "RSSI_BRANCH_3", "RSSI_BRANCH_4"]))

            def get_col_widths(dataframe):
                # First we find the maximum length of the index column
                idx_max = max(
                    [len(str(s)) for s in dataframe.index.values] + [len(str(dataframe.index.name))])
                # Then, we concatenate this to the max of the lengths of column name and its values for each column, left to right
                return_list = [idx_max] + [max([len(str(s)) for s in dataframe[col].values] + [
                    len(col)]) for col in dataframe.columns]
                # st.sidebar.write(return_list)
                return [idx_max] + [max([len(str(s)) for s in dataframe[col].values] + [
                    len(col)]) for col in dataframe.columns]

            def to_excel(df_with_style, df_original, df_vswr=None, df2=None):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df_with_style = df_with_style.set_properties(
                    **{'text-align': 'left'})
                df_with_style.to_excel(
                    writer, index=False, na_rep='N/A', startrow=2)

                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                header_format = workbook.add_format({'bg_color': 'yellow'})
                header_format.set_align('left')
                header_format.set_text_wrap()

                text_format = workbook.add_format()
                text_format.set_bold()
                text_format.set_italic()
                text_format.set_font_size(11)
                text_format.set_font_color('navy')

                footer_format = workbook.add_format({'bg_color': '#ffd54f'})
                header_bottom_format = workbook.add_format(
                    {'bg_color': '#ffd54f', 'border': 1})
                header_top_format = workbook.add_format(
                    {'bottom': 0, 'top': 1, 'left': 0, 'right': 0})
                header_top_end_format = workbook.add_format(
                    {'bottom': 0, 'top': 1, 'left': 0, 'right': 1})
                atof_format = workbook.add_format({'bg_color': '#8fc1f7'})
                # footer_format.set_bg_color('skyblue')
                # footer_format.set_bold()
                footer_format.set_font_size(11)
                # Write the column headers with the defined format.
                for col_num, value in enumerate(df_original.columns.values):
                    worksheet.write(2, col_num, value, header_format)
                # Set the default height of all the rows, efficiently.
                worksheet.set_default_row(16)
                # Set the height of Row len(df_original)+2 to 20.
                # worksheet.set_row(len(df_original)+2, 20)
                # worksheet.set_row(len(df_original)+3, 20)
                # worksheet.set_row(len(df_original)+4, 20)
                # worksheet.set_row(len(df_original)+5, 20)

                col_width_list = get_col_widths(df_original)
                col_width_list[0] = 17  # Cell
                col_width_list[6] = 18  # RSSI_BRANCH_1
                col_width_list[7] = 18  # RSSI_BRANCH_2
                col_width_list[8] = 18  # RSSI_BRANCH_3
                col_width_list[9] = 18  # RSSI_BRANCH_4
                col_width_list[10] = 16  # DiversityImbalance
                col_width_list[11] = 13  # VSWR_BRANCH_1

                for i, width in enumerate(col_width_list):
                    worksheet.set_column(i, i, width)
                worksheet.set_row(1, 16)  # Set the height of Row 1 to 30.
                # worksheet.set_column('A:A', None, format1)
                border_fmt = workbook.add_format(
                    {'bottom': 1, 'top': 1, 'left': 1, 'right': 1})
                border_top_fmt = workbook.add_format(
                    {'bottom': 0, 'top': 1, 'left': 1, 'right': 1})
                border_bottom_fmt = workbook.add_format(
                    {'bottom': 1, 'top': 0, 'left': 1, 'right': 1})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    3, 0, len(df_original)+2, len(df_original.columns) - 1), {'type': 'no_errors', 'format': border_fmt})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    3, 0, len(df_original)+2, 5), {'type': 'no_errors', 'format': atof_format})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    1, 0, 1, 14), {'type': 'no_errors', 'format': header_format})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    1, 0, 1, 5), {'type': 'no_errors', 'format': border_top_fmt})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    2, 6, 2, 9), {'type': 'no_errors', 'format': header_bottom_format})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    1, 6, 1, 9), {'type': 'no_errors', 'format': header_top_format})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    2, 11, 2, 14), {'type': 'no_errors', 'format': header_bottom_format})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    1, 11, 1, 14), {'type': 'no_errors', 'format': header_top_format})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    1, 10, 1, 10), {'type': 'no_errors', 'format': border_top_fmt})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    2, 10, 2, 10), {'type': 'no_errors', 'format': border_bottom_fmt})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(
                    2, 0, 2, 5), {'type': 'no_errors', 'format': border_bottom_fmt})
                # <<< Then you can write to a different row
                worksheet.write(
                    1, 7, "RSSI"
                )
                worksheet.write(
                    1, 12, "ReturnLoss/VSWR"
                )
                worksheet.write(
                    1, 14, "", header_top_end_format
                )

                worksheet.write(
                    2, 6, "Branch 1"
                )
                worksheet.write(
                    2, 7, "Branch 2"
                )
                worksheet.write(
                    2, 8, "Branch 3"
                )
                worksheet.write(
                    2, 9, "Branch 4"
                )
                worksheet.write(
                    2, 11, "Branch 1"
                )
                worksheet.write(
                    2, 12, "Branch 2"
                )
                worksheet.write(
                    2, 13, "Branch 3"
                )
                worksheet.write(
                    2, 14, "Branch 4"
                )
                worksheet.write(
                    0, 0, f"Measured Time:    {rssi_date} {rssi_time}",  text_format)
                worksheet.write(
                    len(df_original)+3, 0, "TargetThresholds", footer_format
                )
                worksheet.write(
                    len(df_original)+3, 1, "", footer_format
                )
                worksheet.write(
                    len(df_original)+3, 6, "-110>5MHz<-98", footer_format
                )
                worksheet.write(
                    len(df_original)+3, 7, "", footer_format
                )
                worksheet.write(
                    len(df_original)+4, 6, "-107>10Mhz<-95", footer_format
                )
                worksheet.write(
                    len(df_original)+4, 7, "", footer_format
                )
                worksheet.write(
                    len(df_original)+5, 6, "-105.2>15Mhz<-93.2", footer_format
                )
                worksheet.write(
                    len(df_original)+5, 7, "", footer_format
                )
                worksheet.write(
                    len(df_original)+6, 6, "-104>20Mhz<-92", footer_format
                )
                worksheet.write(
                    len(df_original)+6, 7, "", footer_format
                )
                worksheet.write(
                    len(df_original)+3, 10, "<3.0", footer_format
                )
                worksheet.write(
                    len(df_original)+3, 12, "<1.5", footer_format
                )

                writer.save()
                processed_data = output.getvalue()
                return processed_data
            df_xlsx = to_excel(df_with_style, df_final)
            st.download_button(label='📥 Download As Excel',
                               data=df_xlsx,
                               file_name=f'AT&T_{siteid}_Output_summary.xlsx')


app()
