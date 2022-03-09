from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from io import StringIO
from io import BytesIO
import streamlit as st
import xlsxwriter
import pandas as pd
import re
import copy
import numpy as np
import time
import xml.dom.minidom
import aloha_dict
pd.options.display.precision = 2
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
xml_dict = aloha_dict.return_dict()


# pattern = re.compile('MRBTS-(\d{7,})')
# print(type(pattern))
# xml_str = re.sub('MRBTS-(\d{7,})', r'MRBTS-1234567', xml_str)
# print(pattern.findall(xml_str))
# # print(xml_str)


def app():
    st.title('5G Nokia Scripting')
    st.session_state['download'] = False

    def process_xml():
        if uploaded_file_tnd_ciq is not None and uploaded_file_nr_ciq is not None and uploaded_file_lte_ciq is not None:

            # my_bar = st.progress(0)

            # for percent_complete in range(100):
            #     time.sleep(0.02)
            #     my_bar.progress(percent_complete + 1)
            with st.spinner('Please Kindly Wait...'):
                for i in key_list:
                    nrcell_modify(i, nrcell_par_dict.get(i))
                # print(soup.find_all(attrs={"name": "btsName"}))mrbts_key_list = [key for key in mrbts_par_dict]
                # print(key_list)
                for mi in mrbts_key_list:
                    modify_mrbts_tag(mi, mrbts_par_dict.get(mi))
                process_tnd_pars()
                remove_blank_spaces()
                process_lte_cellpar()
                st.success('XML successfully parsed :point_down:!!')
                # st.write(f"*VSWR*: :point_down:")
            st.session_state['download'] = True

    with st.form("my_form"):
        with st.container():

            hide_st_style = """
                    <style>
                    # MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
            st.markdown(hide_st_style, unsafe_allow_html=True)

            # col1, col2 = st.columns([1, 3])

            # col1.markdown('**Upload NR CIQ file**.')
            uploaded_file_nr_ciq = st.file_uploader(
                "Upload NR CIQ File", key="nrciq")

            # def form_callback():
            #     print(st.session_state.my_slider)
            #     print(st.session_state.my_checkbox)

            if uploaded_file_nr_ciq is not None:

                uploadedfn = uploaded_file_nr_ciq.name
                siteid = uploadedfn.split('.')[0][3:]

                # To read file as bytes:
                bytes_data = uploaded_file_nr_ciq.getvalue()
                # st.write(bytes_data)

                # To convert to a string based() IO:
                # stringio = StringIO(uploaded_file_nr_ciq.getvalue().decode("utf-8"))
                # st.write(stringio)

                # To read file as string:
                # nr_ciq_data = stringio.read()

                ciq_siteMain = pd.read_excel(
                    uploaded_file_nr_ciq, sheet_name='SiteMainPar', header=3, skiprows=None)
                ciq_siteMain = ciq_siteMain.dropna(thresh=3)
                ciq_siteMain = ciq_siteMain.iloc[:, 1:]
                # ciq_siteMain.head(10)

                ciq_cell_par = pd.read_excel(
                    uploaded_file_nr_ciq, sheet_name='CellPar', header=3, skiprows=None)
                ciq_cell_par = ciq_cell_par.dropna(thresh=5)
                ciq_cell_par = ciq_cell_par.iloc[:, 1:]
                # ciq_cell_par

                ho_cell_par = pd.read_excel(
                    uploaded_file_nr_ciq, sheet_name='HO Inter NR', header=3, skiprows=None)
                ho_cell_par = ho_cell_par.dropna(thresh=5)
                ho_cell_par = ho_cell_par.iloc[:, 1:]
                # ho_cell_par
                # print(ciq_cell_par)
                inr_cell_par = pd.read_excel(
                    uploaded_file_nr_ciq, sheet_name='Idle Inter NR', header=3, skiprows=None)
                inr_cell_par = inr_cell_par.dropna(thresh=5)
                inr_cell_par = inr_cell_par.iloc[:, 1:]
                # inr_cell_par

        with st.container():

            hide_st_style = """
                    <style>
                    # MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
            st.markdown(hide_st_style, unsafe_allow_html=True)

            # col1, col2 = st.columns([1, 3])

            # col1.markdown('**Upload TND CIQ file**.')
            uploaded_file_tnd_ciq = st.file_uploader(
                "Upload TND CIQ File", key="tndciq")

            # def form_callback():
            #     print(st.session_state.my_slider)
            #     print(st.session_state.my_checkbox)
        with st.container():

            hide_st_style = """
                    <style>
                    # MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
            st.markdown(hide_st_style, unsafe_allow_html=True)

            # col1, col2 = st.columns([1, 3])

            # col1.markdown('**Upload TND CIQ file**.')
            uploaded_file_lte_ciq = st.file_uploader(
                "Upload LTE CIQ File", key="lteciq")

            # def form_callback():
            #     print(st.session_state.my_slider)
            #     print(st.session_state.my_checkbox)
            if uploaded_file_lte_ciq is not None:

                uploadedfn_lte = uploaded_file_lte_ciq.name
                # siteid = uploadedfn.split('.')[0][3:]

                # To read file as bytes:
                bytes_data = uploaded_file_lte_ciq.getvalue()
                # st.write(bytes_data)

                # To convert to a string based() IO:
                # stringio = StringIO(uploaded_file_tnd_ciq.getvalue().decode("utf-8"))

                lte_nokia = pd.read_excel(uploaded_file_lte_ciq,
                                          sheet_name='IDLE InterFreq. Template', header=4, skiprows=None)

                freq_list = lte_nokia['EUTRA frequency value'][5:].to_list()
                print(freq_list)

        with st.container():

            hide_st_style = """
                    <style>
                    # MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
            st.markdown(hide_st_style, unsafe_allow_html=True)

            option = st.selectbox(
                'FDD EQM Options',
                ('3AHLOA(shared)_15BW+3AEHC (shared)_100BW_NoAHFIG', '3AHLOA(shared)_20BW+3AEHC (shared)_100BW_NoAHFIG', '3AHLOA(shared)_15BW+3AHFIG(shared)+3AEHC (shared)_20BW', '3AHLOA(shared)_15BW+3AHFIG(shared)+3AEHC (shared)_60BW', '3AHLOA(shared)_15BW+3AHFIG(shared)+3AEHC (shared)_100BW',
                 '3AHLOA(shared)_20BW+3AHFIG (shared)_NoAEHC', '3AHLOA(shared)_20BW+3AHFIG(shared)+3AEHC (shared)_20BW', '3AHLOA(shared)_20BW+3AHFIG(shared)+3AEHC (shared)_40BW', '3AHLOA(shared)_20BW+3AHFIG(shared)+3AEHC (shared)_60BW', '3AHLOA(shared)_20BW+3AHFIG(shared)+3AEHC (shared)_100BW'))

            print('You selected:', option)
            print(list(xml_dict.keys()))
            print(
                len(xml_dict["3AHLOA(shared)_20BW+3AEHC (shared)_100BW_NoAHFIG"]))
            xml_str = str(xml_dict[option]).replace('\n', '')

            if uploaded_file_tnd_ciq is not None:

                uploadedfn_tnd = uploaded_file_tnd_ciq.name
                # siteid = uploadedfn.split('.')[0][3:]

                # To read file as bytes:
                bytes_data = uploaded_file_tnd_ciq.getvalue()
                # st.write(bytes_data)

                # To convert to a string based() IO:
                # stringio = StringIO(uploaded_file_tnd_ciq.getvalue().decode("utf-8"))

                tnd_nokia = pd.read_excel(uploaded_file_tnd_ciq,
                                          sheet_name='gNodeB Nokia', header=4, skiprows=None)
                tnd_nokia = tnd_nokia.dropna(thresh=5)
                tnd_nokia = tnd_nokia.iloc[:, :]

        # tnd_nokia

        # with open(xml_path, 'r') as f:
        #     file = f.read()

        # 'xml' is the parser used. For html files, which BeautifulSoup is typically used for, it would be 'html.parser'.
        soup = BeautifulSoup(xml_str, "xml")
        # print(str(soup))
        print("--cellNAme---")
        print(soup.find_all(attrs={"name": "cellName"}))
        print("--cellName End---")

        # tree = ET.parse(xml_path)
        # root = tree.getroot()
        # print(root.tag)
        # # print(root.attrib)
        # # for child in root:
        # #     print(child.tag, child.attrib)
        # for neighbor in root.iter('header'):
        #     print(neighbor.attrib)
        # rooot = ET.fromstring(tree)

        # TND ---
        def return_band(x):
            if 599 < x < 701:
                return 'B2'
            elif 1950 < x < 2399:
                return 'B4'
            elif 5010 < x < 5179:
                return 'B12'
            elif 68586 < x < 68935:
                return 'B71'
            else:
                return 'Del'

        def return_priority(band, lncel):
            if (band == 'B4' or band == 'B2'):
                if lncel == 'mbw25':
                    return 4
                else:
                    return 5
            elif band == 'B12':
                return 2
            elif band == 'B71':
                return 3

        def return_mbw(lncel):
            if lncel == 21:
                return 'mbw25'
            elif lncel == 22:
                return 'mbw50'
            elif lncel == 23:
                return 'mbw75'
            else:
                return 'mbw100'

        def get_nrlim_index(nrlim_df):
            mf_tags = soup.find_all(attrs={"name": "allowedMeasBw"})
            # NRSYSINFO_PROFILE-0/NRLIM-
            nrlim0_index_list = []
            nrlim1_index_list = []
            for mf_tag in mf_tags:
                if mf_tag.parent.name.find('managedObject') > -1:
                    # print(
                    # f"found managedObject in parent..{mf_tag.parent['class']}")
                    if mf_tag.parent['class'].find('NRLIM') > -1 and mf_tag.parent['distName'].find('NRSYSINFO_PROFILE-0/NRLIM-') > -1:
                        nrlim0_index_list.append(
                            str(mf_tag.parent['distName']).split('/')[-1].split('-')[-1])
                    elif mf_tag.parent['class'].find('NRLIM') > -1 and mf_tag.parent['distName'].find('NRSYSINFO_PROFILE-1/NRLIM-') > -1:
                        nrlim1_index_list.append(
                            str(mf_tag.parent['distName']).split('/')[-1].split('-')[-1])
            return nrlim0_index_list, nrlim1_index_list

        def replace_nrlim_tags(nrlim_df, nrlim_0_dict, nrlim_1_dict):
            mf_tags = soup.find_all(attrs={"name": "allowedMeasBw"})
            # NRSYSINFO_PROFILE-0/NRLIM-
            print(nrlim_df)
            for mf_tag in mf_tags:
                if mf_tag.parent.name.find('managedObject') > -1:
                    # print(
                    # f"found managedObject in parent..{mf_tag.parent['class']}")
                    if mf_tag.parent['class'].find('NRLIM') > -1 and mf_tag.parent['distName'].find('NRSYSINFO_PROFILE-0/NRLIM-') > -1:
                        band_0 = nrlim_0_dict.get(
                            str(mf_tag.parent['distName']).split('/')[-1].split('-')[-1])
                        print(f"band_0...{band_0}")
                        if str(band_0) in nrlim_df.values:
                            band_group_0 = nrlim_df.groupby('band')
                            print(band_group_0.get_group(str(band_0)))
                            print(band_group_0.get_group(
                                str(band_0))['lncel'].iloc[0])
                            print('-------')
                            mf_tag.string = band_group_0.get_group(str(band_0))[
                                'lncel'].iloc[0]
                            freq = mf_tag.find_next_sibling("p")
                            print(band_group_0.get_group(
                                str(band_0))['dlink'].iloc[0])
                            print('-------')
                            freq.string = str(band_group_0.get_group(
                                str(band_0))['dlink'].iloc[0])
                            print(band_group_0.get_group(
                                str(band_0))['priority'].iloc[0])
                            print('-------')
                            prior = freq.find_next_sibling("p")
                            prior.string = str(band_group_0.get_group(
                                str(band_0))['priority'].iloc[0])

                    elif mf_tag.parent['class'].find('NRLIM') > -1 and mf_tag.parent['distName'].find('NRSYSINFO_PROFILE-1/NRLIM-') > -1:
                        band_1 = nrlim_1_dict.get(
                            str(mf_tag.parent['distName']).split('/')[-1].split('-')[-1])
                        print(band_1)
                        if str(band_1) in nrlim_df.values:
                            band_group_1 = nrlim_df.groupby('band')
                            print(type(band_group_1.get_group(str(band_1))))
                            print(band_group_1.get_group(
                                str(band_1))['lncel'].iloc[0])
                            print('-------')
                            mf_tag.string = band_group_0.get_group(str(band_1))[
                                'lncel'].iloc[0]
                            freq = mf_tag.find_next_sibling("p")
                            print(band_group_1.get_group(
                                str(band_1))['dlink'].iloc[0])
                            print('-------')
                            freq.string = str(band_group_1.get_group(
                                str(band_1))['dlink'].iloc[0])
                            print(band_group_1.get_group(
                                str(band_1))['priority'].iloc[0])
                            print('-------')
                            prior = freq.find_next_sibling("p")
                            prior.string = str(band_group_1.get_group(
                                str(band_1))['priority'].iloc[0])

        def process_lte_cellpar():

            cellpar_nokia = pd.read_excel(
                uploaded_file_lte_ciq, sheet_name='CellPar', usecols='J:BQ', header=0, skiprows=3)
            cellpar_nokia = cellpar_nokia.dropna(thresh=3).iloc[:, 1:]
            cellpar_nokia.columns.to_list()
            nrlim_df = cellpar_nokia.iloc[3:, [0, -1]].reset_index(drop=True)
            nrlim_df = nrlim_df.dropna()
            lncel_list = nrlim_df['LNCEL Template Id'].to_list()

            lncel_dict = dict({
                (21, '5Mhz'),
                (22, '10Mhz'),
                (23, '15Mhz'),
                (24, '20Mhz')})
            mbw_dict = dict({
                ('5Mhz', 'mbw25'),
                ('10Mhz', 'mbw50'),
                ('15Mhz', 'mbw75'),
                ('20Mhz', 'mbw100')})
            lncel_list = [int(x) for x in lncel_list]
            llncel_list = [mbw_dict.get(lncel_dict.get(x)) for x in lncel_list]
            nrlim_df['LNCEL Template Id'] = llncel_list
            nrlim_df.columns = ['lncel', 'dlink']
            print(nrlim_df)

            band_list = []
            for row in nrlim_df.itertuples():
                band_list.append(return_band(row.dlink))
            nrlim_df['band'] = band_list
            print(band_list)
            nrlim_df['band'] = band_list
            nrlim_df = nrlim_df[nrlim_df['band'] != 'Del']

            prior_list = []
            mbw_list = []
            for row in nrlim_df.itertuples():
                prior_list.append(return_priority(row.band, row.lncel))
            print(prior_list)
            nrlim_df['priority'] = prior_list
            for row in nrlim_df.itertuples():
                # mbw_list.append(return_mbw(row.lncel))
                mbw_list.append(row.lncel)
            print(mbw_list)
            nrlim_df['lncel'] = mbw_list
            print(nrlim_df.columns.to_list())
            band_group = nrlim_df.groupby('band')

            nrlim0_index_list, nrlim1_index_list = get_nrlim_index(nrlim_df)
            print(int(4 - len(nrlim0_index_list)))
            if len(nrlim0_index_list) > 4:
                nrlim0_index_list = nrlim0_index_list[:int(
                    4 - len(nrlim0_index_list))]
            if len(nrlim1_index_list) > 4:
                nrlim1_index_list = nrlim1_index_list[:int(
                    4 - len(nrlim1_index_list))]
            band_list = ['B2', 'B4', 'B71', 'B12']
            nrlim_0_dict = dict(zip(nrlim0_index_list, band_list))
            nrlim_1_dict = dict(zip(nrlim1_index_list, band_list))

            # Mutate XML

            replace_nrlim_tags(nrlim_df, nrlim_0_dict, nrlim_1_dict)

        def get_tnd_dict(parName):
            mode = tnd_nokia[str(parName)]
            lcrid = tnd_nokia['gNodeB name']
            mode_lst = mode.to_list()
            lcrid_lst = lcrid.to_list()
            comb_list = zip(mode_lst, lcrid_lst)
            # # print(set(comb_list))
            # for mode, lcrid in enumerate(comb_list):
            #     print(mode, lcrid)
            dict = {lcrid: mode for mode, lcrid in comb_list}
            return dict

        def get_bts_name():
            bts_tags = soup.find_all(attrs={"name": "btsName"})
            bts_name = bts_tags[-1].text
            return str(bts_name).lstrip().rstrip()

        def replace_tnd_par1(parName, mf_dict):
            mf_tags = soup.find_all(attrs={"name": str(parName)})
            bts_tags = soup.find_all(attrs={"name": "btsName"})
            bts_name = bts_tags[-1].text
            for mf_tag in mf_tags:
                print(
                    f"parent cubed is-->>>>>>>>>>>>> ..{mf_tag.parent.parent.parent.name}")
                # print(
                # f'printing the managedObject class...{str(mf_tag.find_parents("managedObject"))}')
                soup_m = BeautifulSoup(
                    str(mf_tag.parent.parent.parent))
                print(soup_m.prettify())
                # tag_m = soup_m.managedObject
                # print(f"class is..{tag['distName']}")
                # print("printing the cell number" +
                #   mf_tag.parent['distName'].split('-')[-1])
                # print(mf_dict.get(
                # int(mf_tag.parent['distName'].split('-')[-1])))
                # print('---')
                if mf_tag.parent.name.find('managedObject') > -1:
                    # localIpAddr IPF-1
                    if mf_tag.parent['distName'].find('IPIF-1/IPADDRESSV4-2') > -1:
                        mf_val = mf_dict.get(bts_name.lstrip().rstrip())
                        print(f"bstName is..{bts_name}")
                        print(f"localIpAddr IPF-1  is.. {mf_val}")
                        mf_tag.string = str(mf_val).lstrip().rstrip()

                elif mf_tag.parent.parent.parent['distName'].find('IPNO-1/IPRT-1') > -1:
                    mf_val = mf_dict.get(bts_name.lstrip().rstrip())
                    print(f"bstName is..{bts_name}")
                    print(f"ip address  is.. {mf_val}")
                    mf_tag.string = str(mf_val).lstrip().rstrip()

        def replace_tnd_par2(parName, mf_dict):
            mf_tags = soup.find_all(attrs={"name": str(parName)})
            bts_tags = soup.find_all(attrs={"name": "btsName"})
            bts_name = bts_tags[-1].text
            for mf_tag in mf_tags:
                print(f"parent is-->>>>>>>>>>>>> ..{mf_tag.parent.name}")
        #         print(f'printing the managedObject class...{str(mf_tag.find_parents("managedObject"))}')
        #         soup_m = BeautifulSoup(str(mf_tag.find_parents("managedObject")))
        #         print(soup_m.prettify())
        #         tag_m = soup_m.managedObject
        #         print(f"class is..{tag['distName']}")
        #         print("printing the cell number" + mf_tag.parent['distName'].split('-')[-1])
        #         print(mf_dict.get(int(mf_tag.parent['distName'].split('-')[-1])))
                print('---')
                if mf_tag.parent.name.find('managedObject') > -1:
                    # localIpAddr IPF-2
                    if mf_tag.parent['distName'].find('IPIF-2/IPADDRESSV4-1') > -1:
                        mf_val = mf_dict.get(bts_name.lstrip().rstrip())
                        print(f"bstName is..{bts_name}")
                        print(f"localIpAddr IPF-2  is.. {mf_val}")
                        mf_tag.string = str(mf_val).lstrip().rstrip()
                elif mf_tag.parent.parent.parent['distName'].find('IPNO-1/IPRT-1') > -1:
                    mf_val = mf_dict.get(bts_name.lstrip().rstrip())
                    print(f"bstName is..{bts_name}")
                    print(f"ip address  is.. {mf_val}")
                    mf_tag.string = str(mf_val).lstrip().rstrip()

        def replace_tnd_vlan(parName, mf_dict):
            mf_tags = soup.find_all(attrs={"name": str(parName)})
            bts_tags = soup.find_all(attrs={"name": "btsName"})
            bts_name = bts_tags[-1].text
            for mf_tag in mf_tags:
                print(f"parent is-->>>>>>>>>>>>> ..{mf_tag.parent.name}")
                print('---')
                if mf_tag.parent.name.find('managedObject') > -1:
                    # localIpAddr IPF-2
                    if mf_tag.parent['distName'].find('ETHIF-1/VLANIF-1') > -1:
                        mf_val = mf_dict.get(bts_name.lstrip().rstrip())
                        print(f"bstName is..{bts_name}")
                        print(f"vlan ID  is.. {int(mf_val)}")
                        mf_tag.string = str(int(mf_val))

        # MRBTS ---

        # MRBTS-1841114/EQM-1/APEQM-1
        # MRBTS-1841114/EQM-1/APEQM-1/RMOD-5/PHYANT-1
        # str.replace("is", "was")

        def replace_mrbts_id(mrbts_str, new_id):
            list = mrbts_str.split('/')
            m_list = list[1:]
            print(m_list)
            suffix = '/'.join(m_list)
            print(suffix)
            id_str = list[0]
            new_id_str = id_str.replace(id_str.split('-')[1], str(new_id))
            result_str = new_id_str + '/' + suffix
            print(result_str)
            if result_str[-1] == '/':
                result_str = result_str[:-1]

            return result_str

        def get_mrbts_value(parName):
            par_col = ciq_siteMain[parName]
            print(par_col.to_list())
            print(par_col.to_list()[-1])
            par_str = par_col.to_list()[-1]
        #     bts_str = 'NPH20115B'
            print(par_str)
            return par_str

        def modify_mrbts_tag(attr_name, mrbts_par):
            bts_tags = soup.find_all(attrs={"name": str(attr_name)})
            pattern = re.compile('MRBTS-(\d{7,})')
            # print(type(pattern))
            mrbts_str = re.sub(
                'MRBTS-(\d{7,})', f'MRBTS-{str(get_mrbts_value("mrBtsId")).lstrip().rstrip()}', str(soup))
            # print(pattern.findall(mrbts_str))
            # ''' change all values of MRBTS-
            # '''
            for txt in soup.findAll(text=True):
                if re.search('MRBTS-(\d{7,})', txt, re.I):
                    newtext = re.sub(
                        r'MRBTS-(\d{7,})', f'MRBTS-{str(get_mrbts_value("mrBtsId")).lstrip().rstrip()}', txt)
                    txt.replaceWith(newtext)
            # for txt in soup.findAll(text=True):
            #     if re.search('identi',txt,re.I) and txt.parent.name != 'a':
            #     newtext = re.sub(r'identi(\w+)', r'replace\1', txt.lower())
            #     txt.replaceWith(newtext)
            for bts_tag in bts_tags:
                if str(bts_tag.parent['class']).find('RMOD') > -1:
                    print(f"Modifying {attr_name}..")
                    print(f"printing text..{str(bts_tag.text)}")
                    print(str(get_mrbts_value(mrbts_par)))
                    bts_tag.string = replace_mrbts_id(
                        str(bts_tag.text), str(get_mrbts_value(mrbts_par)))
                    # Also replace all MRBTS ids in managedObjects
                    tagz = soup.find_all('managedObject')
                    for tag in tagz:
                        print(f"-----")
                        print(tag['distName'])
                        tag['distName'] = replace_mrbts_id(
                            tag['distName'], str(get_mrbts_value('mrBtsId')).lstrip().rstrip())
                elif str(bts_tag.parent['class']).find('MRBTS') or str(bts_tag.parent['class']).find('APEQM') > -1:
                    print(f"Modifying {attr_name}..")
                    bts_tag.string = str(get_mrbts_value(
                        mrbts_par)).lstrip().rstrip()

        # NRCELL---

        def get_nrcell_dict(parName):
            mode = ciq_cell_par[str(parName)]
            lcrid = ciq_cell_par['lcrid']
            mode_lst = mode.to_list()
            lcrid_lst = lcrid.to_list()
            comb_list = zip(mode_lst, lcrid_lst)
            # # print(set(comb_list))
            # for mode, lcrid in enumerate(comb_list):
            #     print(mode, lcrid)
            dict = {lcrid: mode for mode, lcrid in comb_list}
            # print(dict)
            print('---')
            print(dict.get(1))
            # print(list(mode))
            mode_set = set(mode.to_list())
            # print(mode_set)
        #     mode_list = mode_set.to_list()
            # print(len(mode_list))
            return dict

        def get_band_dict():
            band_dict = {}

            for key, val in enumerate(freq_list):
                if 600 < int(val) < 700:
                    band_dict.__setitem__(2, val)
                elif 1950 < int(val) < 2399:
                    band_dict.__setitem__(1, val)
                elif 5010 < int(val) < 5179:
                    band_dict.__setitem__(4, val)
                elif 68586 < int(val) < 68935:
                    band_dict.__setitem__(3, val)
            print(band_dict)
            return band_dict

        def get_ho_dict(parName):
            mode = ho_cell_par[str(parName)]
            lcrid = ho_cell_par['lcrid']
            mode_lst = mode.to_list()
            lcrid_lst = lcrid.to_list()
            comb_list = zip(mode_lst, lcrid_lst)
            # # print(set(comb_list))
            # for mode, lcrid in enumerate(comb_list):
            #     print(mode, lcrid)
            dict = {lcrid: mode for mode, lcrid in comb_list}
            # print(dict)
            print('---')
            print(dict.get(1))
            # print(list(mode))
            mode_set = set(mode.to_list())
            # print(mode_set)
        #     mode_list = mode_set.to_list()
            # print(len(mode_list))
            return dict

        def get_physcell_dict(parName):
            mode = ciq_cell_par[str(parName)]
            lcrid = ciq_cell_par['cellName']
            mode_lst = mode.to_list()
            lcrid_lst = lcrid.to_list()
            lcrid_flst = list(filter(lambda k: 'KPH' in k, lcrid_lst))
            comb_list = zip(mode_lst, lcrid_lst)
            dict = {lcrid: mode for mode, lcrid in comb_list}
            res = [dict[i] for i in lcrid_flst if i in dict]
            dict_res = {}
            for k, v in enumerate(res):
                dict_res[k] = v
            print(f"physCell dict is..{dict_res}")

            print('---')
            print(dict.get(1))
            # print(list(mode))
            mode_set = set(mode.to_list())
            # print(mode_set)
        #     mode_list = mode_set.to_list()
            # print(len(mode_list))
            return dict_res

        def get_idle_dict(parName):
            mode = inr_cell_par[str(parName)]
            lcrid = inr_cell_par['freqBandIndicatorNR']
            mode_lst = mode.to_list()
            lcrid_lst = lcrid.to_list()
            comb_list = zip(mode_lst, lcrid_lst)
            # # print(set(comb_list))
            # for mode, lcrid in enumerate(comb_list):
            #     print(mode, lcrid)
            dict = {lcrid: mode for mode, lcrid in comb_list}
            print(f"dlCarrir dict..{dict}")
            print('---')
            print(dict.get(1))
            # print(list(mode))
            mode_set = set(mode.to_list())
            # print(mode_set)
        #     mode_list = mode_set.to_list()
            # print(len(mode_list))
            return dict

        def replace_nrcell_par(parName, mf_dict, physcell_dict={}):
            print(f"inside replace function par name..{parName}")
            print(f"inside replace function mf dict {mf_dict}")
            # print(f"physcell_dict is.. {physcell_dict}")
            mf_tags = soup.find_all(
                attrs={"name": str(parName).rstrip().lstrip()})
            # print(f"--parName .. {str(parName).rstrip().lstrip()}")
            # print(f"--Start{parName}---")
            # print(f"mf tags..{len(mf_tags)}")
            # print(f"--End{parName}---")
            for mf_tag in mf_tags:
                # print(
                # f"parent tag is..{mf_tag.parent.parent.parent.name}")
                # print(f"parent is-->>>>>>>>>>>>> ..{mf_tag.parent.name}")
                #         print(f'printing the managedObject class...{str(mf_tag.find_parents("managedObject"))}')
                #         soup_m = BeautifulSoup(str(mf_tag.find_parents("managedObject")))
                #         print(soup_m.prettify())
                #         tag_m = soup_m.managedObject
                #         print(f"class is..{tag['distName']}")
                #         print("printing the cell number" + mf_tag.parent['distName'].split('-')[-1])
                #         print(mf_dict.get(int(mf_tag.parent['distName'].split('-')[-1])))
                #         print('---')
                if mf_tag.parent.name.find('managedObject') > -1:
                    # print(
                    # f"found managedObject in parent..{mf_tag.parent['class']}")
                    if mf_tag.parent['class'].find('NRCELL') > -1:
                        mf_val = mf_dict.get(
                            int(mf_tag.parent['distName'].split('-')[-1]))
                        # print(f"replacing with value..{mf_val}")
                        mf_tag.string = str(mf_val).lstrip().rstrip()
                    # NRHOIF name=ssbFrequency
                    elif mf_tag.parent['class'].find('NRHOIF') > -1:
                        #                 mf_val = mf_dict.get(int(mf_tag.parent['distName'].split('/')[2].split('-')[-1]))
                        cell_value = re.findall(
                            r'NRCELL-[0-9]+', mf_tag.parent['distName'])[-1].split('-')[1]
                        print(f"cell value is..{cell_value}")
                        mf_val = mf_dict.get(int(cell_value))
                        print(
                            f"the  mf value is.. {str(mf_val).lstrip().rstrip()}")
                        mf_tag.string = str(mf_val).lstrip().rstrip()
                    elif mf_tag.parent['class'].find('NRPLMNSET_NSA') > -1:
                        #                 mf_val = mf_dict.get(int(mf_tag.parent['distName'].split('/')[2].split('-')[-1]))
                        cell_value = re.findall(
                            r'NRCELL-[0-9]+', mf_tag.parent['distName'])[-1].split('-')[1]
                        print(f"cell value is..{cell_value}")
                        mf_val = mf_dict.get(int(cell_value))
                        print(
                            f"the  mf value is.. {str(mf_val).lstrip().rstrip()}")
                        mf_tag.string = str(mf_val).lstrip().rstrip()
        #           name="trackingAreaDN"
                    elif mf_tag.parent['class'].find('NRPLMNSET_SA') > -1:
                        cell_value = re.findall(
                            r'NRCELL-[0-9]+', mf_tag.parent['distName'])[-1].split('-')[1]
                        print(f"cell value is..{cell_value}")
                        mf_val = mf_dict.get(int(cell_value))
                        print(
                            f"the  mf value is.. {str(mf_val).lstrip().rstrip()}")
                        existing_tdn = re.findall(
                            r'TRACKINGAREA-[0-9]+', mf_tag.text)[-1].split('-')[1]
                        print(f"existing tracking area is..{existing_tdn}")
                        print(f"text is..{mf_tag.text}")
                        replacement_text = mf_tag.text.replace(
                            str(existing_tdn), str(mf_val).lstrip().rstrip())
                        print(f"replacement text is .. {replacement_text}")
                        mf_tag.string = str(replacement_text)
                    # fiveGsTac
                    elif mf_tag.parent['class'].find('TRACKINGAREA') > -1:
                        cell_value = 1  # fiveGsTac requirement
                        print(f"cell value is..{cell_value}")
                        mf_val = mf_dict.get(int(cell_value))
                        print(
                            f"the  mf value is.. {str(mf_val).lstrip().rstrip()}")
                        existing_tdn = re.findall(
                            r'TRACKINGAREA-[0-9]+', mf_tag.parent['distName'])[-1].split('-')[1]
                        print(f"existing tracking area is..{existing_tdn}")
                        print(f"text is..{mf_tag.parent['distName']}")
                        replacement_text = mf_tag.parent['distName'].replace(
                            str(existing_tdn), str(mf_val).lstrip().rstrip())
                        print(f"replacement text is .. {replacement_text}")
                        mf_tag.string = str(mf_val).lstrip().rstrip()
                        mf_tag.parent['distName'] = str(replacement_text)
                    # dlCarrierFreq
                    elif mf_tag.parent['class'].find('NRIRFIM') > -1:
                        # find sibling freqBandIndicatorNR
                        # 'a[href*=".com/el"]'
                        freq_band = mf_tag.find_next_sibling("p")
                        cell_value = freq_band.text
                        print(f"cell value is..{cell_value}")
                        mf_val = mf_dict.get(int(cell_value))
                        print(
                            f"the  mf value is.. {str(mf_val).lstrip().rstrip()}")
                        print(f"old text..{mf_tag}")
                        mf_tag.string = str(mf_val).lstrip().rstrip()
                        print(f"changed text..{mf_tag}")
                    elif mf_tag.parent['class'].find('NRREDRT') > -1:
                        # find sibling redirPrio
                        redirPrio = mf_tag.find_next_sibling("p")
                        cell_value = redirPrio.text
                        print(f"cell value is..{cell_value}")
                        mf_val = mf_dict.get(int(cell_value))
                        print(
                            f"the  mf value is.. {str(mf_val).lstrip().rstrip()}")
                        print(
                            f"lenght of mf_val is..{type(mf_val)}")
                        print(f"old text..{mf_tag}")
                        if not str(mf_val).find('None') > -1:
                            mf_tag.string = str(mf_val).lstrip().rstrip()
                            print(f"changing text..{mf_tag}")

                elif mf_tag.parent.parent.parent['class'].find('NRIAFIM') > -1:
                    list_physcell = mf_tag.parent.parent.find_all(
                        "p", attrs={"name": "physCellId"})
                    print(type(list_physcell))
                    for index, tagg in enumerate(list_physcell):
                        print(f"index is..{index}")
                        print(f"tag is..{tagg.text}")
                        print(physcell_dict.get(index))
                        tagg.string = str(physcell_dict.get(index))

        def nrcell_modify(key, value):
            if value.find('dlCarrierFreq') > -1:
                pci_dict = get_idle_dict(str(value))
                print(f"pMax Dict..{pci_dict}")
            elif value.find('ssbFrequency') > -1:
                pci_dict = get_ho_dict(str(value))
                print(f"pMax Dict..{pci_dict}")
            elif value.find('redirFreqEutra') > -1:
                pci_dict = get_band_dict()
                print(f"pMax Dict..{pci_dict}")
            else:
                pci_dict = get_nrcell_dict(str(value))

            replace_nrcell_par(str(key), pci_dict,
                               get_physcell_dict('physCellId'))

        def process_tnd_pars():
            tnd_dict = get_tnd_dict('CORENET Default  Gateway (s1,x2,U,C)')
            print(tnd_dict.get('NPH20115B'))

            replace_tnd_par1('gateway', tnd_dict)

            col18_dict = get_tnd_dict(
                "gNodeB's user plane IP address   (s1,x2)")
            print(col18_dict.get('NPH20115B'))
            replace_tnd_par1('localIpAddr', col18_dict)

            col14_dict = get_tnd_dict("gNodeB OAM IP address")
            print(col14_dict.get('NPH20115B'))
            replace_tnd_par2('localIpAddr', col14_dict)

            vlan_dict = get_tnd_dict(
                "CORENET AAV Ethernet VLAN ID (s1,x2,U,C)")
            print(vlan_dict.get('NPH20115B'))
            replace_tnd_vlan('vlanId', vlan_dict)

        def remove_blank_spaces():
            tags = soup.find_all('p')
            for tag in tags:
                # print(tag.string)
                tag.string = str(tag.text).lstrip().rstrip()

        nrcell_par_dict = {
            'cellName': 'cellName',
            'gscn': 'gscn',
            'dlMimoMode': 'dlMimoMode',
            'msg1FrequencyStart': 'Msg1FrequencyStart',
            'pMax': 'pMax',
            'prachConfigurationIndex': 'prachConfigurationIndex',
            'prachRootSequenceIndex': 'prachRootSequenceIndex',
            'zeroCorrelationZoneConfig': 'zeroCorrelationZoneConfig',
            'configuredEpsTac': 'configuredEpsTac',
            'type0CoresetConfigurationIndex': 'type0CoresetConfigurationIndex',
            'physCellId': 'physCellId',
            'nrarfcnDl': 'nrarfcnDl',
            'nrarfcnUl': 'nrarfcnUl',
            'nrarfcn': 'nrarfcn',
            'fiveGsTac': 'fiveGsTac',
            'trackingAreaDN': 'fiveGsTac',
            'dlCarrierFreq': 'dlCarrierFreq',
            'ssbFrequency': 'ssbFrequency',
            'redirFreqEutra': 'redirFreqEutra'
        }
        mrbts_par_dict = {
            'location': 'moduleLocation',
            'btsName': 'btsName',
            'radioMasterDN': 'mrBtsId',
            # 'secondEndpointDN': 'mrBtsId',
            # 'firstEndpointDN': 'mrBtsId',
        }

        key_list = [key for key in nrcell_par_dict]
        mrbts_key_list = [key for key in mrbts_par_dict]
        print(key_list)

        submitted = st.form_submit_button("Process XML")
        if submitted:
            process_xml()

    if st.session_state['download']:

        # or xml.dom.minidom.parseString(xml_string)
        # print(soup.prettify())
        dom = xml.dom.minidom.parseString(str(soup))
        # print(f"minidom..{str(dom.toxml())}")
        pretty_xml_as_string = dom.toprettyxml()
        # pretty_xml_as_string = str(soup.prettify(formatter=None))
        # print(f"pretty xml...{pretty_xml_as_string}")
        st.download_button(label='📥 Download XML ',
                           data=pretty_xml_as_string,
                           file_name=f'{str(get_mrbts_value("mrBtsId"))}-{get_bts_name()}-{option}.xml')


app()
