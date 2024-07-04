import streamlit as st
import pandas as pd
import pyreadstat
import tempfile
from io import BytesIO

from io import StringIO 

import numpy as np



#update 2023.12.30 wegen Fehlermeldung
#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report

#2023.06.18 pygwalker visualization library
import pygwalker as pyg

#These are the visualization libraries. Matplotlib is standard and is what most people use.
#Seaborn works on top of matplotlib, as we mentioned in the course.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



st.set_page_config(page_title='SPSS Viewer',layout="wide")

st.title("SPSS File Viewer")
st.info("Here you can view SPSS-Files with/without labels and also export them to Excel and do some basic statistical testing and tabulation")
st.warning("Unfortunatlely this app crashes kinda often, i think due to data usage limits")

col_names_labels_df = pd.DataFrame()

# File upload widget
file = st.sidebar.file_uploader("Upload SPSS file", type=[".sav"])



st.sidebar.write("")
latinEncoding= st.sidebar.checkbox("Deactivate latin-1 decoding if you get an error message. For me latin-1 has worked better, therefore it's set as default", value=True)
st.write("")

#if file is None:
#    st.info("It may take some time to load and convert the SPSS-File, depending on the size of the dataset")

if file is not None:
    # Convert SPSS file to dataframe
    try:
        # Save file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())

        # Use pyreadstat to open and read SPSS file
        if latinEncoding == True:
            data, meta = pyreadstat.read_sav(tmp_file.name, encoding = "latin1")
        if latinEncoding == False:
            data, meta = pyreadstat.read_sav(tmp_file.name)

        # Extract dataframe from pyreadstat output
        df = pd.DataFrame(data)
        
        labelledData = df.copy()
        rawData = df.copy()

        # Checkbox to allow renaming columns with variable labels ############################


        st.write("")
        rename_columns = st.checkbox("Rename column names with labels \n (Attention - as of now there have to be Variable Labels in SPSS to all Variables!")


        st.write("")
        dropEmptyColumns= st.checkbox("Drop all columns that only contain Nan or None Values - helps if renaming does not work")
        if dropEmptyColumns:
            labelledData = labelledData.dropna(axis=1, how='all')
            rawData = rawData.dropna(axis=1, how='all')




        st.write("")
        st.write("")


        st.write("")
        st.write("")



 ########################################## RawData File ##############################################################################################################################


        rawDataExpander = st.expander("Show & save Raw Data?")  ############################
        with rawDataExpander:
            st.write("## Raw Data without labels")


            if rename_columns:
                st.info("Datafile with renamed columns")
                
                 # Replace column names with variable labels if requested

                # Extract variable labels from metadata
                column_labels = meta.column_labels

                # Extract column names to labels dictionary
                column_names_to_labels = meta.column_names_to_labels

                # Convert to DataFrame
                col_names_labels_df = pd.DataFrame(column_names_to_labels.items(),
                                                columns=['Column Name', 'Variable Label'])
                col_names_labels_df['Zeilennummer'] = col_names_labels_df.index.astype(str)

                # st.write("## Dataframe - Column Names to Labels")
                # st.write(col_names_labels_df)

                # Create a new column in col_names_labels_df with variable labels formatted as specified
                col_names_labels_df['VariableLabelFormatted'] = col_names_labels_df['Variable Label'].str.replace(' ', '_')
                col_names_labels_df['VariableLabelUnique'] = col_names_labels_df['Zeilennummer'] + '_' + \
                                                            col_names_labels_df['VariableLabelFormatted']

                # Rename columns in the df DataFrame using VariableLabelFormatted values
                for i, row in col_names_labels_df.iterrows():
                    if row['Column Name'] in rawData.columns:
                        rawData.rename(columns={row['Column Name']: row['VariableLabelUnique']}, inplace=True)









            st.write(rawData)

            #st.info("columns with numeric data types")
            #numeric_df = rawData.select_dtypes(include='number')
            #st.write(numeric_df)



            def to_excel(rawData):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                rawData.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
                writer.close()
                processed_data = output.getvalue()
                return processed_data

            st.write("")
            df_xlsx = to_excel(rawData)
            st.download_button(label='📥 Export Rawdata to Excel?',
                               data=df_xlsx,
                               file_name='SPSSRawDataToExcel.xlsx')


            
            if st.checkbox("Show Column Data Types?"):
                st.write("Column Data Types")
                st.write(rawData.dtypes)

            if st.checkbox("Show Variable descriptions (Max/Min/Mean/Count Values)?"):
                st.write("Description of Variables")
                st.write(rawData.describe())





            # Checkbox for statistical profile reporting ############################
            st.write("")
            st.write("")
            st.write("")
            statisticalTestsRawData = st.checkbox("Perform statistical tests?", key='RawdataTests')
            st.write("")
            st.write("")
            st.write("")
            # Replace column names with variable labels if requested
            if statisticalTestsRawData:
 
               my_korrelationsVariablenSelect = st.multiselect("Choose variables for tests",rawData.columns.tolist())
               df_statistischeTestrawData = rawData[my_korrelationsVariablenSelect]

               st.write("")

               if st.checkbox("Create simple Chart?"):
               #bygga in en chartmodul här?
                   #st.write(df_statistischeTestrawData)
                   averages = df_statistischeTestrawData[my_korrelationsVariablenSelect].mean()
                   #st.write(averages)

                   chart_type = st.radio('Select chart type', ('Horizontal Bar Chart', 'Vertical Line Chart'))

                   st.set_option('deprecation.showPyplotGlobalUse', False) #wegen Fehlermeldung



                   chart_data = pd.DataFrame({'Columns': my_korrelationsVariablenSelect, 'Averages': averages})
                   if chart_type == 'Horizontal Bar Chart':
                       plt.barh(chart_data['Columns'], chart_data['Averages'])
                       plt.xlabel('Columns')
                       plt.ylabel('Average Value')
                       plt.title('Average Values of Selected Columns')
                       for i, avg in enumerate(averages):
                           plt.text(avg, i, f'{avg:.2f}', ha='left', va='center')
                       st.pyplot()

                   elif chart_type == 'Vertical Line Chart':
                    #vertikal line chart
                       plt.plot(chart_data['Averages'], chart_data['Columns'], marker='o')
                       plt.xlabel('Columns')
                       plt.ylabel('Average Value')
                       plt.title('Average Values of Selected Columns')
                       plt.xticks(rotation=45)
                       st.pyplot()

               st.write("")

               if len(my_korrelationsVariablenSelect)>0:
  
                   if st.checkbox("Show descriptive Info?"):
                       st.write(df_statistischeTestrawData.describe())
                       st.write("")
                       st.write(df_statistischeTestrawData.info())
                       st.write("")

                   st.write("")

                   if st.checkbox("Show Pearson correlation coefficients?"):                 
                     # Compute Pearson correlation coefficient for the features in our data set.
                        # The correlation method in pandas, it has the Pearson correlation set as default.
                    st.write(df_statistischeTestrawData.corr())
                    
                    df_korr = df_statistischeTestrawData.corr()
                    df_korr['Variable'] = df_korr.index


                    # move column 'B' to the leftmost position
                    col_name = 'Variable'
                    col_pos = 0
                    df_korr.insert(col_pos, col_name, df_korr.pop(col_name))
                    
                    st.write("Correlation Heatmap")
                    fig, ax = plt.subplots()
                    sns.heatmap(df_statistischeTestrawData.corr(),annot=False,cmap='RdBu')
                    plt.title('Correlation Heatmap',fontsize=8)
                    st.write(fig)


                   st.write("")
                   _="""
                   if st.button("Show Profile-Reporting?"):
  
                        st.write("ProfileReport:")
                        profile = ProfileReport(df_statistischeTestrawData)
                        st_profile_report(profile)
    
    
                        export=profile.to_html()
                        st.download_button(label="Download Profile Report?", data=export, file_name='report.html')
                    """



        st.write("")
        st.write('---')
        st.write("")
        
        
 ########################################## Data with labeled Values ######################################################################################################
        

        LabelledDataExpander = st.expander("Show & save Data with labeled Values?") ############################
        with LabelledDataExpander:
            st.write("## Data with Labels")
            
             # Replace values with value labels
            for var in meta.variable_value_labels:
                if var in labelledData.columns:
                    value_labels = meta.variable_value_labels[var]
                    labelledData[var] = labelledData[var].replace(value_labels)   
    
            if rename_columns:
                st.info("Datafile with renamed columns")
                            # Extract variable labels from metadata
                column_labels = meta.column_labels

                # Extract column names to labels dictionary
                column_names_to_labels = meta.column_names_to_labels

                # Convert to DataFrame
                col_names_labels_df = pd.DataFrame(column_names_to_labels.items(),
                                                columns=['Column Name', 'Variable Label'])
                col_names_labels_df['Zeilennummer'] = col_names_labels_df.index.astype(str)

                # st.write("## Dataframe - Column Names to Labels")
                # st.write(col_names_labels_df)

                # Create a new column in col_names_labels_df with variable labels formatted as specified
                col_names_labels_df['VariableLabelFormatted'] = col_names_labels_df['Variable Label'].str.replace(' ', '_')
                col_names_labels_df['VariableLabelUnique'] = col_names_labels_df['Zeilennummer'] + '_' + \
                                                            col_names_labels_df['VariableLabelFormatted']

                # Rename columns in the df DataFrame using VariableLabelFormatted values
                for i, row in col_names_labels_df.iterrows():
                    if row['Column Name'] in labelledData.columns:
                        labelledData.rename(columns={row['Column Name']: row['VariableLabelUnique']}, inplace=True)  
    

            #Filer  Data set
            filterLabelledDataColumns = st.checkbox("Filter Dataset", key="filterLabelledDataColumns")
            unique_values = {}
            
            if filterLabelledDataColumns:
            # Spaltenauswahl
                selected_columns = st.multiselect("Spalten auswählen", options=labelledData.columns.tolist())

                if selected_columns:
                # Eindeutige Werte in ausgewählten Spalten ermitteln
                    unique_values = {}
                for column in selected_columns:
                    unique_values[column] = labelledData[column].unique()

                # Multiselect-Boxen für eindeutige Werte erstellen
                selected_values = {}
                for column, values in unique_values.items():
                    selected_values[column] = st.multiselect(f"Auswahl für {column}", options=values)

                if any(selected_values.values()):
                # DataFrame basierend auf den ausgewählten Werten filtern
                    filtered_df = labelledData.copy()
                for column, values in selected_values.items():
                    if values:
                        labelledData = filtered_df[filtered_df[column].isin(values)]

                # Gefiltertes DataFrame anzeigen
                #st.dataframe(filtered_df)

            st.write(labelledData)
    
            if st.checkbox("Show Column Data Types of labelled data?", key="labelledData.dtypes"):
                st.write(labelledData.dtypes)

            def to_excel(labelledData):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                labelledData.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
                writer.close()
                processed_data = output.getvalue()
                return processed_data


            df_xlsx = to_excel(labelledData)
            st.download_button(label='📥 Export Date with labeled Values to Excel?',
                               data=df_xlsx,
                               file_name='SPSSLabelledDataToExcel.xlsx')

        
    
                # Checkbox for statistical profile reporting ############################
            st.write("")
            st.write("")
            st.write("")
            statisticalTests = st.checkbox("Show statistical info (descriptive Info, Profile-Reporting)?",key='statTestLabeledData')
            st.write("")
            st.write("")
            st.write("")
            # Replace column names with variable labels if requested
            if statisticalTests:
 
               my_korrelationsVariablenSelect = st.multiselect("Choose a minimum of 2 labelled variables",labelledData.columns.tolist(), key='LabeledData')
               df_statistischeTestLabeledData = labelledData[my_korrelationsVariablenSelect]
               
               if len(my_korrelationsVariablenSelect)>1:
                    
                if st.checkbox("Show descriptive Info of the chosen labeled variables?"):
                       #st.write(df_statistischeTestLabeledData.describe())

                       st.write(df_statistischeTestLabeledData.describe(include=np.object))

                       st.write("")
                       st.write(df_statistischeTestLabeledData.info())
                       st.write("")                
                   
                #if st.checkbox("Show Pearson correlation coefficients?"):                 
                     # Compute Pearson correlation coefficient for the features in our data set.
                        # The correlation method in pandas, it has the Pearson correlation set as default.
                   # st.write(df_statistischeTestLabeledData.corr())                  
                   
                   
                   
                _="""   
                if st.button("Show Profile-Reporting?", key='profileReporLabeledeDataReport'):
     
                    st.write("ProfileReport:")
                    profile = ProfileReport(df_statistischeTestLabeledData)
                    st_profile_report(profile)
                    
                    export=profile.to_html()
                    st.download_button(label="Download Profile Report", data=export, file_name='report.html')
                """




        ########################################## Combined Data -   merge raw (categoriel data) and labelled data  ##############################################################################################################################
        st.write("")
        st.write('---')
        st.write("")

        if len(rawData)>1 and len(labelledData)>1:


            MergedDataExpander = st.expander("Combined datasets") ############################
            with MergedDataExpander:

                # Load the data frames
                df1 = rawData
                df2 = labelledData

                # Get the column names
                numeric_cols = df1.columns.tolist()
                categorical_cols = df2.columns.tolist()

                # Add prefixes to column names to avoid conflicts
                df1_prefixed = df1.add_prefix('numeric_')
                df2_prefixed = df2.add_prefix('categorical_')

                #st.write(df1_prefixed)
                #st.write(df2_prefixed)
                st.subheader("Combined Dataset containing all categorical and numerical Variables")
                #mergeAll = st.button("Merge all columns with categorical and numerical data?")
                mergeAll = True
                mergeAll_df = pd.DataFrame()
                if mergeAll:

                    _="""
                    mergeAll_df = df2_prefixed.merge(
                        df1_prefixed,
                        left_on='categorical_Participant',
                        right_on='numeric_Participant',
                        # You can choose 'inner', 'outer', 'left', or 'right' depending on your requirements
                    )
                    """

                    mergeAll_df = pd.merge(df2_prefixed, df1_prefixed, left_index=True, right_index=True)



                # Display the merged data frame
                if len(mergeAll_df)>0:
                    #st.write("## Combined dataset with all variables")

                    st.write(mergeAll_df)
                    #st.write(mergeAll_df.describe())




                if len(mergeAll_df)>0:

                    def to_excel(mergeAll_df):
                        output = BytesIO()
                        writer = pd.ExcelWriter(output, engine='xlsxwriter')
                        mergeAll_df.to_excel(writer, index=True, sheet_name='Sheet1')
                        workbook = writer.book
                        worksheet = writer.sheets['Sheet1']
                        format1 = workbook.add_format({'num_format': '0.00'})
                        worksheet.set_column('A:A', None, format1)
                        writer.close()
                        processed_data = output.getvalue()
                        return processed_data


                    df_xlsx = to_excel(mergeAll_df)
                    st.download_button(label='📥 Export combined dataset to Excel?',
                                       data=df_xlsx,
                                       file_name='SPSSallCombinedColumnsToExcel.xlsx')







                # Let the user select columns to merge #####################
                st.divider()
                st.subheader("")
                st.subheader("Create a new dataset of chosen Variables")
                selected_categorical_cols = st.multiselect('Select catgorical variables (men/women, old/young..):', categorical_cols)
                st.write("")
                st.write("")
                selected_numeric_cols = st.multiselect('Select numeric variables (brand awareness 0-100, sympathy 1-7,...) :', numeric_cols)
                st.write("")
                st.write("")

                # Merge the selected columns
                merged_df = pd.DataFrame()

                # Merge categorical columns
                for col in selected_categorical_cols:
                    merged_df[col] = df2_prefixed['categorical_' + col]



                for col in selected_numeric_cols:
                    merged_df[col] = df1_prefixed['numeric_' + col]


                st.write("")
                st.write("")

                # Recode Variables? #######################

                if st.checkbox("Select Variables to recode while changing the original variables?"):
                    selected_RecodeColumns = st.multiselect('Select variables to recode', merged_df.columns)
                    if selected_RecodeColumns:
                        recode_data = {}
                        keyNr = 0
                        # Recode values
                        for column in selected_RecodeColumns:
                            unique_values = merged_df[column].unique()
                            st.info(f"Recode {column}")
                            recode_map = {}
                            for value in unique_values:
                                keyNr = keyNr+1
                                new_value = st.text_input(f"Insert new value for {value}", value, key=f"new_valueRecode{value}"+str(keyNr))
                                recode_map[value] = new_value
                            recode_data[column] = recode_map

                        recode_button = st.checkbox(":point_right: :orange[Now recode these values and replace the original variables!]", key="recode_button")
                        if recode_button:
                            #st.subheader("Recoded Dataset")
                            #recoded_df = merged_df.copy()
                            for column, recode_map in recode_data.items():
                                merged_df[column] = merged_df[column].map(recode_map)



                st.write("")
                st.write("")

                if st.checkbox("Select Variables to recode into new Variables?"):
                    selected_RecodeToNewColumns = st.multiselect('Select the variables to recode into new Variables', merged_df.columns, key="selected_RecodeToNewColumns")
                    if selected_RecodeToNewColumns:
                        recode_data = {}
                        # Recode values
                        for column in selected_RecodeToNewColumns:
                            unique_values = merged_df[column].unique()
                            st.info(f"Recode {column}")
                            recode_map = {}
                            for value in unique_values:
                                new_value = st.text_input(f"New value for {value}", value, key=f"new_valueRecodeToNew{value}")
                                recode_map[value] = new_value
                            recode_data[column] = recode_map

                        recodetoNewVariable_button = st.checkbox(":point_right: :orange[Now Recode these values into new variables!]", key="recodetoNewVariable_button")
                        if recodetoNewVariable_button:
                            #st.subheader("Recoded Dataset with new recodes Variables")
                            #recoded_df = merged_df.copy()
                            for column, recode_map in recode_data.items():
                                merged_df[column + "_recoded"] = merged_df[column].map(recode_map)




                st.write("")
                st.write("")

                # Display the merged data frame
                if len(merged_df)>0:
                    st.write()




                #Filter  Data set
                filterMergedDataColumns = st.checkbox("Filter Dataset", key="filterMergedDataColumns")
                unique_values = {}
                
                if filterMergedDataColumns:
                # Spaltenauswahl
                    selected_columns = st.multiselect("Spalten auswählen", options=merged_df.columns.tolist(), key="mergedDFcolumns")

                    if selected_columns:
                    # Eindeutige Werte in ausgewählten Spalten ermitteln
                        unique_values = {}
                    for column in selected_columns:
                        unique_values[column] = merged_df[column].unique()

                    # Multiselect-Boxen für eindeutige Werte erstellen
                    selected_values = {}
                    for column, values in unique_values.items():
                        selected_values[column] = st.multiselect(f"Auswahl für {column}", options=values)

                    if any(selected_values.values()):
                    # DataFrame basierend auf den ausgewählten Werten filtern
                        filtered_merged_df = merged_df.copy()
                    for column, values in selected_values.items():
                        if values:
                            merged_df = filtered_merged_df[filtered_merged_df[column].isin(values)]

                    # Gefiltertes DataFrame anzeigen





                    #st.subheader("Dataset with selected columns (merged_df):")
                    #st.dataframe(merged_df)




                if len(merged_df)>0:
                    merged_df = st.data_editor(merged_df, num_rows="dynamic")

                    st.write("")
                    st.write("")

                    def to_excel(merged_df):
                        output = BytesIO()
                        writer = pd.ExcelWriter(output, engine='xlsxwriter')
                        merged_df.to_excel(writer, index=True, sheet_name='Sheet1')
                        workbook = writer.book
                        worksheet = writer.sheets['Sheet1']
                        format1 = workbook.add_format({'num_format': '0.00'})
                        worksheet.set_column('A:A', None, format1)
                        writer.close()
                        processed_data = output.getvalue()
                        return processed_data


                    df_xlsx = to_excel(merged_df)
                    st.download_button(label='📥 Export Table with the selected columns to Excel?',
                                       data=df_xlsx,
                                       file_name='SPSSselectedColumnsToExcel.xlsx')


                    st.write("")

                if len(merged_df)>0:
                    if st.checkbox("Show Column Data Types?", key="merged_df.dtypes"):
                        st.write(merged_df.dtypes)

                if len(merged_df)>0:
                    st.write("")
                    if st.checkbox("Explore the dataset visually?"):
                        def load_config(file_path):
                            with open(file_path, 'r') as config_file:
                                config_str = config_file.read()
                            return config_str


                            #config = load_config('config.json') pyg config laden

                            #pyg.walk(merged_df, env='Streamlit', dark='dark', spec=config)
                            pyg.walk(merged_df, env='Streamlit', dark='dark')

                st.write("")
                st.write("")









                #Tabellen mit Häufigkeiten und Prozenten #########################################


                prozenteAnzahl_GesamtTabelle = pd.DataFrame()

                if st.checkbox("Show frequencies and percentages of values for every chosen variable"):

                    st.subheader("Separate tables for every variable:")
                    st.info("Sum per Variable is 100%")

                    prozente_anzahl_df = pd.DataFrame()
                    for column in merged_df.columns[0:]:
                        
                        prozente_df = (merged_df[column].value_counts(normalize=True).reset_index())
                        prozente_df.columns.values[0] = "Label"
                        prozente_df.rename(columns={prozente_df.columns[1]: 'Percentage'}, inplace=True)
                        #prozente_df['Variable'] = column
                        prozente_df.insert(0, 'Variable', column)

                        anzahl_df = (merged_df[column].value_counts().reset_index())
                        anzahl_df.rename(columns={anzahl_df.columns[1]: 'Anzahl'}, inplace=True)
                        #st.write(anzahl_df)
                        prozente_df['Cases'] = anzahl_df.Anzahl


                        prozenteAnzahl_GesamtTabelle = pd.concat([prozente_anzahl_df, prozente_df], axis=1)

                        st.write(column)
                        st.write(prozente_df)

                st.write("")
                st.write("")
                #st.subheader("All column-percentages and frequencies of the selected variables in one Table:")
                #st.info("Sum per Variable is 100%")
                #st.write(prozenteAnzahl_GesamtTabelle)

                _="""
                def to_excel(prozenteAnzahl_GesamtTabelle):
                        output = BytesIO()
                        writer = pd.ExcelWriter(output, engine='xlsxwriter')
                        prozenteAnzahl_GesamtTabelle.to_excel(writer, index=False, sheet_name='Sheet1')
                        workbook = writer.book
                        worksheet = writer.sheets['Sheet1']
                        format1 = workbook.add_format({'num_format': '0.00'})
                        worksheet.set_column('A:A', None, format1)
                        writer.close()
                        processed_data = output.getvalue()
                        return processed_data


                df_xlsx = to_excel(prozenteAnzahl_GesamtTabelle)
                st.download_button(label='📥 Export Table with all percentages and frequencies to Excel?',
                                       data=df_xlsx,
                                       file_name='SPSSFrequencyPercentageTableToExcel.xlsx')


                st.write("")
                st.write("")

                """


                    #dataframe mit den häufigkeiten der Kombinationen ####################
                AlleKombinationenProzent = merged_df[selected_categorical_cols].value_counts(normalize=True).reset_index()
                    #AlleKombinationenProzent.columns.values[0] = "Label"
                    #AlleKombinationenProzent.rename(columns={AlleKombinationenProzent.columns[1]: 'Percentage'}, inplace=True)
                st.write("Occurence of combinations of the categorical variables")
                st.dataframe(AlleKombinationenProzent)


                _="""
                if st.checkbox("Show labelling/unique values?"):
                    anzahlVariablen = len(selected_numeric_cols) + len(selected_categorical_cols)
                    st.write("Anzahl Variablen: ",anzahlVariablen)

                    st.write("selected_numeric_cols:", selected_numeric_cols)
                    st.write("selected_categorical_cols:", selected_categorical_cols)

                    st.markdown("#### Selected Variables and their Labels/Values:")

                    cols = st.columns(anzahlVariablen)
                    
                    VariablenKolumnenAuswahl = merged_df.columns.values.tolist()
                    #st.write("VariablenKolumnenAuswahl: ",VariablenKolumnenAuswahl)



                    dict_of_merged_df= {} # initialize empty dictionary

                    st.info("None's are not deleted automatically, can cause problems/error message. I'll add functionality here if/when i figure out how")
                    if st.checkbox("Delete Nones"):
                        merged_df = merged_df.dropna(axis = 0, how ='any') 

                    for i in range(anzahlVariablen):
                        col = cols[i%anzahlVariablen]
                        AuspraegungenAuswahlEinzeln = col.multiselect(f"" + VariablenKolumnenAuswahl[i],
                        options=merged_df[VariablenKolumnenAuswahl[i]].unique(),
                        default = merged_df[VariablenKolumnenAuswahl[i]].unique(),
                        key=i
                        )

                    AnzahlKategorischeVariablen = len(selected_categorical_cols)
                    for k in range(AnzahlKategorischeVariablen):
                        KategorienListe = merged_df[VariablenKolumnenAuswahl[k]].unique()



                        st.write("KategorienListe: ",KategorienListe)

                        #st.write("i: ",i)
                        #col.text_input('Ausprägung', key=i)
                        #Dataframe je Variable mit den Ausprägungen bildenm ev können wir hier dann ein Labeling machen?
                        
                        dict_of_merged_df["merged_df_{}".format(i)] = merged_df[VariablenKolumnenAuswahl[i]].unique()
                        #col.write(dict_of_merged_df["merged_df_{}".format(i)])
                """
                        

                if st.checkbox("Show descriptive Infos?"):
                    st.write(merged_df.describe())
                    st.write("")
                    st.write("")



                ################### mehrfachantwortensets####################################
                if st.checkbox("Create Multiresponse-sets"):

                    merged_df_KatVariablenMehrfach = labelledData
                    categorical_cols_forCrossTable = merged_df_KatVariablenMehrfach.columns.tolist()

                    st.write("")
                    selected_categorical_cols_forMultiResponseSet = st.multiselect('Select variables for the multiresponse-set:',categorical_cols_forCrossTable, key="selected_categorical_cols_forMultiResponseSet")
                    if len(selected_categorical_cols_forMultiResponseSet)>1:
                        # Daten in ein Pandas DataFrame laden
                        df_MultiresponseSet = pd.DataFrame(merged_df_KatVariablenMehrfach, columns=selected_categorical_cols_forMultiResponseSet)
                        st.write("df_MultiresponseSet:", df_MultiresponseSet)

                        # marken zählen
                        counts = {}
                        total_responses = 0
                        for column in df_MultiresponseSet.columns:
                            for brand in df_MultiresponseSet[column]:
                                if pd.notna(brand):
                                #originalcode if pd.notna(brand) and brand.strip() != '':
                                    if brand in counts:
                                        counts[brand] += 1
                                    else:
                                        counts[brand] = 1
                                    total_responses += 1

                        # Ergebnisse als Tabelle anzeigen
                        MultiresponseSetresult_df = pd.DataFrame(list(counts.items()), columns=['Values', 'Anzahl'])
                        MultiresponseSetresult_df = MultiresponseSetresult_df.sort_values('Anzahl', ascending=False)

                        # Prozentwerte berechnen
                        MultiresponseSetresult_df['% Befragte'] = (MultiresponseSetresult_df['Anzahl'] / len(df)) * 100
                        MultiresponseSetresult_df['% Antworten'] = (MultiresponseSetresult_df['Anzahl'] / total_responses) * 100

                        st.subheader("Multiresponse-Set - Values and Percentages:")
                        st.write(MultiresponseSetresult_df)



                #st.write(merged_df.dtypes)

                st.write("")
                st.write("")

                ################### cross tabulations ####################################
                if st.checkbox("Create cross-tabulations?"):
                    st.subheader("Cross Tables with Average Values - Beta")

                    # Create multiselect widgets for object and float variables
                    selected_object_vars = selected_categorical_cols
                    selected_float_vars = selected_numeric_cols

                    # Generate cross table with average values
                    if selected_object_vars and selected_float_vars:

                        st.info("         ")
                        st.write("Count of cases: ",len(merged_df))

                        # Group by selected object variables and calculate average values for selected float variables

                        ThomasFormatiertesDataframe = pd.DataFrame(columns=['KatVariable'] + list(selected_float_vars))
                        AnzahlKategorischeVariablen = len(selected_categorical_cols)
                        for t in range(AnzahlKategorischeVariablen):
                            Thomasgrouped_df = merged_df.groupby(selected_object_vars[t])[selected_float_vars].mean().reset_index()
                            Thomasgrouped_df.columns.values[0] = "KatVariable"
                            ThomasFormatiertesDataframe = pd.concat([ThomasFormatiertesDataframe, Thomasgrouped_df])
                            ThomasFormatiertesDataframe = ThomasFormatiertesDataframe.reset_index(drop=True)
                            ThomasFormatiertesDataframe.index = ThomasFormatiertesDataframe['KatVariable']
                            TransposedDataframe = ThomasFormatiertesDataframe.T
                            TransposedDataframe = TransposedDataframe.drop(TransposedDataframe.index[0])


                        st.write("")
                        st.write("Table with average values - categories in the columns:  ", TransposedDataframe)

                        def to_excel(TransposedDataframe):
                            output = BytesIO()
                            writer = pd.ExcelWriter(output, engine='xlsxwriter')
                            TransposedDataframe.to_excel(writer, index=True, sheet_name='Sheet1')
                            workbook = writer.book
                            worksheet = writer.sheets['Sheet1']
                            format1 = workbook.add_format({'num_format': '0.00'})
                            worksheet.set_column('A:A', None, format1)
                            writer.close()
                            processed_data = output.getvalue()
                            return processed_data


                        df_xlsx = to_excel(TransposedDataframe)
                        st.download_button(label='📥 Export Table with categories in the columns to Excel?',
                                           data=df_xlsx,
                               file_name='SPSSCrossTableToExcel.xlsx')

                        st.write("")
                        st.write("")
                        ThomasFormatiertesDataframe.drop(['KatVariable'], axis=1, inplace=True)
                        st.write("Table with average values - categories in the rows: ",ThomasFormatiertesDataframe)


                        def to_excel(ThomasFormatiertesDataframe):
                            output = BytesIO()
                            writer = pd.ExcelWriter(output, engine='xlsxwriter')
                            ThomasFormatiertesDataframe.to_excel(writer, index=True, sheet_name='Sheet1')
                            workbook = writer.book
                            worksheet = writer.sheets['Sheet1']
                            format1 = workbook.add_format({'num_format': '0.00'})
                            worksheet.set_column('A:A', None, format1)
                            writer.close()
                            processed_data = output.getvalue()
                            return processed_data


                        df_xlsx = to_excel(ThomasFormatiertesDataframe)
                        st.download_button(label='📥 Export Table with categories in the rows to Excel?',
                                           data=df_xlsx,
                               file_name='SPSSCrossTTableRowCategoriesToExcel.xlsx')



                    st.write("")
                    st.write("")


                #Tabellen mit Haeufigkeiten und Prozenten  -  mit Kreuztabellen#########################################
                #Kat Variablen werden zu Spalten #############
                    # Generate cross table with average values

                    st.subheader("Crosstable with column percentages of chosen variables")
                    st.info("All variables are treated as categorical values here")

                    merged_df_KatVariablen = labelledData
                    categorical_cols_forCrossTable = merged_df_KatVariablen.columns.tolist()

                    st.write("")
                    selected_categorical_cols_forCrossTable = st.multiselect('Select variables (men/women, old/young..):',categorical_cols_forCrossTable, default = selected_categorical_cols,key="selected_categorical_cols_forCrossTable")


                    if selected_categorical_cols_forCrossTable:
                        #KategorienAlsSpalten_df = pd.DataFrame(columns=['KatVariable']+ list(selected_categorical_cols))
                        KategorienAlsSpalten_df = pd.DataFrame()


                        #ThomasFormatiertesZahlenDataframe = pd.DataFrame(columns=['KatVariable'] + list(selected_float_vars))
                        AnzahlKategorischeVariablen = len(selected_categorical_cols_forCrossTable)

                        
                        #SpaltenVariableTabelle = pd.DataFrame(columns= list(selected_object_vars))
                        

                        SpaltenVariableKumulierteTabelle = pd.DataFrame()

                        for spaltenVariableNr in range(AnzahlKategorischeVariablen):
                            
                            SpaltenVariableTabelle = pd.DataFrame()
                            #SpaltenVariableTabelle['VariableName'] = selected_categorical_cols_forCrossTable[spaltenVariableNr]
                            VariableName = selected_categorical_cols_forCrossTable[spaltenVariableNr]
                            #st.write("VariableName: ",VariableName)

                            #st.write("spaltenVariableNr: ",spaltenVariableNr)
                            #prozente_Spalten_df = pd.DataFrame()
                            for zeilenVariableNr in range(AnzahlKategorischeVariablen):
                                #st.write("zeilenVariableNr: ",zeilenVariableNr)
                                #st.write("selected_object_vars[t]", selected_object_vars[zeilenVariableNr])
                                #ThomasTestgrouped_df = merged_df.groupby(selected_object_vars[t])[selected_float_vars].mean().reset_index()
                                #st.write("ThomasTestgrouped_df: ", ThomasTestgrouped_df)
                                

                                zwischenSpaltenVariableTabelle = pd.crosstab(merged_df_KatVariablen[selected_categorical_cols_forCrossTable[zeilenVariableNr]], merged_df_KatVariablen[selected_categorical_cols_forCrossTable[spaltenVariableNr]], normalize='columns') *100 #normalize='columns' gibt Spalten% normalize=True gibt Tabellen% , margins=True, margins_name="Total" gibt Totalspalte
                                zwischenSpaltenVariableTabelle.index.names = ['Variable']

                                

                                #zwischenSpaltenVariableTabelle['Variabe'] = zwischenSpaltenVariableTabelle.index
                                #st.write("zwischenSpaltenVariableTabelle: ",zwischenSpaltenVariableTabelle)
                                
                                #ProbeTest = (merged_df[selected_object_vars[t]].value_counts(normalize=True).reset_index())
                                #st.write("ProbeTest: ",ProbeTest)
                                #SpaltenVariableTabelle = SpaltenVariableTabelle.append(zwischenSpaltenVariableTabelle) #fügt die Zwischentabellen nacheinander zusammen 
                                SpaltenVariableTabelle = pd.concat([SpaltenVariableTabelle, zwischenSpaltenVariableTabelle]) #fügt die Zwischentabellen nacheinander zusammen, axis=1 könnte ein tip sein..
                                
                                
                                
                                #st.write("SpaltenVariableTabelle: ",SpaltenVariableTabelle)
                                #SpaltenVariableTabelle = SpaltenVariableTabelle.assign(key=0).merge(zwischenSpaltenVariableTabelle.assign(key=0))
                            
                            #st.write("SpaltenVariableTabelle: ",SpaltenVariableTabelle)   
                            SpaltenVariableKumulierteTabelle =  pd.concat([SpaltenVariableKumulierteTabelle, SpaltenVariableTabelle], axis=1 ) #yes!!! axis=1 hat's gebracht!!!!
                            #st.write("SpaltenVariableKumulierteTabelle: ",SpaltenVariableKumulierteTabelle)   

                        #runden
                        SpaltenVariableKumulierteTabelle = SpaltenVariableKumulierteTabelle.round(decimals = 2)

                        #Prozentzeichen einfügen
                        SpaltenVariableKumulierteTabelle = SpaltenVariableKumulierteTabelle.astype(str).apply(lambda x:x + '%') 



                        st.write("Table with column-percentages per variable: ",SpaltenVariableKumulierteTabelle)
                        st.write("")
                        st.write("")
                        #st.subheader("Table Versuchsanfang Crosstable ..with all percentages and frequencies of the selected variables")
                        #st.write(KategorienAlsSpalten_df)



                ################### cross tabulations - end ####################################



                st.write("")
                st.write("")


                if st.checkbox("Show Pearson correlation coefficients of the selected numeric variables?"):
                    # Compute Pearson correlation coefficient for the features in our data set.
                    # The correlation method in pandas, it has the Pearson correlation set as default.
                    st.write(merged_df.corr())

                    df_korr = merged_df.corr()
                    df_korr['Variable'] = df_korr.index

                    # move column 'B' to the leftmost position
                    col_name = 'Variable'
                    col_pos = 0
                    df_korr.insert(col_pos, col_name, df_korr.pop(col_name))

                    st.write("Correlation Heatmap")
                    fig, ax = plt.subplots()
                    sns.heatmap(merged_df.corr(), annot=False, cmap='RdBu')
                    plt.title('Correlation Heatmap', fontsize=8)
                    st.write(fig)



                ################### profile reporting ####################################


                st.write("")
                st.write("")

                _="""
                if st.button("Show Profile-Reporting of all selected variables?", key='profileReporLabeledeData'):
                    st.write("ProfileReport:")
                    profile = ProfileReport(merged_df)
                    st_profile_report(profile)

                    export = profile.to_html()
                    st.download_button(label="Download Profile Report of selected variables", data=export, file_name='report.html')

                """


        ########################################## Metadata  ##############################################################################################################################
       
       
        st.write("")
        st.write('---')
        st.write("")

        MetaDataExpander = st.expander("Show & save Meta-Data (Definition of Variables, Labels) ?") ############################
        with MetaDataExpander:
            st.subheader("Metadata")
            meta_dict = meta.__dict__

            if st.checkbox("Show meta_dict.items()?"):
                st.write("meta_dict.items(): ",meta_dict.items())


            if st.checkbox("Show Dictionary with all Infos?"):
                st.write("meta_dict: ",meta_dict)

            meta_list = [{'Name': k, 'Value': v} for k, v in meta_dict.items()]

            meta_list_Namen = [{'Name': k} for k in meta_dict.items()]

            if st.checkbox("Show meta_list_Namen?"):
                #st.write("meta_list_Namen: ",meta_list_Namen)
                meta_list_Namen_df = pd.DataFrame(meta_list_Namen)
                meta_list_variable_value_labels = meta_list.get('variable_value_labels')         
                #st.write("meta_list_Namen_df: ",meta_list_Namen_df)
                st.write("meta_list_variable_value_labels: ", meta_list_variable_value_labels)


            if st.checkbox("Show meta_list?"):
                st.write("meta_list: ",meta_list)      

            meta_df = pd.DataFrame(meta_list)
            st.write(meta_df)

            if st.checkbox("Show meta_df_gedreht?"):
                meta_df_gedreht = meta_df.T
                st.write("meta_df_gedreht:",meta_df_gedreht)

            #st.write("meta_df.variable_value_labels")


            def to_excel(meta_df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                meta_df.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
                writer.close()
                processed_data = output.getvalue()
                return processed_data


            df_xlsx = to_excel(meta_df)
            st.download_button(label='📥 Export MetaData to Excel?',
                               data=df_xlsx,
                               file_name='SPSSMetaDataToExcel.xlsx')

            if len(col_names_labels_df) > 0:
                st.write("")
                st.write("Names of columns - Original from SPSS and renamed")
                st.write(col_names_labels_df)


                def to_excel(col_names_labels_df):
                    output = BytesIO()
                    writer = pd.ExcelWriter(output, engine='xlsxwriter')
                    col_names_labels_df.to_excel(writer, index=False, sheet_name='Sheet1')
                    workbook = writer.book
                    worksheet = writer.sheets['Sheet1']
                    format1 = workbook.add_format({'num_format': '0.00'})
                    worksheet.set_column('A:A', None, format1)
                    writer.close()
                    processed_data = output.getvalue()
                    return processed_data


                df_xlsx = to_excel(col_names_labels_df)
                st.download_button(label='📥 Export Column Variables to Excel?',
                                   data=df_xlsx,
                                   file_name='SPSSColumnVariablesToExcel.xlsx')

        # Delete temporary file
        tmp_file.close()

    except Exception as e:
        st.write("Error:", e)
