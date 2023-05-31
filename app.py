import streamlit as st
import pandas as pd
import pyreadstat
import tempfile
from io import BytesIO

from io import StringIO 

import numpy as np

#Um den Datensatz zu analysieren:
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport


#These are the visualization libraries. Matplotlib is standard and is what most people use.
#Seaborn works on top of matplotlib, as we mentioned in the course.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



st.set_page_config(page_title='SPSS Viewer',layout="wide")

st.title("SPSS File Viewer")
st.subheader("Here you can view SPSS-Files with/without labels and also export to Excel")

col_names_labels_df = pd.DataFrame()

# File upload widget
file = st.file_uploader("Upload SPSS file", type=[".sav"])

if file is None:
    st.info("It may take some time to load and convert the SPSS-File, depending on the size of the dataset")

if file is not None:
    # Convert SPSS file to dataframe
    try:
        # Save file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())

        # Use pyreadstat to open and read SPSS file
        data, meta = pyreadstat.read_sav(tmp_file.name)

        # Extract dataframe from pyreadstat output
        df = pd.DataFrame(data)
        
        labelledData = df.copy()
        rawData = df.copy()

        # Checkbox to allow renaming columns with variable labels ############################
        st.write("")
        st.write("")
        st.write("")
        rename_columns = st.checkbox("Rename column names with labels")
        st.write("")
        st.write("")
        st.write("")


        st.write("")
        st.write("")



 ########################################## RawData File ##############################################################################################################################


        rawDataExpander = st.expander("Show & save Raw Data?")  ############################
        with rawDataExpander:
            st.write("## Raw Data")


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
            
            if st.checkbox("Show Column Data Types?"):
                st.write("Column Data Types")
                st.write(rawData.dtypes)


            def to_excel(rawData):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                rawData.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
                writer.save()
                processed_data = output.getvalue()
                return processed_data

            st.write("")
            df_xlsx = to_excel(rawData)
            st.download_button(label='📥 Export Rawdata to Excel?',
                               data=df_xlsx,
                               file_name='SPSSRawDataToExcel.xlsx')


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

               if st.checkbox("Show simple Chart?"):
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

                    _="""
                    # move column 'B' to the leftmost position
                    col_name = 'Variable'
                    col_pos = 0
                    df_korr.insert(col_pos, col_name, df_korr.pop(col_name))
                    
                    st.write("Correlation Heatmap")
                    fig, ax = plt.subplots()
                    sns.heatmap(df.corr(),annot=False,cmap='RdBu')
                    plt.title('Correlation Heatmap',fontsize=8)
                    st.write(fig)
                   """

                   st.write("")
                   
                   if st.button("Show Profile-Reporting?"):

           
                        
                        
                    st.write("ProfileReport:")
                    profile = ProfileReport(df_statistischeTestrawData)
                    st_profile_report(profile)


                    export=profile.to_html()
                    st.download_button(label="Download Profile Report?", data=export, file_name='report.html')



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
    

            st.write(labelledData)
    
            if st.checkbox("Show Column Data Types of labelled data?"):
                st.write(labelledData.dtypes)

            def to_excel(labelledData):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                labelledData.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
                writer.save()
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
            statisticalTests = st.checkbox("Perform statistical tests?",key='statTestLabeledData')
            st.write("")
            st.write("")
            st.write("")
            # Replace column names with variable labels if requested
            if statisticalTests:
 
               my_korrelationsVariablenSelect = st.multiselect("Choose variables for tests",labelledData.columns.tolist(), key='LabeledData')
               df_statistischeTestLabeledData = labelledData[my_korrelationsVariablenSelect]
               
               if len(my_korrelationsVariablenSelect)>1:
                    
                if st.checkbox("Show descriptive Info of labeled Data?"):
                       #st.write(df_statistischeTestLabeledData.describe())

                       st.write(df_statistischeTestLabeledData.describe(include=np.object))

                       st.write("")
                       st.write(df_statistischeTestLabeledData.info())
                       st.write("")                
                   
                #if st.checkbox("Show Pearson correlation coefficients?"):                 
                     # Compute Pearson correlation coefficient for the features in our data set.
                        # The correlation method in pandas, it has the Pearson correlation set as default.
                   # st.write(df_statistischeTestLabeledData.corr())                  
                   
                   
                   
                   
                if st.button("Show Profile-Reporting?", key='profileReporLabeledeData'):

               
                        
                        
                    st.write("ProfileReport:")
                    profile = ProfileReport(df_statistischeTestLabeledData)
                    st_profile_report(profile)
                    
                    export=profile.to_html()
                    st.download_button(label="Download Profile Report", data=export, file_name='report.html')




        ########################################## Combined Data -   merge raw (categoriel data) and labelled data  ##############################################################################################################################
        st.write("")
        st.write('---')
        st.write("")

        if len(rawData)>1 and len(labelledData)>1:


            MergedDataExpander = st.expander("Merge categorical and numerical variables to a new dataset?") ############################
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

                # Let the user select columns to merge
                st.write("")
                selected_numeric_cols = st.multiselect('Select columns with numeric values (brand awareness 0-100, sympathy 1-7,...) :', numeric_cols)
                st.write("")
                selected_categorical_cols = st.multiselect('Select columns with categories (men/women, old/young..):', categorical_cols)
                st.write("")
                # Merge the selected columns
                merged_df = pd.DataFrame()

                # Merge numeric columns

                # Merge categorical columns
                for col in selected_categorical_cols:
                    merged_df[col] = df2_prefixed['categorical_' + col]



                for col in selected_numeric_cols:
                    merged_df[col] = df1_prefixed['numeric_' + col]


                st.write("## Combined dataset")
                if rename_columns:
                    st.info("Datafile with renamed columns")
                # Display the merged data frame
                st.write("Table with selected columns (merged_df):")
                st.dataframe(merged_df)


                if st.checkbox("Show labelling/unique values?"):
                    anzahlVariablen = len(selected_numeric_cols) + len(selected_categorical_cols)
                    st.write("Anzahl Variablen: ",anzahlVariablen)

                    st.write("selected_numeric_cols:", selected_numeric_cols)
                    st.write("selected_categorical_cols:", selected_categorical_cols)

                    st.markdown("#### Ausgewähle Variablen und die jeweilige Auswahl an Ausprägungen:")
                    st.info("Zeilen die None enthalten werden gelöscht")
                    cols = st.columns(anzahlVariablen)
                    
                    VariablenKolumnenAuswahl = merged_df.columns.values.tolist()
                    #st.write("VariablenKolumnenAuswahl: ",VariablenKolumnenAuswahl)



                    dict_of_merged_df= {} # initialize empty dictionary

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

                        #st.write("KategorienListe: ",KategorienListe)

                        #st.write("i: ",i)
                        #col.text_input('Ausprägung', key=i)
                        #Dataframe je Variable mit den Ausprägungen bildenm ev können wir hier dann ein Labeling machen?
                        
                        dict_of_merged_df["merged_df_{}".format(i)] = merged_df[VariablenKolumnenAuswahl[i]].unique()
                        #col.write(dict_of_merged_df["merged_df_{}".format(i)])
                        

                if st.checkbox("Show descriptive Infos?"):
                    st.write(merged_df.describe())
                    st.write("")


                #st.write(merged_df.dtypes)

                if st.checkbox("Create cross-tabulations?"):
                    st.title("Cross Tables with Average Values")

                    # Create multiselect widgets for object and float variables
                    selected_object_vars = selected_categorical_cols
                    selected_float_vars = selected_numeric_cols

                    # Generate cross table with average values
                    if selected_object_vars and selected_float_vars:

                        _="""
                        cross_table = pd.pivot_table(merged_df, values=selected_float_vars, columns=selected_object_vars,
                                                     aggfunc='mean')
                        cross_table.columns = cross_table.columns.map(' '.join)
                        cross_table = cross_table.reset_index().rename(columns={'index': 'VariableName'})

                        st.write("cross_table mit untervariablen in der Spalte")
                        st.write(cross_table)
                        """



                        st.info("cross tables with average of categries")


                        # Group by selected object variables and calculate average values for selected float variables

                        #thomasTest

                        ThomasFormatiertesDataframe = pd.DataFrame(columns=['KatVariable'] + list(selected_float_vars))
                        AnzahlKategorischeVariablen = len(selected_categorical_cols)
                        for t in range(AnzahlKategorischeVariablen):
                            Thomasgrouped_df = merged_df.groupby(selected_object_vars[t])[selected_float_vars].mean().reset_index()
                            Thomasgrouped_df.columns.values[0] = "KatVariable"
                            #st.write("Thomasgrouped_df: ", Thomasgrouped_df)

                            #ThomasFormatiertesDataframe.append(Thomasgrouped_df)
                            ThomasFormatiertesDataframe = pd.concat([ThomasFormatiertesDataframe, Thomasgrouped_df])
                            ThomasFormatiertesDataframe = ThomasFormatiertesDataframe.reset_index(drop=True)
                            ThomasFormatiertesDataframe.index = ThomasFormatiertesDataframe['KatVariable']
                            #ThomasFormatiertesDataframe = ThomasFormatiertesDataframe.set_index('KatVariable')
                            #ThomasFormatiertesDataframe = ThomasFormatiertesDataframe.drop(['KatVariable'],axis = 1, inplace = True)
                            TransposedDataframe = ThomasFormatiertesDataframe.T
                            TransposedDataframe = TransposedDataframe.drop(TransposedDataframe.index[0])
                            #ThomasFormatiertesDataframe = ThomasFormatiertesDataframe.transpose()


                        st.write("Table with categries in the columns:  ", TransposedDataframe)

                        def to_excel(TransposedDataframe):
                            output = BytesIO()
                            writer = pd.ExcelWriter(output, engine='xlsxwriter')
                            TransposedDataframe.to_excel(writer, index=True, sheet_name='Sheet1')
                            workbook = writer.book
                            worksheet = writer.sheets['Sheet1']
                            format1 = workbook.add_format({'num_format': '0.00'})
                            worksheet.set_column('A:A', None, format1)
                            writer.save()
                            processed_data = output.getvalue()
                            return processed_data


                        df_xlsx = to_excel(TransposedDataframe)
                        st.download_button(label='📥 Export Table with categories in the columns to Excel?',
                                           data=df_xlsx,
                               file_name='SPSSCrossTableToExcel.xlsx')

                        st.write("")
                        ThomasFormatiertesDataframe.drop(['KatVariable'], axis=1, inplace=True)
                        st.write("Table with categories in the rows: ",ThomasFormatiertesDataframe)


                        def to_excel(ThomasFormatiertesDataframe):
                            output = BytesIO()
                            writer = pd.ExcelWriter(output, engine='xlsxwriter')
                            ThomasFormatiertesDataframe.to_excel(writer, index=True, sheet_name='Sheet1')
                            workbook = writer.book
                            worksheet = writer.sheets['Sheet1']
                            format1 = workbook.add_format({'num_format': '0.00'})
                            worksheet.set_column('A:A', None, format1)
                            writer.save()
                            processed_data = output.getvalue()
                            return processed_data


                        df_xlsx = to_excel(ThomasFormatiertesDataframe)
                        st.download_button(label='📥 Export Table with categories in the rows to Excel?',
                                           data=df_xlsx,
                               file_name='SPSSCrossTTableRowCategoriesToExcel.xlsx')









        ########################################## Metadata  ##############################################################################################################################
       
       
        st.write("")
        st.write('---')
        st.write("")

        MetaDataExpander = st.expander("Show & save Meta-Data (Definition of Variables, Labels) ?") ############################
        with MetaDataExpander:
            st.subheader("Metadata")
            meta_dict = meta.__dict__
            meta_list = [{'Name': k, 'Value': v} for k, v in meta_dict.items()]
            meta_df = pd.DataFrame(meta_list)
            st.write(meta_df)


            def to_excel(meta_df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                meta_df.to_excel(writer, index=False, sheet_name='Sheet1')
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format1 = workbook.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
                writer.save()
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
                    writer.save()
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
