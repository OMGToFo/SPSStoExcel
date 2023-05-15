import streamlit as st
import pandas as pd
import pyreadstat
import tempfile
from io import BytesIO

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
   
               
               if len(my_korrelationsVariablenSelect)>0:
  
                   if st.checkbox("Show descriptive Info?"):
                       st.write(df_statistischeTestrawData.describe())
                       st.write("")
                       st.write(df_statistischeTestrawData.info())
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
                   
                   
                   if st.button("Show Profile-Reporting?"):

           
                        
                        
                    st.write("ProfileReport:")
                    profile = ProfileReport(df_statistischeTestrawData)
                    st_profile_report(profile)


                    export=profile.to_html()
                    st.download_button(label="Download Profile Report?", data=export, file_name='report.html')



        st.write("")
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
                       st.write(df_statistischeTestLabeledData.describe())
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
    
        
        
        
 ########################################## Metadata  ##############################################################################################################################   
       
       
        st.write("")
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
