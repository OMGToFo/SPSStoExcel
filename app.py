import streamlit as st
import pandas as pd
import pyreadstat
import tempfile
from io import BytesIO

st.set_page_config(page_title='SPSS Viewer',layout="wide")

st.title("SPSS File Viewer")
st.subheader("Here you can view your SPSS-Files with/without Labels and also export them to Excel")

col_names_labels_df = pd.DataFrame()

# File upload widget
file = st.file_uploader("Upload SPSS file", type=[".sav"])

if file is None:
    st.info("It may take some time to load and concert the SPSS-Files, depending on the size of the dataset")

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

        # Checkbox to allow renaming columns with variable labels ############################
        st.write("")
        st.write("")
        st.write("")
        rename_columns = st.checkbox("Rename column names with labels")
        st.write("")
        st.write("")
        st.write("")
        # Replace column names with variable labels if requested
        if rename_columns:
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
                if row['Column Name'] in df.columns:
                    df.rename(columns={row['Column Name']: row['VariableLabelUnique']}, inplace=True)

        st.write("")
        st.write("")

        rawDataExpander = st.expander("Show & save Raw Data?")
        with rawDataExpander:
            st.write("## Raw Data")
            rawData = df.copy()

            if rename_columns:
                st.info("Datafile with renamed columns")
            st.write(rawData)


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

        # Replace values with value labels
        for var in meta.variable_value_labels:
            if var in df.columns:
                value_labels = meta.variable_value_labels[var]
                df[var] = df[var].replace(value_labels)

        st.write("")
        st.write("")

        LabelledDataExpander = st.expander("Show & save Data with labeled Values?")
        with LabelledDataExpander:
            st.write("## Data with Labels")
            labelledData = df.copy()
            if rename_columns:
                st.info("Datafile with renamed columns")
            st.write(labelledData)


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

        st.write("")
        st.write("")

        MetaDataExpander = st.expander("Show & save Meta-Data (Definition of Variables, Labels) ?")
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
