import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import base64
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#session_config = tf.compat.v1.ConfigProto(
#   inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
#    intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
#    allow_soft_placement=True)

#distribution_strategy = distribution_utils.get_distribution_strategy(
#   flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

#run_config = tf.estimator.RunConfig(
      train_distribute=distribution_strategy, session_config=session_config)

"""
#  Streamlit app - Text Analysis
## This app will focus on text analyzing by application of NLP - Google Universal Sentence Encoder. 
"""

#st.header('Streamlit app - Text Analysis')

st.write('Please upload your CSV file and choose options below')
# A feature of selection of separator (delimeter) for uploaded file has to be developed.  - done
# Also user should have option to choose which column contains target lines of text. - done
# And there should be an option of uploading pre-processed file, where only 1 column with text and hence without any delimeters. - works by choosing column '0'

delimiter = st.selectbox(
    'Which column delimiter file has?',
    ('Comma', 'Tabulation', 'Other option (write below)', 'Not applicable (There is only one column with text)'))
#st.write('You selected:', delimiter)


if delimiter == 'Other option (write below)':
    delimiter = st.text_input('Choose your delimiter', 'Type here')  
elif delimiter == 'Comma':
    delimiter = ','
elif delimiter == 'Tabulation':
    delimiter = '\t'
else:
    pass

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    uploaded_file.seek(0)
    dataframe = pd.read_csv(uploaded_file, sep = delimiter, engine ='python', header = None)
    # Setting range slider
    st.write('You may adjust correlation between phrases of the text on the slider below, where 1 indicates exact match and 0 indicates totally different phrase. Recommended range is from 0.75 to 1.')
    corr_values = st.slider('Select a range of values', 0.0, 1.0, (0.75, 1.0), 0.01)
    (min_value, max_value) = corr_values
    st.write('First 5 rows of the uploaded document:')
    st.write(dataframe.head(5))




# Next will be: user have to choose target column by selectbox or writing. - done
target_column = st.text_input('Choose target text column', 'Type here')

if target_column == 'Type here':
    pass
else:
    st.write('Check first five items if they have been grabbed correctly:')
    d1list = dataframe[int(target_column)].dropna().tolist()
    st.write(d1list[:5])
    st.write('Analyzing data... Please wait.')
    # Loading Google USE model
    module = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

    # Creating encoding and similarity matrices
    encoding_matrix = module(d1list)
    similarity_matrix = np.inner(encoding_matrix, encoding_matrix)
    # The following matrix will be used for findig popular phrases
    matrix = pd.DataFrame(data=similarity_matrix)

    # Next steps are finding most popular tweets. Method: counting of number of cells correlated with current cell. Correlation between 0.75 & 1 as an example
    # Custom correlation controls can be added
    #min_value = 0.75
    #max_value = 1

    # Following function allow us to count quantity of similar phrases in the dataframe for each phrase 
    def corr_counter(cell_number):
      count = matrix[cell_number][(matrix.iloc[cell_number]>min_value) & (matrix.iloc[cell_number]<max_value)].count()
      return count

    # Creating a dataframe with result of counting
    arr = []
    for i in range(len(matrix.index)):
      arr.append(corr_counter(i))
    arr_df = pd.DataFrame(arr, columns = ['popularity_counter'])

    #st.write('Description of the data \n', arr_df.describe())

    # Adding length feature to the dataframe
    arr_df_length = pd.DataFrame(d1list, columns = ['text'])
    arr_df_length['length'] = arr_df_length['text'].apply(len)
    arr_df = arr_df.join(arr_df_length)
    #st.write(arr_df.head())
    #Removing duplicates
    def redundant_sent_idx(sim_matrix):
        dup_idx = []
        for i in range(sim_matrix.shape[0]):
            if i not in dup_idx:
                tmp = [t+i+1 for t in list(np.where( sim_matrix[i][i+1:] > 0.75 )[0])]
                dup_idx.extend(tmp)
        return dup_idx
    dup_indexes  = redundant_sent_idx(similarity_matrix)
    sorted_df = arr_df.drop(dup_indexes).sort_values(by = 'popularity_counter', ascending = False)
    st.write('Ten most popular phrases of the document:', sorted_df.head(10))
    
    def get_table_download_link(df):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=True)
        b64 = base64.b64encode(
            csv.encode()
        ).decode()  # some strings <-> bytes conversions necessary here
        return f'<a href="data:file/csv;base64,{b64}" download="sorted_by_popularity.csv">Download full csv file</a>'
    st.markdown(get_table_download_link(sorted_df), unsafe_allow_html=True)
