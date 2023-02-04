import base64
import streamlit as st
import regex as re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import string
import plotly.express as px
import pandas as pd
import nltk
import time
from stqdm import stqdm
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def prep_text(text):
    """
    function for preprocessing text
    """

    # remove trailing characters (\s\n) and convert to lowercase
    clean_sents = [] # append clean con sentences
    sent_tokens = sent_tokenize(str(text))
    for sent_token in sent_tokens:
        word_tokens = [str(word_token).strip().lower() for word_token in sent_token.split()]
        #word_tokens = [word_token for word_token in word_tokens if word_token not in punctuations]
        clean_sents.append(' '.join((word_tokens)))
    joined = ' '.join(clean_sents).strip(' ')
    joined = re.sub(r'`', "", joined)
    joined = re.sub(r'"', "", joined)
    return joined


# model name or path to model
checkpoint = "sadickam/sdg-classification-bert"


# encode df to CSV for downloading
# @st.cache
# def convert_df(df):
#     return df.to_csv().encode('utf-8')


# Load and cache model
@st.cache(allow_output_mutation=True)
def load_model():
    return AutoModelForSequenceClassification.from_pretrained(checkpoint)


# Load and cache tokenizer
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return AutoTokenizer.from_pretrained(checkpoint)


# Configure app page
st.set_page_config(
    page_title="SDG Classifier", layout= "wide", initial_sidebar_state="auto", page_icon="🚦"
)

st.header("🚦 Sustainable Development Goals (SDG) Text Classifier")
st.markdown("")

# upload button recieve input text
st.markdown("##### Column to be analysed must be titled 'text_inputs'")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# lists for appending predictions
predicted_labels = []
prediction_score = []

if uploaded_file is not None:

    # read csv file
    df_docs = pd.read_csv(uploaded_file)
    text_list = df_docs["text_inputs"].tolist()

    # SDG labels list
    label_list = [
        'GOAL_1_No Poverty',
        'GOAL_2_Zero Hunger',
        'GOAL_3_Good Health and Well-being',
        'GOAL_4_Quality Education',
        'GOAL_5_Gender Equality',
        'GOAL_6_Clean Water and Sanitation',
        'GOAL_7_Affordable and Clean Energy',
        'GOAL_8_Decent Work and Economic Growth',
        'GOAL_9_Industry, Innovation and Infrastructure',
        'GOAL_10_Reduced Inequality',
        'GOAL_11_Sustainable Cities and Communities',
        'GOAL_12_Responsible Consumption and Production',
        'GOAL_13_Climate Action',
        'GOAL_14_Life Below Water',
        'GOAL_15_Life on Land',
        'GOAL_16_Peace, Justice and Strong Institutions'
    ]

    # Pre-process text
    for text_input in stqdm(text_list):
        time.sleep(0.02)
        joined_clean_sents = prep_text(text_input)

        # tokenize pre-processed text
        tokenizer = load_tokenizer()
        tokenized_text = tokenizer(joined_clean_sents, return_tensors="pt")

        # predict pre-processed
        model = load_model()
        text_logits = model(**tokenized_text).logits
        predictions = torch.softmax(text_logits, dim=1).tolist()[0]
        predictions = [round(a, 3) for a in predictions]

        # dictionary with label as key and percentage as value
        pred_dict = (dict(zip(label_list, predictions)))

        # sort 'pred_dict' by value and index the highest at [0]
        sorted_preds = sorted(pred_dict.items(), key=lambda g: g[1], reverse=True)

        # Zip explode sorted_preds and append label with highets probability at index 0 to predicted_labels list
        u, v = zip(*sorted_preds)
        x = list(u)
        predicted_labels.append(x[0])
        y = list(v)
        prediction_score.append(y[0])

    # append label and score to df_csv
    df_docs['SDG_predicted'] = predicted_labels
    df_docs['prediction_score'] = prediction_score

    st.empty()

    tab1, tab2 = st.tabs(["💹 SDG Histogram", "⏬ Download CSV file with predictions"])

    with tab1:
        st.markdown("##### Prediction outcome")
        # plot graph of predictions
        fig = px.histogram(df_docs, y="SDG_predicted")

        fig.update_layout(
            # barmode='stack',
            template='seaborn',
            font=dict(
                family="Arial",
                size=14,
                color="black"
            ),
            autosize=False,
            width=800,
            height=500,
            xaxis_title="SDG counts",
            yaxis_title="Sustainable development goals (SDG",
            # legend_title="Topics"
        )

        fig.update_xaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_yaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_annotations(font_size=14)  # this changes y_axis, x_axis and subplot title font sizes

        # Plot
        st.plotly_chart(fig, use_container_width=False)

        st.success("SDGs successfully predicted. ", icon="✅")

    with tab2:
        st.header("")
        csv = df_docs.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64, {b64}" download="sdg_predictions.csv">Download CSV file with predicted SDGs and scores </a>'
        st.markdown(href, unsafe_allow_html=True)

        # st.download_button(
        #     label="Download CSV file with predictions",
        #     data=csv,
        #     file_name='large_df.csv',
        #     mime='text/csv',
        # )


