import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import string
import plotly.express as px
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')

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

st.title("🚦 Sustainable Development Goals (SDG) Text Classifier")
# st.header("")

with st.expander("About this app", expanded=False):
    st.write(
        """
        - Artificial Intelligence (AI) tool for automatic classification of text with respect to the UN Sustainable Development Goals (SDG)
        - Note that 16 out of the 17 SDGs are covered
        - This tool is for sustainability assessment and benchmarking and is not limited to a specific industry
        - The model powering this app was developed using the OSDG Community Dataset (OSDG-CD) [Link - https://zenodo.org/record/5550238#.Y8Sd5f5ByF5]
        """
    )


# Form to recieve input text 
st.markdown("##### Text Input")
with st.form(key="my_form"):
    Text_entry = st.text_area(
        "Paste or type text in the box below (i.e., input)"
    )
    submitted = st.form_submit_button(label="👉 Get SDG prediction!")

if submitted:

    # SDG labels list

    label_list = [
        'GOAL 1: No Poverty',
        'GOAL 2: Zero Hunger',
        'GOAL 3: Good Health and Well-being',
        'GOAL 4: Quality Education',
        'GOAL 5: Gender Equality',
        'GOAL 6: Clean Water and Sanitation',
        'GOAL 7: Affordable and Clean Energy',
        'GOAL 8: Decent Work and Economic Growth',
        'GOAL 9: Industry, Innovation and Infrastructure',
        'GOAL 10: Reduced Inequality',
        'GOAL 11: Sustainable Cities and Communities',
        'GOAL 12: Responsible Consumption and Production',
        'GOAL 13: Climate Action',
        'GOAL 14: Life Below Water',
        'GOAL 15: Life on Land',
        'GOAL 16: Peace, Justice and Strong Institutions'
    ]
    
    # Pre-process text
    joined_clean_sents = prep_text(Text_entry)

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
    sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)

    # Make dataframe for plotly bar chart
    u, v = zip(*sorted_preds)
    x = list(u)
    y = list(v)
    df2 = pd.DataFrame()
    df2['SDG'] = x
    df2['Likelihood'] = y

    c1, c2, c3 = st.columns([1.5, 0.5, 1])

    with c1:
        st.markdown("##### Prediction outcome")
        # plot graph of predictions
        fig = px.bar(df2, x="Likelihood", y="SDG", orientation="h")

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
            xaxis_title="Likelihood of SDG",
            yaxis_title="Sustainable development goals (SDG)",
            # legend_title="Topics"
        )

        fig.update_xaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_yaxes(tickangle=0, tickfont=dict(family='Arial', color='black', size=14))
        fig.update_annotations(font_size=14)  # this changes y_axis, x_axis and subplot title font sizes

        # Plot
        st.plotly_chart(fig, use_container_width=False)

    with c3:
        st.header("")
        predicted = st.markdown("###### Predicted " + str(sorted_preds[0][0]))
        Prediction_confidence = st.metric("Prediction confidence", (str(round(sorted_preds[0][1]*100, 1))+"%"))

        st.success("SDG successfully predicted. ", icon="✅")
