# sdg-classification-bert
This streamlit app is powered by a BERT transformer model fined tuned for classifying text with respect the 16 of the 17 United Nations Sustainable Development Goals (SDG). The labelled data used in fine tuning the BERT model was obtained fron the OSDG Community Dataset publicly available at https://zenodo.org/record/5550238#.Y93vry9ByF4. The OSDG dataset include text from diverse fields; hence, the fine tuned BERT model and the streamlit app are generic and can be used to predict the SDG of most texts. 

This streamlit app supports **SDG 1 to SDG 16** shown in the image below
![image](https://user-images.githubusercontent.com/73560591/216751462-ced482ba-5d8e-48aa-9a48-5557979a35f1.png)
Source:https://www.un.org/development/desa/disabilities/about-us/sustainable-development-goals-sdgs-and-disability.html

### Streamlit app link and key functions
You can access the streamlit app at: https://sadickam-sdg-classification-bert-main-pypeui.streamlit.app/
The app has the following key functions:
- _**Single text prediction:**_ copy/paste or type in a text box
- _**Multiple text prediction:**_ upload a csv file (Note: The column contaning the text to be predicted must be title **"text_inputs"**. The app will generate an output csv file that you can download. This downloadable file will include all the original columns in the uploaded cvs, a column for predicted SDGs, and a columns prediction probability scores.

### Use fine tuned BERT Transformer model directly
If you would like to directly use the fine tuned BERT model, you can easily achieve that unsing the code below: 
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("sadickam/sdg-classification-bert")

model = AutoModelForSequenceClassification.from_pretrained("sadickam/sdg-classification-bert")
```
Or just clone the model repo from Hugging Face using the code below:
```
git lfs install
git clone https://huggingface.co/sadickam/sdg-classification-bert

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

### OSDG online tool
The OSDG has online tool for SDG clsssification of text. I will encourage you to check it out at https://www.osdg.ai/ or visit their github page at https://github.com/osdg-ai/osdg-data to learm more about their tool.

### To do
- Add model evaluation metrics
- Citation information
