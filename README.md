# sdg-classification-bert (sdgBERT App)
This repository relates to **two web applications** powered by a fine-tuned BERT, _**sdgBERT**_, for classifying text concerning the United Nations Sustainable Development Goals (SDG). The manually labeled data used in fine-tuning sdgBERT was obtained from the OSDG Community Dataset, publicly available at https://zenodo.org/record/5550238#.Y93vry9ByF4. The OSDG dataset includes text from diverse fields; hence, the sdgBERT model and the web apps are generic and can be used to predict the SDG of most texts. Note that sdgBERT predicts SDG1 to SDG16 only, excluding SDG17 and the "Other" category for non-SDG text. 

**sdgBERT Repository:** You can access the sdgBERT model repository at: https://huggingface.co/sadickam/sdgBERT

The two apps supports **SDG 1 to SDG 16** shown in the image below
![image](https://user-images.githubusercontent.com/73560591/216751462-ced482ba-5d8e-48aa-9a48-5557979a35f1.png)
Source:https://www.un.org/development/desa/disabilities/about-us/sustainable-development-goals-sdgs-and-disability.html

### App 1: SDG Text Classifier App
This app can be accessed from: 
- (Hugging Face Space): https://sadickam-sdg-text-classifier-app.hf.space 

Key functions of App 1:
- _**Single text prediction:**_ copy/paste or type in a text box
- _**Multiple text prediction:**_ upload a CSV file (Note: The column containing the texts to be predicted must be titled **"text_inputs"**. The app will generate an output csv file that you can download. This downloadable file will include all the original columns in the uploaded CVS, a column for predicted SDGs, and a column for prediction probability scores. If any of the text in text_inputs is longer than the maximum model sequence length of approximately 300 - 400 words (i.e., 512-word pieces), it will be automatically truncated. 

### App 2: Document SDG App
This app can be accessed from: 
- (Hugging Face Space): https://sadickam-document-sdg-app-cpu.hf.space 

This app allows users to analyze PDF documents to check their alignment with the United Nations Sustainable Development Goals (SDGs). When a PDF is uploaded a PDF, the app processes the text to identify and classify content corresponding to the first 16 UN SDGs. The analysis can be conducted at the **page-level** or **sentence-level**, and users can specify the range of PDF document pages to be analyzed. This page specification function can be used to exclude tables of contents, references, appendices, etc.   


### Use fine-tuned BERT Transformer model directly
If you would like to directly use the fine-tuned BERT model, you can easily achieve that using the code below: 
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("sadickam/sdgBERT")

model = AutoModelForSequenceClassification.from_pretrained("sadickam/sdgBERT")
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
The OSDG has an online tool for SDG clsssification of text. I will encourage you to check it out at https://www.osdg.ai/ or visit their github page at https://github.com/osdg-ai/osdg-data to learm more about their tool.

### To do
- Add model evaluation metrics
- Citation information
