import streamlit as st
from PIL import Image
import webbrowser
from menu_home import display_home
from menu_data_collection import display_data_collection
from menu_data_annotation import display_data_annotation
from menu_eda import display_eda
from menu_data_preprocessing import display_data_preprocessing
from menu_feature_extraction import display_feature_extraction
from menu_modeling import display_modeling
from menu_inference import display_inference


st.set_page_config(layout="wide")

# menu sidebar
list_menu = ['Home', 'Data Collection', 'Data Annotation', 'Exploratory Data Analysis', 'Data Preprocessing', 
             'Feature Extraction', 'Modeling', 'Inference']
menu_choice = st.sidebar.selectbox("Select a menu", list_menu)

url = 'https://shrouded-dusk-24137.herokuapp.com/'
if st.sidebar.button('Demo Sentiment Analysis 🚀'):
    webbrowser.open_new_tab(url)

st.sidebar.title('Teams')
text = """
1. Alfi Fauzia Hanifah ([LinkedIn](https://www.linkedin.com/in/alfifauziahanifah/) / [IG](https://www.instagram.com/alfifao/))
2. Annisa P A ([LinkedIn](https://www.linkedin.com/in/annisapa/) / [IG](https://www.instagram.com/annisapa__/))
3. Javas Alfreda Belva ([LinkedIn](https://www.linkedin.com/in/javasalfredabyp/) / [IG](https://www.instagram.com/javasalfreda_byp/))
4. Muhammad Aghassi ([LinkedIn](https://www.linkedin.com/in/maghassiz/) / [IG](https://www.instagram.com/maghassiz/))
5. Yasir Abdur Rohman ([LinkedIn](https://www.linkedin.com/in/yasirabd/) / [IG](https://www.instagram.com/yasirabdr/))

---

"""
st.sidebar.markdown(text, unsafe_allow_html=True)

doccano = Image.open('images/dsi_logo.png')
st.sidebar.image(doccano, caption='', width=100)
text = """
**Divisi Research**<br>
Data Science Indonesia Chapter Jawa Tengah
"""
st.sidebar.markdown(text, unsafe_allow_html=True)

### MENU: HOME ###
if menu_choice == 'Home':
    display_home()
    
### MENU: DATA COLLECTION ###
if menu_choice == 'Data Collection':
    display_data_collection()

### MENU: DATA ANNOTATION ###
if menu_choice == 'Data Annotation':
    display_data_annotation()

### MENU: EXPLORATORY DATA ANALYSIS ###
if menu_choice == 'Exploratory Data Analysis':
    display_eda()

### MENU: DATA PREPROCESSING ###
if menu_choice == 'Data Preprocessing':
    display_data_preprocessing()

### MENU: FEATURE EXTRACTION ###
if menu_choice == 'Feature Extraction':
    display_feature_extraction()

### MENU: MODELING ###
if menu_choice == 'Modeling':
    display_modeling()

### MENU: INFERENCE ###
if menu_choice == 'Inference':
    display_inference()