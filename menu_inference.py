import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter


def display_inference():
    # read data
    df_comments = pd.read_csv('data/data_complete_visualization.csv')
    df_coronas = pd.read_csv('data/data_corona_semarang_14112021.csv')
    df_sentiment_counter = pd.read_csv('data/sentiment_counter.csv')

    # convert data types
    df_comments['datetime'] = pd.to_datetime(df_comments['datetime'])
    df_coronas['Tanggal'] = pd.to_datetime(df_coronas['Tanggal'])
    df_sentiment_counter['date'] = pd.to_datetime(df_sentiment_counter['date'])
    df_sentiment_counter['neutral'] = pd.to_numeric(df_sentiment_counter['neutral'])
    df_sentiment_counter['positive'] = pd.to_numeric(df_sentiment_counter['positive'])
    df_sentiment_counter['negative'] = pd.to_numeric(df_sentiment_counter['negative'])

    # filter only for data for inference
    df_comments = df_comments[df_comments['data_type'] == 'inference'].reset_index(drop=True)
    # get start date and end date
    start_date = df_comments['datetime'].min()
    end_date = df_comments['datetime'].max()

    df_coronas = df_coronas[(df_coronas['Tanggal'] >= start_date) & (df_coronas['Tanggal'] <= end_date)]
    df_sentiment_counter = df_sentiment_counter[(df_sentiment_counter['date'] >= start_date) & (
        df_sentiment_counter['date'] <= end_date)]

    text = """
    # Model Inference
    Model machine learning yang sudah dilatih digunakan untuk memprediksi data baru (*inference*). Berdasarkan 
    tahap modeling, model machine learning terbaik adalah `TF-IDF SVM` dengan akurasi skor `74.60%` pada data test.

    ### Data inference
    Data baru yang digunakan untuk proses *inference* memiliki deskripsi sebagai berikut.
    """
    st.markdown(text, unsafe_allow_html=True)
    st.markdown(f"Shape dari data inference: {df_comments.shape[0]} baris, {df_comments.shape[1]} kolom")
    st.markdown(
        f'Rentang waktu: `{df_comments["datetime"].min().strftime("%d %b %Y")}` sampai `{df_comments["datetime"].max().strftime("%d %b %Y")}`')
    st.dataframe(df_comments.head(10))
    st.info('Data yang ditampilkan sudah dilakukan *preprocessing*, *masking*, dan diprediksi labelnya.')

    text = """
    ### Hasil inference
    Berikut ini adalah hasil inference model pada data baru.

    | Sentiment  | Jumlah |
    |------------|--------|
    | `neutral`  | 8718   |
    | `positive` | 3050   |
    | `negative` | 3491   |
    <br>
    """
    st.markdown(text, unsafe_allow_html=True)

    temp = df_comments['label'].value_counts().reset_index()
    temp = temp.rename(columns={'index': 'label', 'label': 'count'})
    # funnel chart
    st.markdown("Untuk visualisasi lebih baik, kita buat Funnel-Chart untuk mengetahui persentase dari tiap sentimen.")
    fig_funnel = px.funnel_area(names=temp['label'],
                                values=temp['count'],
                                title='Funnel-Chart of Sentiment Distribution',
                                color_discrete_map={'neutral': '#636EFA', 'positive': '#00CC96', 'negative': '#EF553B'})
    st.plotly_chart(fig_funnel)

    # make plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_coronas['Tanggal'], y=df_coronas['POSITIVE ACTIVE'],
                         name='Positive Active',
                         marker_color='rgba(255, 161, 90, 0.8)',
                         marker_line_width=0),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x=df_sentiment_counter['date'], y=df_sentiment_counter['neutral'],
                             mode='lines+markers', marker_color='#636EFA', name='Sentiment Neutral'),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=df_sentiment_counter['date'], y=df_sentiment_counter['positive'],
                             mode='lines+markers', marker_color='#00CC96', name='Sentiment Positive'),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=df_sentiment_counter['date'], y=df_sentiment_counter['negative'],
                             mode='lines+markers', marker_color='#EF553B', name='Sentiment Negative'),
                  secondary_y=True)

    # Add figure title
    fig.update_layout(
        title_text="COVID-19 Positive Active Cases vs Sentiment (Neutral, Positive, Negative)",
        template='presentation',
        plot_bgcolor='rgb(275, 270, 273)'
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="Number of Sentiments",
                     secondary_y=True, rangemode='tozero')
    fig.update_yaxes(title_text="Number of Cases",
                     secondary_y=False, rangemode='tozero')
    st.plotly_chart(fig, use_container_width=True)

    # WORD CLOUD

    # filter data for each sentiment
    neutral = df_comments[df_comments.label == "neutral"]
    positive = df_comments[df_comments.label == "positive"]
    negative = df_comments[df_comments.label == "negative"]

    # create slider for the number of words
    n_words = st.slider("Set the number of words in Word Cloud",
                        min_value=50, max_value=200, step=10)

    # create 3 columns
    col1, col2, col3 = st.columns(3)

    # neutral
    with col1:
        texts = ''
        for val in neutral['text']:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = re.sub(r'[^\x00-\x7F]+', ' ', tokens[i])
                tokens[i] = tokens[i].strip()
            texts += " ".join(tokens)+" "
        wc = WordCloud(
            colormap="Blues",
            mode="RGBA",
            max_words=n_words,
            background_color="white",
            collocations=False,
            width=400, height=400,
        )
        wc.generate(texts)

        st.markdown("#### Sentiment Neutral", unsafe_allow_html=False)
        # plot
        fig = plt.figure(figsize=(20, 8), dpi=80)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

        # top 10 words
        top10 = Counter(texts.split()).most_common(10)
        df_top10 = pd.DataFrame(top10, columns=['word', 'freq'])
        df_top10 = df_top10.sort_values(by='freq', ascending=True)

        # plot
        fig_bar = px.bar(df_top10, x="freq", y="word",
                         orientation='h', height=400, width=350)
        fig_bar.update_layout(title_text='Top 10 words sentiment neutral',
                              plot_bgcolor='rgb(275, 270, 273)')
        fig_bar.update_traces(marker_color='#636EFA')

        # Set x-axes and y-axes titles
        fig_bar.update_yaxes(title_text="")
        fig_bar.update_xaxes(title_text="frequency")
        st.plotly_chart(fig_bar)

    # positive
    with col2:
        texts = ''
        for val in positive['text']:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = re.sub(r'[^\x00-\x7F]+', ' ', tokens[i])
                tokens[i] = tokens[i].strip()
            texts += " ".join(tokens)+" "
        wc = WordCloud(
            colormap="Greens",
            mode="RGBA",
            max_words=n_words,
            background_color="white",
            collocations=False,
            width=400, height=400,
        )
        wc.generate(texts)

        st.markdown("#### Sentiment Positive", unsafe_allow_html=False)
        # plot
        fig = plt.figure(figsize=(20, 8), dpi=80)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

        # top 10 words
        top10 = Counter(texts.split()).most_common(10)
        df_top10 = pd.DataFrame(top10, columns=['word', 'freq'])
        df_top10 = df_top10.sort_values(by='freq', ascending=True)

        # plot
        fig_bar = px.bar(df_top10, x="freq", y="word",
                         orientation='h', height=400, width=350)
        fig_bar.update_layout(title_text='Top 10 words sentiment positive',
                              plot_bgcolor='rgb(275, 270, 273)')
        fig_bar.update_traces(marker_color='#00CC96')

        # Set x-axes and y-axes titles
        fig_bar.update_yaxes(title_text="")
        fig_bar.update_xaxes(title_text="frequency")
        st.plotly_chart(fig_bar)
    # negative
    with col3:
        texts = ''
        for val in negative['text']:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = re.sub(r'[^\x00-\x7F]+', ' ', tokens[i])
                tokens[i] = tokens[i].strip()
            texts += " ".join(tokens)+" "
        wc = WordCloud(
            colormap="Reds",
            mode="RGBA",
            max_words=n_words,
            background_color="white",
            collocations=False,
            width=400, height=400,
        )
        wc.generate(texts)

        st.markdown("#### Sentiment Negative", unsafe_allow_html=False)
        # plot
        fig = plt.figure(figsize=(20, 8), dpi=80)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

        # top 10 words
        top10 = Counter(texts.split()).most_common(10)
        df_top10 = pd.DataFrame(top10, columns=['word', 'freq'])
        df_top10 = df_top10.sort_values(by='freq', ascending=True)

        # plot
        fig_bar = px.bar(df_top10, x="freq", y="word",
                         orientation='h', height=400, width=350)
        fig_bar.update_layout(title_text='Top 10 words sentiment negative',
                              plot_bgcolor='rgb(275, 270, 273)')
        fig_bar.update_traces(marker_color='#EF553B')

        # Set x-axes and y-axes titles
        fig_bar.update_yaxes(title_text="")
        fig_bar.update_xaxes(title_text="frequency")
        st.plotly_chart(fig_bar)