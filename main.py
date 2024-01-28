import queue
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from itertools import islice
from itertools import cycle
from dbscan import dbscan
from kmeans import kmeans

def visualize(data, cluster, no_of_clusters, placeholder):
    n = len(data)
    colors = np.array(list(islice(cycle(['#FE4A49', '#2AB7CA', 'yellow', 'purple', 'pink', 'orange', 'green', 'cyan']), 8)))

    plt.figure(figsize=(20, 15))
    for i in range(no_of_clusters):
        if (i == 0):
            color = '#781F19'
        else:
            color = colors[i % len(colors)]

        x, y = [], []
        
        for j in range(n):
            if cluster[j] == i:
                x.append(data[j, 0])
                y.append(data[j, 1])
                plt.scatter(x, y,s=200, c=color, alpha=1, marker='o')
                with placeholder.container():
                    st.pyplot(plt)
                    time.sleep(0.1)
                placeholder.empty()
    with placeholder.container():
        st.pyplot(plt)

if __name__ == "__main__":

    st.markdown("<h1 style='text-align: center; color: #781F19'>Implementing DBSCAN and KMeans</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #781F19'>Algorithm and Visualization</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #781F19'>GROUP 01_04 7</h4>", unsafe_allow_html=True)


    st.write("")
    col1, col2, col3 = st.columns(3)
    df = pd.read_csv('./data/Mall_Customers.csv')

    placeholder = st.empty()
    with col1:
        if st.button("Start DBSCAN Visualization", on_click = None):
            X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

            X = StandardScaler().fit_transform(X)

            point_labels, no_of_clusters = dbscan(X, 0.3, 2)

            visualize(X, point_labels, no_of_clusters, placeholder)
    with col2:
        if st.button("Start KMeans Visualization", on_click = None):
            X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

            X = StandardScaler().fit_transform(X)

            # dbscan = DBSCAN_Cust()
            # point_labels, clusters = dbscan.fit(X, 0.5, 10)
            no_of_clusters = 8

            point_labels = kmeans(X,no_of_clusters, 1000)

            visualize(X, point_labels, no_of_clusters, placeholder)
    with col3:
        if st.button("Clear"):
            placeholder.empty()