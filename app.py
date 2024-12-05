import streamlit as st
import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import chi2_contingency

st.set_page_config(layout='wide')
st.title("QuPath Estimator")
