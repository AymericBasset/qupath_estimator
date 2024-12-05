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


# Hardcoded column and class names
TRAIN_TEST_COLUMN = 'Parent'
PROBA_COLUMN = 'Detection probability'
CATEGORY_COLUMN = 'Classification'

# Helper function for bootstrap
# This function resamples the dataset multiple times (n_bootstrap) to calculate average metrics and their variance


def bootstrap_metrics(y_true, y_proba, n_bootstrap=100):
    np.random.seed(42)
    aucs, prs = [], []
    for _ in range(n_bootstrap):
        idx = np.random.choice(
            range(len(y_true)), len(y_true), replace=True)
        y_true_boot = y_true[idx]
        y_proba_boot = y_proba[idx]

        # AUC
        fpr, tpr, _ = roc_curve(y_true_boot, y_proba_boot)
        aucs.append(auc(fpr, tpr))

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(
            y_true_boot, y_proba_boot)
        prs.append(auc(recall, precision))

    return aucs, prs


# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Validate columns
    if TRAIN_TEST_COLUMN not in df.columns or PROBA_COLUMN not in df.columns or CATEGORY_COLUMN not in df.columns:
        st.error(
            f"The CSV must contain '{TRAIN_TEST_COLUMN}', '{PROBA_COLUMN}', and '{CATEGORY_COLUMN}' columns.")
    else:
        # Display dataframe
        st.write("Data Preview:", df.head())

        # Get unique values from the selected column
        unique_values = df[TRAIN_TEST_COLUMN].unique()

        # Select Train and Test classes
        TRAIN_CLASS_NAME = st.selectbox(
            "Select Train Class:", unique_values)
        TEST_CLASS_NAME = st.selectbox("Select Test Class:", [
            val for val in unique_values if val != TRAIN_CLASS_NAME])

        # Get unique values from the selected classification column
        category_values = df[CATEGORY_COLUMN].unique()

        # Select Positive Classes
        POSITIVE_CLASSES = st.multiselect(
            "Select Positive Classes:", category_values)
        if len(POSITIVE_CLASSES) > 0:
            # Split data into training and testing sets based on TRAIN_TEST_COLUMN
            df_train = df[df[TRAIN_TEST_COLUMN] == TRAIN_CLASS_NAME]
            df_test = df[df[TRAIN_TEST_COLUMN] == TEST_CLASS_NAME]

            # Summary of dataset
            st.subheader("Dataset Summary")

            # Function to create a summary of the dataset
            def get_summary(df, positive_classes):
                train_df = df[df['Parent'] == TRAIN_CLASS_NAME]
                test_df = df[df['Parent'] == TEST_CLASS_NAME]
                summary_data = {
                    'train': [
                        sum(train_df['Classification'].isin(positive_classes)),
                        len(train_df) -
                        sum(train_df['Classification'].isin(positive_classes)),
                        len(train_df),
                    ],
                    'test': [
                        sum(test_df['Classification'].isin(positive_classes)),
                        len(test_df) -
                        sum(test_df['Classification'].isin(positive_classes)),
                        len(test_df),
                    ],
                    'all': [
                        sum(df['Classification'].isin(positive_classes)),
                        len(df) -
                        sum(df['Classification'].isin(positive_classes)),
                        len(df)
                    ]
                }

                return pd.DataFrame(summary_data, index=['positive', 'negative', 'both'])

            summary_df = get_summary(df, POSITIVE_CLASSES)
            st.dataframe(summary_df)

            # Prepare train data
            y_train = np.array(
                [1 if c in POSITIVE_CLASSES else 0 for c in df_train[CATEGORY_COLUMN]])
            y_proba_train = df_train[PROBA_COLUMN].values

            # Prepare test data
            y_test = np.array(
                [1 if c in POSITIVE_CLASSES else 0 for c in df_test[CATEGORY_COLUMN]])
            y_proba_test = df_test[PROBA_COLUMN].values

            # ROC Curve for Train
            fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
            roc_auc_train = auc(fpr_train, tpr_train)

            # Precision-Recall Curve for Train
            precision_train, recall_train, _ = precision_recall_curve(
                y_train, y_proba_train)
            pr_auc_train = auc(recall_train, precision_train)

            # Bootstrap Analysis for Train
            aucs_train, prs_train = bootstrap_metrics(y_train, y_proba_train)
            auc_mean_train, auc_std_train = np.mean(
                aucs_train), np.std(aucs_train)
            pr_mean_train, pr_std_train = np.mean(prs_train), np.std(prs_train)

            # Classification Report for Train
            y_pred_train = (y_proba_train > 0.5).astype(int)
            report_train = classification_report(
                y_train, y_pred_train, output_dict=True, zero_division=0)
            report_df_train = pd.DataFrame(report_train).transpose()

            # Confusion Matrix for Train
            cm_train = confusion_matrix(y_train, y_pred_train)
            cm_train_fig = ff.create_annotated_heatmap(cm_train, x=['Pred Neg', 'Pred Pos'], y=[
                'True Neg', 'True Pos'], colorscale='Viridis')
            cm_train_fig.update_layout(
                title_text="Confusion Matrix", width=300, height=300)

            # ROC Curve for Test
            fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
            roc_auc_test = auc(fpr_test, tpr_test)

            # Precision-Recall Curve for Test
            precision_test, recall_test, _ = precision_recall_curve(
                y_test, y_proba_test)
            pr_auc_test = auc(recall_test, precision_test)

            # Bootstrap Analysis for Test
            aucs_test, prs_test = bootstrap_metrics(y_test, y_proba_test)
            auc_mean_test, auc_std_test = np.mean(aucs_test), np.std(aucs_test)
            pr_mean_test, pr_std_test = np.mean(prs_test), np.std(prs_test)

            # Classification Report for Test
            y_pred_test = (y_proba_test > 0.5).astype(int)
            report_test = classification_report(
                y_test, y_pred_test, output_dict=True, zero_division=0)
            report_df_test = pd.DataFrame(report_test).transpose()

            # Confusion Matrix for Test
            cm_test = confusion_matrix(y_test, y_pred_test)
            cm_test_fig = ff.create_annotated_heatmap(cm_test, x=['Pred Neg', 'Pred Pos'], y=[
                'True Neg', 'True Pos'], colorscale='Viridis')
            cm_test_fig.update_layout(
                title_text="Confusion Matrix", width=300, height=300)

            st.subheader("Tests")
            a, b, c, d = st.columns(4)

            pos_train = sum(y_train == 1)
            neg_train = sum(y_train == 0)
            pos_test = sum(y_test == 1)
            neg_test = sum(y_test == 0)

            with a:
                # Chi-squared test for class proportions in train/test
                contingency_table = [
                    [pos_train, neg_train], [pos_test, neg_test]]
                chi2, p_val, _, _ = chi2_contingency(contingency_table)
                if p_val > 0.05:
                    st.success(
                        f"Class proportions are statistically similar (Chi-squared test, p={p_val:.3f})")
                else:
                    st.error(
                        f"Class proportions differ significantly (Chi-squared test, p={p_val:.3f})")

            with b:
                # Test for class balance based on literature

                pos_ratio_train = pos_train / (pos_train+neg_train)
                if (pos_ratio_train - 0.5) < np.abs(0.1):
                    st.success("Balanced problem")
                elif (pos_ratio_train - 0.5) < np.abs(0.25):
                    st.warning("Unbalanced training set")
                else:
                    st.error("Imbalance detected compared to literature.")

            with c:
                # Test if model predictions are better than random
                random_auc = 0.5
                if roc_auc_train > random_auc and roc_auc_test > random_auc:
                    st.success("Model predictions are better than random.")
                else:
                    st.error("Model predictions are not better than random.")

            with d:
                # Test for overfitting
                overfitting_threshold = 0.05  # Adjust based on tolerance
                if abs(roc_auc_train - roc_auc_test) > overfitting_threshold:
                    st.error("Model shows signs of overfitting.")
                else:
                    st.success("No significant overfitting detected.")

            # Display Results for Train and Test Side by Side
            st.subheader("Evaluation Metrics - Training Set vs Testing Set")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Training Set")
                st.write(f"ROC AUC: {roc_auc_train:.3f} ± {auc_std_train:.3f}")
                st.write(f"PR AUC: {pr_auc_train:.3f} ± {pr_std_train:.3f}")
                st.dataframe(report_df_train)
                st.plotly_chart(cm_train_fig)
            with col2:
                st.write("### Testing Set")
                st.write(f"ROC AUC: {roc_auc_test:.3f} ± {auc_std_test:.3f}")
                st.write(f"PR AUC: {pr_auc_test:.3f} ± {pr_std_test:.3f}")
                st.dataframe(report_df_test)
                st.plotly_chart(cm_test_fig)

            # ROC Curve Plot Comparison
            st.subheader("ROC Curve")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr_train, y=tpr_train, mode='lines',
                                         name=f'Train ROC Curve (AUC = {roc_auc_train:.3f})'))
            fig_roc.add_trace(go.Scatter(x=fpr_test, y=tpr_test, mode='lines',
                                         name=f'Test ROC Curve (AUC = {roc_auc_test:.3f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(
                dash='dash'), name="Random Classifier"))
            fig_roc.update_layout(title="ROC Curve - Train vs Test", xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_roc)

            # Precision-Recall Curve Plot Comparison
            st.subheader("Precision-Recall Curve")
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall_train, y=precision_train,
                                        mode='lines', name=f'Train PR Curve (AUC = {pr_auc_train:.3f})'))
            fig_pr.add_trace(go.Scatter(x=recall_test, y=precision_test,
                                        mode='lines', name=f'Test PR Curve (AUC = {pr_auc_test:.3f})'))
            fig_pr.add_trace(go.Scatter(x=[0, 1], y=[0.5, 0.5], mode='lines', line=dict(
                dash='dash'), name="Random Classifier"))
            fig_pr.update_layout(title="Precision-Recall Curve - Train vs Test", xaxis_title="Recall",
                                 yaxis_title="Precision", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_pr)

            # Precision-Recall Plot with Probability Threshold
            st.subheader("Precision and Recall vs. Probability Threshold")
            thresholds = np.linspace(0, 1, 100)
            precision_list, recall_list, f1_list = [], [], []
            for threshold in thresholds:
                y_pred_threshold = (y_proba_test >= threshold).astype(int)
                report = classification_report(
                    y_test, y_pred_threshold, output_dict=True, zero_division=0)
                precision_list.append(report['1']['precision'])
                recall_list.append(report['1']['recall'])
                f1_list.append(report['1']['f1-score'])

            # Find the threshold that maximizes the F1 score
            max_f1_index = np.argmax(f1_list)
            best_threshold = thresholds[max_f1_index]
            st.write(
                f"Probability threshold that maximizes F1 score: {np.round(best_threshold,2)}")

            fig_threshold = go.Figure()
            fig_threshold.add_trace(go.Scatter(
                x=thresholds, y=precision_list, mode='lines', name="Precision"))
            fig_threshold.add_trace(go.Scatter(
                x=thresholds, y=recall_list, mode='lines', name="Recall"))
            fig_threshold.update_layout(title="Precision and Recall vs. Probability Threshold",
                                        xaxis_title="Probability Threshold", yaxis_title="Metric Value")
            st.plotly_chart(fig_threshold)
