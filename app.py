import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# Added accuracy_score to the imports below
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(layout="wide")

# --- Caching Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned1_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("ERROR: 'cleaned1_dataset.csv' not found. Please make sure it's in your GitHub repository.")
        return None

df = load_data()

if df is not None:
    st.title("Crime Data Analysis and Prediction Dashboard")
    st.write("This dashboard presents the regression, clustering, and classification models from your notebook.")

    # Define feature columns
    regression_features = df.drop(columns=["total_crimes", "state_name", "district_name"])
    y_reg = df["total_crimes"]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(regression_features, y_reg, test_size=0.2, random_state=42)

    selected_crime_columns = [
        'dowry_prohibition', 'immoral_traffic_prevention', 'women_protection_from_domestic_voilence',
        'protection_of_children_from_sexual_offences', 'prevention_of_atrocities_against_scs',
        'prevention_of_atrocities_against_sts', 'information_technology_act', 'excise_act',
        'gambling_act', 'motor_vehicle_act', 'other_sll_crimes'
    ]

    # --- 1. REGRESSION ---
    st.header("1. Regression: Predicting Total Crimes")

    col1, col2, col3 = st.columns(3)

    # --- Random Forest Regressor ---
    with col1:
        st.subheader("Random Forest Regressor")
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train_reg, y_train_reg)
        y_pred_rf = rf.predict(X_test_reg)
        r2_rf = r2_score(y_test_reg, y_pred_rf)
        
        st.write(f"**R² Score: {r2_rf:.4f}**")
        st.write("R² = 0.93 means the model explains about 93% of the variance in total crimes. This indicates a strong fit.")
        
        importances_rf = pd.Series(rf.feature_importances_, index=regression_features.columns).sort_values(ascending=False)
        fig_rf, ax_rf = plt.subplots()
        importances_rf.head(10).plot(kind="barh", color="teal", ax=ax_rf)
        ax_rf.invert_yaxis()
        ax_rf.set_title("Top 10 Features (Random Forest)")
        st.pyplot(fig_rf)

    # --- XGBoost Regressor ---
    with col2:
        st.subheader("XGBoost Regressor")
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        xgb.fit(X_train_reg, y_train_reg)
        y_pred_xgb = xgb.predict(X_test_reg)
        r2_xgb = r2_score(y_test_reg, y_pred_xgb)
        
        st.write(f"**R² Score: {r2_xgb:.4f}**")
        st.write("XGBoost explains 89% of the variance. It has a lower average error (MAE) but struggles more with extreme values than Random Forest.")
        
        importances_xgb = pd.Series(xgb.feature_importances_, index=regression_features.columns).sort_values(ascending=False)
        fig_xgb, ax_xgb = plt.subplots()
        importances_xgb.head(10).plot(kind="barh", color="orange", ax=ax_xgb)
        ax_xgb.invert_yaxis()
        ax_xgb.set_title("Top 10 Features (XGBoost)")
        st.pyplot(fig_xgb)

    # --- LightGBM Regressor ---
    with col3:
        st.subheader("LightGBM Regressor")
        lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1, num_leaves=64, subsample=0.8, colsample_bytree=0.8, random_state=42)
        lgbm.fit(X_train_reg, y_train_reg)
        y_pred_lgbm = lgbm.predict(X_test_reg)
        r2_lgbm = r2_score(y_test_reg, y_pred_lgbm)
        
        st.write(f"**R² Score: {r2_lgbm:.4f}**")
        st.write("LightGBM explains 80% of the variance, performing worse than the other two models on this dataset.")
        
        importances_lgbm = pd.Series(lgbm.feature_importances_, index=regression_features.columns).sort_values(ascending=False)
        fig_lgbm, ax_lgbm = plt.subplots()
        importances_lgbm.head(10).plot(kind="barh", color="green", ax=ax_lgbm)
        ax_lgbm.invert_yaxis()
        ax_lgbm.set_title("Top 10 Features (LightGBM)")
        st.pyplot(fig_lgbm)

    st.subheader("Regression Conclusion")
    st.write("**Best Model:** Random Forest is the most reliable model with the highest R² (0.9326), indicating the strongest overall fit.")

    # --- 2. CLUSTERING ---
    st.header("2. Clustering: Identifying Crime Profiles")
    st.write("Using K-Means with K=2 to group districts into 'High Crime' and 'Low Crime' profiles after handling outliers.")

    cluster_df = df[selected_crime_columns].copy()
    
    # Handle Outliers
    for col in selected_crime_columns:
        Q1 = cluster_df[col].quantile(0.25)
        Q3 = cluster_df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        cluster_df[col] = cluster_df[col].clip(upper=upper_bound)
    
    X_cluster = cluster_df[selected_crime_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_centers = cluster_df.groupby('cluster')[selected_crime_columns].mean().round(2)
    
    col_cl1, col_cl2 = st.columns(2)
    with col_cl1:
        st.subheader("Distribution of Districts")
        st.bar_chart(cluster_df['cluster'].value_counts())
    
    with col_cl2:
        st.subheader("Mean Crime Rates per Cluster")
        fig_cluster, ax_cluster = plt.subplots(figsize=(12, 8))
        cluster_centers.plot(kind='bar', ax=ax_cluster, colormap='viridis')
        ax_cluster.set_title('Comparison of Mean Crime Rates Across Clusters')
        ax_cluster.set_ylabel('Average Number of Cases')
        ax_cluster.set_xlabel('Cluster')
        ax_cluster.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig_cluster)

    st.subheader("Clustering Conclusion")
    st.write("""
    The analysis clearly segments districts into two profiles:
    * **Cluster 0 (High Crime):** Significantly higher rates, especially in `excise_act`, `gambling_act`, and `protection_of_children_from_sexual_offences`.
    * **Cluster 1 (Low Crime):** Comparatively low crime rates across all categories.
    """)

    # --- 3. CLASSIFICATION ---
    st.header("3. Classification: Predicting Crime Profile")
    st.write("Using Random Forest and Logistic Regression to predict if a district belongs to the 'High' or 'Low' crime cluster.")

    X_class = cluster_df[selected_crime_columns]
    y_class = cluster_df['cluster']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)

    col_cf1, col_cf2 = st.columns(2)

    # --- Random Forest Classifier ---
    with col_cf1:
        st.subheader("Random Forest Classifier")
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_c, y_train_c)
        y_pred_rf_c = rf_classifier.predict(X_test_c)
        accuracy_rf_c = accuracy_score(y_test_c, y_pred_rf_c)
        
        st.write(f"**Model Accuracy: {accuracy_rf_c * 100:.2f}%**")
        st.text("Classification Report:\n" + classification_report(y_test_c, y_pred_rf_c))
        
        importances_rf_c = pd.Series(rf_classifier.feature_importances_, index=selected_crime_columns).sort_values(ascending=False)
        fig_rf_c, ax_rf_c = plt.subplots()
        importances_rf_c.plot(kind="barh", color="viridis", ax=ax_rf_c)
        ax_rf_c.invert_yaxis()
        ax_rf_c.set_title("Feature Importances (RF Classifier)")
        st.pyplot(fig_rf_c)
        st.write("**Top Predictors:** `gambling_act`, `protection_of_children_from_sexual_offences`, and `other_sll_crimes` are the most important features for classifying a district.")


    # --- Logistic Regression ---
    with col_cf2:
        st.subheader("Logistic Regression")
        # Scale data for Logistic Regression
        scaler_lr = StandardScaler()
        X_train_c_scaled = scaler_lr.fit_transform(X_train_c)
        X_test_c_scaled = scaler_lr.transform(X_test_c)
        
        lr_classifier = LogisticRegression(random_state=42)
        lr_classifier.fit(X_train_c_scaled, y_train_c)
        y_pred_lr_c = lr_classifier.predict(X_test_c_scaled)
        accuracy_lr_c = accuracy_score(y_test_c, y_pred_lr_c)
        
        st.write(f"**Model Accuracy: {accuracy_lr_c * 100:.2f}%**")
        st.text("Classification Report:\n" + classification_report(y_test_c, y_pred_lr_c))
        
        coeff_df = pd.DataFrame({
            'Feature': selected_crime_columns,
            'Coefficient': lr_classifier.coef_[0]
        }).sort_values(by='Coefficient', ascending=False)
        
        fig_lr, ax_lr = plt.subplots()
        sns.barplot(x='Coefficient', y='Feature', data=coeff_df, palette='coolwarm', ax=ax_lr)
        ax_lr.set_title('Logistic Regression Coefficients')
        st.pyplot(fig_lr)
        st.write("**Interpretation:** The model found two sets of crimes that are not high in the same districts. For example, high `excise_act` (blue) strongly predicts one cluster, while high `other_sll_crimes` (red) strongly predicts the other.")

    # --- 4. HYPOTHESIS TESTING ---
    st.header("4. Hypothesis Testing")
    
    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        st.subheader("Hypothesis: IT Act vs. Dowry")
        # Cap data for visualization
        df_plot_it = df.copy()
        for col in ['information_technology_act', 'dowry_prohibition']:
             upper_bound = df_plot_it[col].quantile(0.99)
             df_plot_it[col] = df_plot_it[col].clip(upper=upper_bound)
             
        fig_it, ax_it = plt.subplots()
        sns.regplot(data=df_plot_it, x='information_technology_act', y='dowry_prohibition', scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'}, ax=ax_it)
        ax_it.set_title('Relationship: IT Act and Dowry Incidents')
        st.pyplot(fig_it)
        
        lin_reg_it = stats.linregress(df['information_technology_act'], df['dowry_prohibition'])
        st.write(f"**Correlation (r-value): {lin_reg_it.rvalue:.4f}**")
        st.write(f"**P-value:** {lin_reg_it.pvalue:.2e}")
        st.write("**Conclusion:** Confirmed. A moderate, positive, and statistically significant correlation (r=0.41) exists. This is likely due to a lurking variable, **Urbanization**, as both cybercrime and the *reporting* of dowry crimes are highest in urban centers like Bengaluru.")

    with col_h2:
        st.subheader("Hypothesis: Immoral Traffic vs. POCSO")
        # Cap data for visualization
        df_plot_pocso = df.copy()
        for col in ['immoral_traffic_prevention', 'protection_of_children_from_sexual_offences']:
             upper_bound = df_plot_pocso[col].quantile(0.99)
             df_plot_pocso[col] = df_plot_pocso[col].clip(upper=upper_bound)
             
        fig_pocso, ax_pocso = plt.subplots()
        sns.regplot(data=df_plot_pocso, x='immoral_traffic_prevention', y='protection_of_children_from_sexual_offences', scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'}, ax=ax_pocso)
        ax_pocso.set_title('Relationship: Immoral Traffic and POCSO Incidents')
        st.pyplot(fig_pocso)
        
        lin_reg_pocso = stats.linregress(df['immoral_traffic_prevention'], df['protection_of_children_from_sexual_offences'])
        st.write(f"**Correlation (r-value): {lin_reg_pocso.rvalue:.4f}**")
        st.write(f"**P-value:** {lin_reg_pocso.pvalue:.2e}")
        st.write("**Conclusion:** Confirmed. A moderate, positive, and statistically significant correlation (r=0.35) exists. This suggests that these crimes are often co-located, likely occurring in the same high-risk environments or involving overlapping networks.")

else:
    st.info("Please upload 'cleaned1_dataset.csv' to the repository to run the dashboard.")