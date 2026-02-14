import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Car Sales Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1F77B4;
        font-weight: 700;
    }
    h2, h3 {
        color: #2C3E50;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .improvement-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_resource
def load_artifacts():
    """Load trained model and preprocessing objects"""
    try:
        # Try to load tuned model first
        tuned_model_path = Path('models/tuned_xgboost_model.pkl')
        if tuned_model_path.exists():
            with open(tuned_model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open('models/tuned_model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            model_type = "tuned"
            st.sidebar.success("🎯 Using Hyperparameter-Tuned Model")
        else:
            # Fallback to baseline model
            with open('models/best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            with open('models/model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            model_type = "baseline"
            st.sidebar.info("📊 Using Baseline Model")
        
        # Load preprocessors
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        return model, scaler, encoders, metadata, model_type
    
    except FileNotFoundError as e:
        st.error("⚠️ Model files not found! Please train the model first.")
        st.info("""
        **Steps to fix:**
        1. Open Jupyter Notebook: `jupyter notebook`
        2. Run `notebooks/01_eda.ipynb` completely
        3. Run `notebooks/02_modeling.ipynb` completely
        4. (Optional) Run `notebooks/03_advanced_features.ipynb` for tuned model
        5. This will create the required model files in `models/` directory
        """)
        st.stop()

model, scaler, encoders, metadata, model_type = load_artifacts()

# Load sample data to get feature info
@st.cache_data
def load_sample_data():
    """Load processed data to extract feature ranges"""
    try:
        df = pd.read_csv('data/processed/car_sales_clean.csv')
        return df
    except FileNotFoundError:
        st.warning("⚠️ Processed data not found. Run 01_eda.ipynb first.")
        return None

sample_data = load_sample_data()

# ===== HEADER =====
st.title("🚗 Car Sales Price Prediction System")
st.markdown("### Machine Learning Vehicle Pricing Tool")

# Show improvement badge if using tuned model
if model_type == "tuned" and 'baseline_metrics' in metadata:
    baseline_r2 = metadata['baseline_metrics']['test_r2']
    tuned_r2 = metadata['metrics']['test_r2']
    improvement = ((tuned_r2 - baseline_r2) / baseline_r2) * 100
    
    st.markdown(f"""
    <div class="improvement-badge">
        ⚡ Performance Improved by {improvement:.2f}% through Hyperparameter Tuning
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===== SIDEBAR =====
st.sidebar.header("📊 Model Information")

# Display model metrics
if model_type == "tuned":
    st.sidebar.markdown(f"""
    **Model:** {metadata.get('model_name', 'Tuned XGBoost')}  
    **Tuning Method:** {metadata.get('tuning_method', 'RandomizedSearchCV')}  
    **R² Score:** {metadata['metrics']['test_r2']:.4f}  
    **RMSE:** ${metadata['metrics']['test_rmse']:,.2f}  
    **MAE:** ${metadata['metrics']['test_mae']:,.2f}
    """)
    
    # Show improvement over baseline
    if 'baseline_metrics' in metadata:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📈 vs Baseline:**")
        baseline_r2 = metadata['baseline_metrics']['test_r2']
        improvement = ((metadata['metrics']['test_r2'] - baseline_r2) / baseline_r2) * 100
        st.sidebar.metric("R² Improvement", f"+{improvement:.2f}%")
else:
    st.sidebar.markdown(f"""
    **Model:** {metadata['model_name']}  
    **R² Score:** {metadata['metrics']['test_r2']:.4f}  
    **RMSE:** ${metadata['metrics']['test_rmse']:,.2f}  
    **MAE:** ${metadata['metrics']['test_mae']:,.2f}
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **How to Use:**
1. Enter car features in the form
2. Click 'Predict Price'
3. View prediction with confidence interval
""")

# ===== MAIN CONTENT =====
# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📈 Model Performance", "🔍 Explainability", "ℹ️ About"])

with tab1:
    st.header("Enter Car Details")
    
    if sample_data is not None:
        # Create input form based on actual features
        col1, col2, col3 = st.columns(3)
        
        user_input = {}
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = sample_data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if present
        price_cols = [col for col in numeric_features if 'price' in col.lower()]
        if price_cols:
            numeric_features.remove(price_cols[0])
        
        # Dynamically create inputs
        feature_count = 0
        
        # Numeric features
        for feature in numeric_features:
            col = [col1, col2, col3][feature_count % 3]
            
            with col:
                min_val = float(sample_data[feature].min())
                max_val = float(sample_data[feature].max())
                mean_val = float(sample_data[feature].mean())
                
                user_input[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=f"Range: {min_val:.2f} - {max_val:.2f}"
                )
            
            feature_count += 1
        
        # Categorical features
        for feature in categorical_features:
            col = [col1, col2, col3][feature_count % 3]
            
            with col:
                if feature in encoders:
                    options = encoders[feature].classes_.tolist()
                    user_input[feature] = st.selectbox(
                        f"{feature}",
                        options=options
                    )
            
            feature_count += 1
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            try:
                # Prepare input
                input_df = pd.DataFrame([user_input])
                
                # Encode categorical variables
                for col in categorical_features:
                    if col in encoders:
                        input_df[col] = encoders[col].transform(input_df[col])
                
                # Ensure correct feature order
                feature_order = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else list(user_input.keys())
                input_df = input_df[feature_order]
                
                # Scale features
                input_scaled = scaler.transform(input_df)
                
                # Predict
                prediction = model.predict(input_scaled)[0]
                
                # Calculate confidence interval (using RMSE as std)
                rmse = metadata['metrics']['test_rmse']
                confidence_lower = prediction - (1.96 * rmse)
                confidence_upper = prediction + (1.96 * rmse)
                
                # Display results
                st.markdown("## 🎉 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Predicted Price</h3>
                        <h1>${prediction:,.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Lower Bound (95% CI)", f"${confidence_lower:,.2f}")
                
                with col3:
                    st.metric("Upper Bound (95% CI)", f"${confidence_upper:,.2f}")
                
                # Confidence gauge
                st.markdown("### 📊 Prediction Confidence")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=metadata['metrics']['test_r2'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Model Confidence (R² Score)", 'font': {'size': 24}},
                    delta={'reference': 80, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#FFE5E5'},
                            {'range': [50, 75], 'color': '#FFF4E5'},
                            {'range': [75, 100], 'color': '#E5F5E5'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                model_name = metadata.get('model_name', 'XGBoost')
                st.info(f"""
                **💡 Interpretation:**  
                Based on the {model_name}, we predict this car's price to be **${prediction:,.2f}**.  
                With 95% confidence, the actual price should fall between **${confidence_lower:,.2f}** and **${confidence_upper:,.2f}**.  
                This model explains **{metadata['metrics']['test_r2']*100:.1f}%** of price variation in our dataset.
                """)
            
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("Please check that all inputs are valid and try again.")
    
    else:
        st.warning("⚠️ Sample data not available. Please run `01_eda.ipynb` to generate processed data.")

with tab2:
    st.header("📈 Model Performance Metrics")
    
    # Metrics comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{metadata['metrics']['test_r2']:.4f}", 
                 help="Proportion of variance explained (closer to 1 is better)")
    
    with col2:
        st.metric("RMSE", f"${metadata['metrics']['test_rmse']:,.2f}", 
                 help="Root Mean Squared Error (lower is better)")
    
    with col3:
        st.metric("MAE", f"${metadata['metrics']['test_mae']:,.2f}", 
                 help="Mean Absolute Error (lower is better)")
    
    st.markdown("---")
    
    # Show hyperparameter tuning results if available
    if model_type == "tuned" and 'best_params' in metadata:
        st.subheader("🔧 Optimized Hyperparameters")
        
        params_df = pd.DataFrame([metadata['best_params']]).T
        params_df.columns = ['Value']
        st.dataframe(params_df, use_container_width=True)
    
    st.markdown("---")
    
    # Model comparison chart (if available)
    st.subheader("📊 Performance Visualization")
    
    if Path('visualizations/model_comparison.png').exists():
        st.image('visualizations/model_comparison.png', 
                caption='Model Performance Comparison', 
                use_container_width=True)
    else:
        st.info("Run `02_modeling.ipynb` to generate model comparison visualization.")
    
    if Path('visualizations/feature_importance.png').exists():
        st.image('visualizations/feature_importance.png', 
                caption='Feature Importance Analysis', 
                use_container_width=True)
    else:
        st.info("Run `02_modeling.ipynb` to generate feature importance visualization.")

with tab3:
    st.header("🔍 Model Explainability (SHAP)")
    
    st.markdown("""
    SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions,
    making the model's decisions transparent and interpretable.
    """)
    
    # Check if SHAP visualizations exist
    shap_dir = Path('visualizations/shap')
    
    if shap_dir.exists():
        # SHAP Summary Plot
        if (shap_dir / 'shap_summary.png').exists():
            st.subheader("📊 Global Feature Impact")
            st.image(str(shap_dir / 'shap_summary.png'), 
                    caption='SHAP Summary Plot - How features impact predictions globally',
                    use_container_width=True)
        
        # SHAP Feature Importance
        if (shap_dir / 'shap_importance.png').exists():
            st.subheader("🎯 Feature Importance Ranking")
            st.image(str(shap_dir / 'shap_importance.png'),
                    caption='Mean absolute SHAP values - Feature importance',
                    use_container_width=True)
        
        # SHAP Waterfall
        if (shap_dir / 'shap_waterfall.png').exists():
            st.subheader("💧 Individual Prediction Breakdown")
            st.image(str(shap_dir / 'shap_waterfall.png'),
                    caption='SHAP Waterfall Plot - How features contributed to a single prediction',
                    use_container_width=True)
        
        # SHAP Dependence Plots
        dependence_plots = list(shap_dir.glob('shap_dependence_*.png'))
        if dependence_plots:
            st.subheader("🔗 Feature Interaction Analysis")
            for plot_path in dependence_plots[:3]:  # Show max 3
                st.image(str(plot_path),
                        caption=f'Feature interaction: {plot_path.stem.replace("shap_dependence_", "").replace("_", " ")}',
                        use_container_width=True)
    else:
        st.info("""
        **SHAP visualizations not yet generated.**
        
        Run `notebooks/03_advanced_features.ipynb` to generate:
        - Global feature importance
        - Individual prediction explanations
        - Feature interaction analysis
        """)
        
        st.markdown("""
        ### Why SHAP Matters:
        
        - **🎯 Transparency:** Understand exactly why the model made a prediction
        - **🔍 Trust:** Identify if the model is using reasonable patterns
        - **⚖️ Fairness:** Detect potential biases in predictions
        - **📊 Insights:** Discover which features matter most for pricing
        """)

with tab4:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This is a car price prediction system built using machine learning regression models.
    The system analyzes various car features to estimate market value.
    
    ### 🛠️ Technology Stack
    - **Frontend:** Streamlit
    - **ML Models:** Linear Regression, Random Forest, XGBoost
    - **Hyperparameter Tuning:** RandomizedSearchCV / GridSearchCV
    - **Model Explainability:** SHAP (SHapley Additive exPlanations)
    - **Data Processing:** Pandas, NumPy, Scikit-learn
    - **Visualization:** Plotly, Matplotlib, Seaborn
    
    ### 📊 Model Selection
    The system automatically selects the best-performing model based on R² score on test data.
    """)
    
    if model_type == "tuned":
        st.markdown(f"""
        **Current deployed model:** {metadata.get('model_name', 'Tuned XGBoost')}  
        **Performance:** {metadata['metrics']['test_r2']:.4f} R² Score  
        **Optimization:** Hyperparameter-tuned for optimal performance
        """)
    else:
        st.markdown(f"""
        **Current deployed model:** {metadata['model_name']}  
        **Performance:** {metadata['metrics']['test_r2']:.4f} R² Score
        """)
    
    st.markdown("""
    ### 🎓 Use Cases
    - Car dealerships for pricing optimization
    - Individual sellers for market value estimation
    - Buyers for fair price validation
    - Market trend analysis
    
    ### ✨ Advanced Features
    - ⚡ **Hyperparameter Tuning:** Automated optimization for best performance
    - 🔍 **SHAP Explainability:** Understand why predictions are made
    - 📊 **Interactive Visualizations:** Explore model insights visually
    - 🎯 **Confidence Intervals:** Know the prediction uncertainty
    
    ### 👨‍💻 Developer
    Created as part of Data Science internship training.
    
    ### 📧 Contact
    For questions or feedback, please reach out via GitHub.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7F8C8D;'>
    <p>🚗 Car Price Prediction System | Built with ❤️ using Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)