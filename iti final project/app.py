import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Smart Recruitment Assistant",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Smart Recruitment Assistant")
st.markdown("### Predict if a candidate should advance to the next stage")

# ============================================
# LOAD MODEL AND PREPROCESSOR
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model and preprocessor"""
    
    # Try loading v2 models first (with preprocessor)
    try:
        model = joblib.load('best_model_v2.pkl')
        preprocessor = joblib.load('preprocessor_v2.pkl')
        st.success("✅ Model v2 loaded successfully!")
        return model, preprocessor, True
    except Exception as e:
        st.warning(f"⚠️ Could not load v2 model: {str(e)}")
    
    # Try loading model only (old version)
    try:
        model = joblib.load('best_model.pkl')
        st.success("✅ Model loaded successfully!")
        return model, None, False
    except:
        pass
    
    # Try loading pickle version
    try:
        with open('best_model.pickle', 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully using pickle!")
        return model, None, False
    except Exception as e:
        st.error(f"❌ Model not found!\nError: {str(e)}")
        
        # Show available files
        st.write("📁 Files in current directory:")
        for file in os.listdir():
            st.write(f"  - {file}")
        return None, None, False

model, preprocessor, is_v2 = load_model()

# ============================================
# FEATURE NAMES FOR V2 MODEL
# ============================================
if is_v2:
    # These are the 15 features in the exact order expected by the preprocessor
    V2_FEATURES = [
        'city_development_index',
        'gender',
        'relevent_experience',
        'enrolled_university',
        'education_level',
        'major_discipline',
        'experience_cleaned',
        'company_size',
        'company_type',
        'last_new_job',
        'training_hours_capped',
        'experience_ratio',
        'training_per_experience',
        'has_relevant_exp',
        'exp_category'
    ]

# ============================================
# CANDIDATE INPUT FORM
# ============================================
if model is not None:
    with st.form("candidate_form"):
        st.subheader("📝 Candidate Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other', 'Unknown'])
            education = st.selectbox("Education Level", ['Graduate', 'Masters', 'Phd', 'High School', 'Primary School', 'Unknown'])
            experience = st.number_input("Years of Experience", min_value=0, max_value=25, value=5, step=1)
            training_hours = st.number_input("Training Hours", min_value=0, max_value=500, value=50, step=5)
        
        with col2:
            city_index = st.slider("City Development Index", 0.0, 1.0, 0.8, 0.01)
            relevant_exp = st.selectbox("Relevant Experience", ['Has relevent experience', 'No relevent experience'])
            company_size = st.selectbox("Company Size", ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+', 'Unknown'])
            company_type = st.selectbox("Company Type", ['Pvt Ltd', 'Public Sector', 'Funded Startup', 'Early Stage Startup', 'NGO', 'Other', 'Unknown'])
            last_new_job = st.selectbox("Last New Job", ['never', '1', '2', '3', '4', '>4', 'Unknown'])
        
        # Experience Category (only for v2 model)
        if is_v2:
            exp_category = st.selectbox(
                "Experience Category",
                ['Junior (0-2 years)', 'Mid (3-5 years)', 'Senior (6-10 years)', 'Expert (10+ years)', 'Unknown']
            )
        
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
    
    # ============================================
    # PREDICTION
    # ============================================
    if submitted:
        try:
            # Prepare input data based on model version
            if is_v2:
                # Map experience category
                exp_cat_map = {
                    'Junior (0-2 years)': 'junior',
                    'Mid (3-5 years)': 'mid',
                    'Senior (6-10 years)': 'senior',
                    'Expert (10+ years)': 'expert',
                    'Unknown': 'unknown'
                }
                
                # Create input data dictionary
                input_data = {
                    'city_development_index': [city_index],
                    'gender': [gender.lower()],
                    'relevent_experience': [relevant_exp.lower()],
                    'enrolled_university': ['unknown'],
                    'education_level': [education.lower()],
                    'major_discipline': ['unknown'],
                    'experience_cleaned': [experience],
                    'company_size': [company_size.lower()],
                    'company_type': [company_type.lower()],
                    'last_new_job': [last_new_job.lower()],
                    'training_hours_capped': [min(training_hours, 302)],  # استفاده از 302 چون این مقدار cap شده
                    'experience_ratio': [experience / (city_index + 0.1)],
                    'training_per_experience': [training_hours / (experience + 1)],
                    'has_relevant_exp': [1 if relevant_exp == 'Has relevent experience' else 0],
                    'exp_category': [exp_cat_map[exp_category]]
                }
                
                # Create DataFrame with correct column order
                input_df = pd.DataFrame(input_data)
                
                # Ensure columns are in the correct order
                input_df = input_df[V2_FEATURES]
                
                # IMPORTANT FIX: Use model directly, not preprocessor.transform
                # The model already contains the preprocessor inside it (as a Pipeline)
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
            else:
                # Old model - use 14 features (without exp_category)
                input_df = pd.DataFrame({
                    'city_development_index': [city_index],
                    'gender': [gender.lower()],
                    'relevent_experience': [relevant_exp.lower()],
                    'enrolled_university': ['unknown'],
                    'education_level': [education.lower()],
                    'major_discipline': ['unknown'],
                    'experience_cleaned': [experience],
                    'company_size': [company_size.lower()],
                    'company_type': [company_type.lower()],
                    'last_new_job': [last_new_job.lower()],
                    'training_hours': [training_hours],
                    'experience_ratio': [experience / (city_index + 0.1)],
                    'training_per_experience': [training_hours / (experience + 1)],
                    'has_relevant_exp': [1 if relevant_exp == 'Has relevent experience' else 0]
                })
                
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
            
            # ============================================
            # DISPLAY RESULTS
            # ============================================
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Prediction",
                    value="✅ Advance" if prediction == 1 else "❌ Do Not Advance"
                )
            
            with col2:
                st.metric(
                    label="Probability",
                    value=f"{probability:.2%}"
                )
            
            with col3:
                status = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
                st.metric(
                    label="Confidence Level",
                    value=status
                )
            
            # Progress bar
            st.progress(probability)
            
            # Recommendation message
            if prediction == 1:
                st.success(f"✅ This candidate is recommended to advance with {probability:.1%} confidence.")
            else:
                st.warning(f"⚠️ This candidate is not recommended to advance.")
            
            # Show input summary
            with st.expander("📋 View Input Summary"):
                st.dataframe(input_df)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all inputs are filled correctly.")
            
            # Debug info
            with st.expander("🔧 Debug Info"):
                st.write("Model Version:", "v2" if is_v2 else "v1")
                if 'input_df' in locals():
                    st.write("Input DataFrame shape:", input_df.shape)
                    st.write("Input Columns:", list(input_df.columns))
                    st.write("Input Data:")
                    st.dataframe(input_df)
                else:
                    st.write("Input DataFrame not created yet")
    
    # ============================================
    # TOP 10 CANDIDATES
    # ============================================
    st.header("🏆 Top 10 Recommended Candidates")
    
    # Try loading v2 first
    try:
        top_10 = pd.read_csv('top_10_candidates_v2.csv')
        st.dataframe(
            top_10[['enrollee_id', 'probability', 'city_development_index', 
                   'experience_cleaned', 'training_hours']].head(10),
            use_container_width=True
        )
    except:
        try:
            # Try old version
            top_10 = pd.read_csv('top_10_candidates.csv')
            st.dataframe(top_10[['enrollee_id', 'probability']].head(10))
        except:
            st.info("Top candidates data not available. Run the model first.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        Made with ❤️ using Streamlit | Smart Recruitment Assistant v2
    </div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - INFORMATION
# ============================================
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    ### Smart Recruitment Assistant
    
    This application uses Machine Learning to predict whether a candidate 
    should advance to the next stage of the hiring process.
    
    **Features used:**
    - Personal Information (Gender, Education, Experience)
    - Professional Background (Company, Job History)
    - Training Hours
    - City Development Index
    - Experience Category (v2)
    
    **Model:** Random Forest Classifier
    
    **Accuracy:** ~78%
    
    **AUC:** ~0.81
    """)
    
    st.divider()
    
    st.header("📊 Model Info")
    st.write(f"**Model Version:** {'v2 (with exp_category)' if is_v2 else 'v1'}")
    
    if is_v2:
        st.write("**Features (15):**")
        for i, feat in enumerate(V2_FEATURES, 1):
            st.write(f"  {i}. {feat}")
    
    try:
        # Try to load submission file for stats
        sub = pd.read_csv('submission_v2.csv')
        st.metric("Total Predictions", len(sub))
        st.metric("Will Advance", int(sub['prediction'].sum()))
        st.metric("Advance Rate", f"{sub['prediction'].mean() * 100:.1f}%")
    except:
        try:
            sub = pd.read_csv('submission.csv')
            st.metric("Total Predictions", len(sub))
            st.metric("Will Advance", int(sub['prediction'].sum()))
            st.metric("Advance Rate", f"{sub['prediction'].mean() * 100:.1f}%")
        except:
            pass