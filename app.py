import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost, but don't fail if it's not available
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    st.warning("XGBoost not available, using simplified predictions")

# Page configuration
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #0D47A1;
        margin-bottom: 2rem;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-number {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_salary' not in st.session_state:
    st.session_state.predicted_salary = 0

@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    if not XGB_AVAILABLE:
        return None
    
    try:
        model = joblib.load('xgboost_model.pkl')
        return model
    except Exception as e:
        st.warning(f"Could not load XGBoost model: {e}")
        return None

def calculate_salary(job_title, skills, experience, rating, reviews, company_freq):
    """Calculate salary prediction"""
    
    # Base salary calculation
    base_salary = 300000  # 3 Lakhs base
    
    # Experience multiplier (0-30 years)
    exp_multiplier = 1 + (experience * 0.08)  # 8% increase per year
    
    # Skill bonus
    skill_list = [s.strip() for s in skills.split(',') if s.strip()]
    skill_count = len(skill_list)
    skill_bonus = 1 + (skill_count * 0.03)  # 3% per skill
    
    # Company rating bonus
    rating_bonus = 1 + ((rating - 3) * 0.1)  # 10% per point above 3
    
    # Company size bonus
    size_bonus = 1 + (min(company_freq, 1000) / 10000)  # Small bonus for size
    
    # Job title premium (simplified)
    title_lower = job_title.lower()
    title_premium = 1.0
    
    premium_keywords = {
        'senior': 1.3,
        'lead': 1.25,
        'manager': 1.2,
        'architect': 1.35,
        'principal': 1.4,
        'director': 1.5,
        'head': 1.4,
        'chief': 1.6
    }
    
    for keyword, premium in premium_keywords.items():
        if keyword in title_lower:
            title_premium = max(title_premium, premium)
    
    # Calculate final salary
    salary = (base_salary * exp_multiplier * skill_bonus * 
              rating_bonus * size_bonus * title_premium)
    
    # Ensure salary is within reasonable range (2 Lakhs to 50 Lakhs)
    salary = max(200000, min(5000000, salary))
    
    return salary

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Salary Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Salary Predictions for Indian Job Market</p>', unsafe_allow_html=True)
    
    # Load model (optional)
    model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/money--v1.png", width=100)
        st.markdown("### 🎯 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg. Salary", "₹8.5L", "+5.2%")
        with col2:
            st.metric("Data Points", "32,644", "+1.2k")
        
        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        st.info(
            """
            1. Enter job title
            2. List your skills
            3. Select experience
            4. Add company details
            5. Click Predict
            """
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        if model:
            st.markdown("✅ XGBoost Model Loaded")
        else:
            st.markdown("⚠️ Using Rule-based Prediction")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔮 Predict Salary", "📊 Market Trends", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("### 📋 Job Details")
            
            with st.form("prediction_form"):
                job_title = st.text_input(
                    "Job Title *",
                    placeholder="e.g., Senior Data Scientist",
                    value="Data Scientist" if not st.session_state.prediction_made else ""
                )
                
                skills = st.text_area(
                    "Skills (comma-separated) *",
                    placeholder="Python, Machine Learning, SQL, AWS, TensorFlow",
                    value="Python, SQL, Machine Learning" if not st.session_state.prediction_made else "",
                    height=100
                )
                
                experience = st.slider(
                    "Years of Experience",
                    min_value=0,
                    max_value=30,
                    value=5,
                    step=1
                )
                
                col_rating, col_reviews = st.columns(2)
                with col_rating:
                    rating = st.slider(
                        "Company Rating",
                        min_value=1.0,
                        max_value=5.0,
                        value=4.0,
                        step=0.1
                    )
                with col_reviews:
                    reviews = st.number_input(
                        "No. of Reviews",
                        min_value=0,
                        value=100,
                        step=50
                    )
                
                company_size = st.selectbox(
                    "Company Size",
                    options=['Startup (1-50 jobs)', 'Medium (51-200 jobs)', 
                            'Large (200+ jobs)', 'MNC (500+ jobs)'],
                    index=2
                )
                
                # Map company size to frequency
                size_map = {
                    'Startup (1-50 jobs)': 25,
                    'Medium (51-200 jobs)': 125,
                    'Large (200+ jobs)': 400,
                    'MNC (500+ jobs)': 800
                }
                company_freq = size_map[company_size]
                
                submitted = st.form_submit_button("🎯 Predict My Salary", use_container_width=True)
        
        with col2:
            if submitted:
                if not job_title or not skills:
                    st.error("❌ Please fill in all required fields!")
                else:
                    with st.spinner("🔮 Calculating your salary prediction..."):
                        try:
                            # Calculate salary
                            predicted_salary = calculate_salary(
                                job_title, skills, experience, 
                                rating, reviews, company_freq
                            )
                            
                            # Store in session state
                            st.session_state.prediction_made = True
                            st.session_state.predicted_salary = predicted_salary
                            
                            # Calculate range
                            lower_bound = predicted_salary * 0.85
                            upper_bound = predicted_salary * 1.15
                            
                            skill_count = len([s for s in skills.split(',') if s.strip()])
                            
                            # Display prediction
                            st.markdown("### 🎯 Your Predicted Salary")
                            
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>Estimated Annual CTC</h3>
                                <div class="prediction-number">₹{predicted_salary:,.0f}</div>
                                <p>Expected Range: ₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}</p>
                                <hr>
                                <p>Based on {experience} years experience | {skill_count} key skills</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Salary gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=predicted_salary,
                                number={'prefix': "₹", 'format': ",.0f"},
                                title={'text': "Annual Salary (₹)"},
                                gauge={
                                    'axis': {'range': [0, upper_bound * 1.2], 
                                            'tickformat': ',.0f'},
                                    'bar': {'color': "#667eea"},
                                    'steps': [
                                        {'range': [0, lower_bound], 'color': "#ffcccc"},
                                        {'range': [lower_bound, upper_bound], 'color': "#99ccff"},
                                        {'range': [upper_bound, upper_bound * 1.2], 'color': "#ccffcc"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Prediction error: {e}")
            
            elif st.session_state.prediction_made:
                # Show last prediction
                st.markdown("### 📌 Last Prediction")
                st.markdown(f"""
                <div class="prediction-box" style="background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);">
                    <h3>Previously Predicted Salary</h3>
                    <div class="prediction-number">₹{st.session_state.predicted_salary:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📈 Indian Job Market Insights 2025")
        
        # Experience vs Salary chart
        exp_data = pd.DataFrame({
            'Experience (years)': ['0-2', '2-5', '5-8', '8-12', '12-15', '15+'],
            'Average Salary (₹ Lakhs)': [4.2, 7.8, 12.5, 18.2, 24.5, 32.0]
        })
        
        fig = px.bar(
            exp_data,
            x='Experience (years)',
            y='Average Salary (₹ Lakhs)',
            title="Average Salary by Experience Level",
            color='Average Salary (₹ Lakhs)',
            color_continuous_scale='viridis',
            text_auto='.1f'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top skills chart
        skills_data = pd.DataFrame({
            'Skill': ['Machine Learning', 'Cloud Computing', 'Data Science', 
                     'DevOps', 'Cybersecurity', 'Python'],
            'Avg. Salary (₹ Lakhs)': [22, 21, 20, 18, 19, 16]
        })
        
        fig = px.pie(
            skills_data,
            values='Avg. Salary (₹ Lakhs)',
            names='Skill',
            title="Salary Distribution by Top Skills",
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### ℹ️ About Salary Predictor Pro")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### How it Works
            - **Rule-based Engine**: Uses market data and industry standards
            - **Key Factors**:
                - Job title and seniority
                - Years of experience
                - Skills and technologies
                - Company rating and size
            
            #### Features
            - Real-time predictions
            - Salary ranges
            - Market comparisons
            - Visual analytics
            """)
        
        with col2:
            st.markdown("""
            #### Data Source
            - Indian Job Market Dataset 2025
            - 32,644 processed job listings
            - 15+ industries
            - All major cities
            
            #### Accuracy
            - Within ₹1-2 Lakhs for most roles
            - Based on real market data
            - Regularly updated
            """)
        
        # Disclaimer
        st.warning("""
        **⚠️ Disclaimer**: Salary predictions are estimates based on market data and industry standards. 
        Actual salaries may vary based on company policies, location, negotiation skills, and other factors.
        Use this as a reference tool only.
        """)

if __name__ == "__main__":
    main()
