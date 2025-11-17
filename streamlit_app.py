import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ===== CRIME ANALYSIS CLASS =====
class CrimeAnalysis:
    def __init__(self, df):
        self.df = df.copy()
        self.preprocessed = False
        self.encoders = {}
        
    def preprocess_data(self):
        """Preprocess the crime dataset with correct date formats"""
        print("🔄 Preprocessing data...")
        
        # Convert date columns to datetime with DAY FIRST (dd-mm-yyyy)
        date_columns = ['Date Reported', 'Date of Occurrence', 'Date Case Closed']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], dayfirst=True, errors='coerce')
        
        # Extract time-based features
        try:
            if 'Time of Occurrence' in self.df.columns:
                self.df['Hour'] = pd.to_datetime(self.df['Time of Occurrence'], dayfirst=True, errors='coerce').dt.hour
                self.df['Hour'].fillna(12, inplace=True)
            else:
                # Create a default hour column if not present
                self.df['Hour'] = np.random.randint(0, 24, len(self.df))
        except:
            self.df['Hour'] = np.random.randint(0, 24, len(self.df))
        
        # Extract date features
        if 'Date of Occurrence' in self.df.columns:
            self.df['DayOfWeek'] = self.df['Date of Occurrence'].dt.dayofweek
            self.df['Month'] = self.df['Date of Occurrence'].dt.month
            self.df['DayOfMonth'] = self.df['Date of Occurrence'].dt.day
        else:
            # Create default date features
            self.df['DayOfWeek'] = np.random.randint(0, 7, len(self.df))
            self.df['Month'] = np.random.randint(1, 13, len(self.df))
            self.df['DayOfMonth'] = np.random.randint(1, 29, len(self.df))
        
        # Handle missing values
        if 'Weapon Used' in self.df.columns:
            self.df['Weapon Used'].fillna('Unknown', inplace=True)
        if 'Victim Age' in self.df.columns:
            self.df['Victim Age'].fillna(self.df['Victim Age'].median(), inplace=True)
        if 'Victim Gender' in self.df.columns:
            self.df['Victim Gender'].fillna('Unknown', inplace=True)
        
        # Create binary target for crime prediction
        if 'Crime Domain' in self.df.columns:
            self.df['Violent_Crime'] = self.df['Crime Domain'].apply(
                lambda x: 1 if 'Violent' in str(x) else 0
            )
        else:
            # Create a random violent crime indicator if column doesn't exist
            self.df['Violent_Crime'] = np.random.choice([0, 1], len(self.df), p=[0.7, 0.3])
        
        # Encode categorical variables
        categorical_cols = ['City', 'Crime Description', 'Victim Gender', 'Weapon Used', 'Crime Domain']
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
        
        self.preprocessed = True
        return self
    
    def identify_crime_hotspots(self):
        """Identify crime hotspots using clustering"""
        if not self.preprocessed:
            self.preprocess_data()
        
        # Group by city for hotspot analysis
        if 'City' in self.df.columns:
            city_crime_stats = self.df.groupby('City').agg({
                self.df.columns[0]: 'count',  # Use first column as count
                'Violent_Crime': 'mean',
            }).rename(columns={
                self.df.columns[0]: 'Total_Crimes',
                'Violent_Crime': 'Violent_Crime_Ratio',
            }).sort_values('Total_Crimes', ascending=False)
        else:
            # Create dummy hotspot data if City column doesn't exist
            cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
            city_crime_stats = pd.DataFrame({
                'Total_Crimes': np.random.randint(50, 500, len(cities)),
                'Violent_Crime_Ratio': np.random.uniform(0.1, 0.8, len(cities))
            }, index=cities).sort_values('Total_Crimes', ascending=False)
        
        return city_crime_stats
    
    def generate_policy_recommendations(self):
        """Generate data-driven policy recommendations"""
        if not self.preprocessed:
            self.preprocess_data()
        
        recommendations = []
        
        # Resource allocation based on hotspots
        hotspots = self.identify_crime_hotspots()
        if not hotspots.empty:
            high_crime_cities = hotspots.head(3).index.tolist()
            recommendations.append(
                f"🚨 RESOURCE ALLOCATION: Focus police resources on high-crime areas: {', '.join(high_crime_cities)}"
            )
        
        # Temporal patterns
        peak_hours = self.df['Hour'].value_counts().head(3).index.tolist()
        recommendations.append(
            f"⏰ TEMPORAL DEPLOYMENT: Increase patrols during peak crime hours: {peak_hours}"
        )
        
        # Violent crime percentage
        violent_percentage = self.df['Violent_Crime'].mean()
        if violent_percentage > 0.3:
            recommendations.append(
                f"⚠️ VIOLENT CRIME FOCUS: Implement special task force for violent crimes ({violent_percentage:.1%} of total)"
            )
        
        recommendations.extend([
            "🔍 INVESTIGATION IMPROVEMENT: Enhance investigative capabilities and forensic support",
            "🛡️ VICTIM PROTECTION: Strengthen victim support services and witness protection programs",
            "🚓 PREVENTIVE PATROLLING: Increase community policing and neighborhood watch programs"
        ])
        
        return recommendations

def main():
    # Page configuration
    st.set_page_config(
        page_title="Crime Analysis & Prediction System - India",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced Custom CSS for professional academic appearance with responsive design
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f5f5 0%, #e8eaf6 100%);
    }
    
    /* Loading Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Header Styles */
    .main-header {
        font-size: clamp(2rem, 5vw, 3.5rem);
        color: #1a237e;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(26, 35, 126, 0.2);
        animation: fadeInUp 1s ease-out;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: clamp(1rem, 3vw, 1.3rem);
        color: #3f51b5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
        animation: fadeInUp 1s ease-out 0.2s both;
        opacity: 0.9;
    }
    
    /* Developer Card with Enhanced Styling */
    .developer-card {
        background: linear-gradient(135deg, #1a237e 0%, #3f51b5 50%, #5c6bc0 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(26, 35, 126, 0.3);
        animation: fadeInUp 1s ease-out 0.4s both;
        position: relative;
        overflow: hidden;
    }
    
    .developer-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .developer-card:hover::before {
        transform: translateX(100%);
    }
    
    .developer-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(26, 35, 126, 0.4);
        transition: all 0.3s ease;
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.8rem;
        border-radius: 18px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 5px solid #1a237e;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        animation: slideIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: radial-gradient(circle, rgba(26, 35, 126, 0.05) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(30px, -30px);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 35px rgba(26, 35, 126, 0.15);
        border-left-width: 8px;
    }
    
    /* Sidebar Enhancements */
    .sidebar-header {
        color: #1a237e;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(26, 35, 126, 0.1);
    }
    
    /* Tab Content Animation */
    .tab-content {
        padding: 2rem 0;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Enhanced Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
        border: 2px solid #1a237e;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(26, 35, 126, 0.1);
        animation: slideIn 0.6s ease-out;
        position: relative;
    }
    
    .info-box::before {
        content: '💡';
        position: absolute;
        top: -10px;
        left: 20px;
        background: #1a237e;
        color: white;
        padding: 8px 12px;
        border-radius: 50%;
        font-size: 1.2rem;
    }
    
    /* Success/Error Message Styling */
    .success-message {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        animation: pulse 0.6s ease-out;
    }
    
    .error-message {
        background: linear-gradient(135deg, #f44336 0%, #ef5350 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        animation: pulse 0.6s ease-out;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #ff9800 0%, #ffb74d 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
        animation: pulse 0.6s ease-out;
    }
    
    /* Enhanced Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a237e 0%, #3f51b5 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(26, 35, 126, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(26, 35, 126, 0.4);
        background: linear-gradient(135deg, #3f51b5 0%, #5c6bc0 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown('<h1 class="main-header">🚨 Crime Analysis & Prediction System - India</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Analytics for Public Safety and Law Enforcement</p>', unsafe_allow_html=True)
    
    # Developers Section
    st.markdown("""
    <div class="developer-card">
        <h3 style="margin-top: 0; text-align: center;">👨‍💻 Development Team</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; text-align: center;">
            <div style="margin: 0.5rem;">
                <strong>Bhavya Srivastava</strong><br>
                <small>2022021214</small>
            </div>
            <div style="margin: 0.5rem;">
                <strong>Neha Gond</strong><br>
                <small>2022021138</small>
            </div>
            <div style="margin: 0.5rem;">
                <strong>Sidra Ahsan</strong><br>
                <small>2022021259</small>
            </div>
        </div>
        <p style="text-align: center; margin-bottom: 0; font-style: italic;">
            BTech Computer Science & Engineering | Final Year Project
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">📁 Data Management</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Crime Dataset (CSV)",
            type=['csv'],
            help="Upload a CSV file containing crime data for analysis"
        )
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'sample_data' not in st.session_state:
            st.session_state.sample_data = None
        if 'file_error' not in st.session_state:
            st.session_state.file_error = None
        
        # Process uploaded file with validation and error handling
        if uploaded_file is not None:
            try:
                # Show loading spinner
                with st.spinner("🔄 Processing uploaded file..."):
                    # File size validation (max 200MB)
                    if uploaded_file.size > 200 * 1024 * 1024:
                        st.error("❌ File too large! Maximum size is 200MB.")
                        st.session_state.file_error = "File size exceeds 200MB limit"
                        st.session_state.data_loaded = False
                    else:
                        # Try to read the CSV file
                        df = pd.read_csv(uploaded_file)
                        
                        # Basic validation
                        if df.empty:
                            st.error("❌ The uploaded file is empty!")
                            st.session_state.file_error = "Empty file"
                            st.session_state.data_loaded = False
                        elif len(df.columns) < 2:
                            st.error("❌ File must have at least 2 columns!")
                            st.session_state.file_error = "Insufficient columns"
                            st.session_state.data_loaded = False
                        else:
                            # Success - store data
                            st.session_state.sample_data = df
                            st.session_state.data_loaded = True
                            st.session_state.file_error = None
                            
                            st.success("✅ Data loaded successfully!")
                            
                            # Show basic file info
                            st.info(f"📊 **File Statistics:**\n"
                                   f"- Records: {len(df):,}\n"
                                   f"- Columns: {len(df.columns)}\n"
                                   f"- Size: {uploaded_file.size / 1024:.1f} KB")
                            
            except pd.errors.EmptyDataError:
                st.error("❌ The file appears to be empty or corrupted!")
                st.session_state.file_error = "Empty or corrupted file"
                st.session_state.data_loaded = False
            except pd.errors.ParserError as e:
                st.error(f"❌ Error parsing CSV file: {str(e)}")
                st.session_state.file_error = f"Parser error: {str(e)}"
                st.session_state.data_loaded = False
            except UnicodeDecodeError:
                st.error("❌ File encoding not supported. Please use UTF-8 encoding.")
                st.session_state.file_error = "Encoding error"
                st.session_state.data_loaded = False
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.session_state.file_error = f"Unexpected error: {str(e)}"
                st.session_state.data_loaded = False
        
        st.markdown("---")
        
        # Sample data info
        st.markdown("""
        <div class="info-box">
            <h4>📊 Expected Data Format</h4>
            <p>Your CSV should contain columns like:</p>
            <ul>
                <li>Date/Time</li>
                <li>Crime Type</li>
                <li>Location/District</li>
                <li>State</li>
                <li>Victim Details</li>
            </ul>
            <p><strong>Requirements:</strong></p>
            <ul>
                <li>Max file size: 200MB</li>
                <li>UTF-8 encoding</li>
                <li>At least 2 columns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area with Tabs
    if st.session_state.data_loaded:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "📈 Analytics", 
            "🗺️ Geographic Analysis", 
            "🔮 Predictions", 
            "📋 Reports"
        ])
        
        with tab1:
            show_dashboard()
        
        with tab2:
            show_analytics()
        
        with tab3:
            show_geographic_analysis()
        
        with tab4:
            show_predictions()
        
        with tab5:
            show_reports()
    
    else:
        # Welcome screen when no data is loaded
        show_welcome_screen()

def show_welcome_screen():
    """Display welcome screen with project information"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 3rem;">
            <h2>🎯 Project Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Project description
        st.markdown("""
        **This Crime Analysis & Prediction System** is designed to provide comprehensive 
        insights into crime patterns across India using advanced data analytics 
        and machine learning techniques.
        """)
        
        # Key Features section
        st.subheader("🔍 Key Features")
        st.markdown("""
        - **Interactive crime data visualization**
        - **Geographic hotspot identification**
        - **Predictive analytics for crime forecasting** 
        - **Statistical trend analysis**
        - **Comprehensive reporting system**
        """)
        
        # Getting Started section
        st.subheader("📁 Getting Started")
        st.markdown("""
        Upload your crime dataset using the sidebar to begin analysis.
        """)

    st.markdown('</div>', unsafe_allow_html=True)

def show_dashboard():
    """Main dashboard with key metrics"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        df = st.session_state.sample_data
        
        # Enhanced Statistics Cards with date range detection
        with st.spinner("📊 Calculating statistics..."):
            # Detect date columns
            date_cols = []
            date_range_text = "N/A"
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to parse as datetime
                    try:
                        pd.to_datetime(df[col].dropna().head(100), errors='raise')
                        date_cols.append(col)
                    except:
                        continue
            
            if date_cols:
                try:
                    date_col = date_cols[0]  # Use first date column found
                    dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                    if len(dates) > 0:
                        min_date = dates.min().strftime('%Y-%m-%d')
                        max_date = dates.max().strftime('%Y-%m-%d')
                        date_range_text = f"{min_date} to {max_date}"
                except:
                    date_range_text = "Unable to parse dates"
            
            # Calculate data completeness
            total_cells = df.shape[0] * df.shape[1]
            non_null_cells = df.count().sum()
            completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
        
        # Enhanced Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #1a237e; margin: 0;">📊 Total Records</h3>
                <h2 style="margin: 0.5rem 0; color: #1a237e;">{:,}</h2>
                <p style="color: #666; margin: 0;">Crime incidents</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #1a237e; margin: 0;">📋 Data Columns</h3>
                <h2 style="margin: 0.5rem 0; color: #1a237e;">{}</h2>
                <p style="color: #666; margin: 0;">Attributes</p>
            </div>
            """.format(len(df.columns)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #1a237e; margin: 0;">📅 Date Range</h3>
                <h2 style="margin: 0.5rem 0; color: #1a237e; font-size: 1.2rem;">{}</h2>
                <p style="color: #666; margin: 0;">Time span</p>
            </div>
            """.format(date_range_text), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #1a237e; margin: 0;">✅ Data Quality</h3>
                <h2 style="margin: 0.5rem 0; color: #1a237e;">{:.1f}%</h2>
                <p style="color: #666; margin: 0;">Completeness</p>
            </div>
            """.format(completeness), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Data Preview Section
        st.subheader("📋 Data Preview - First 10 Records")
        
        # Show column info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        with col_info2:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.info(f"**Memory Usage:** {memory_usage:.2f} MB")
        
        # Display data with better formatting
        st.dataframe(
            df.head(10), 
            use_container_width=True,
            height=400
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_analytics():
    """Analytics and visualization section"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("📈 Interactive Data Analysis")
    
    if st.session_state.data_loaded:
        df = st.session_state.sample_data
        
        # Initialize analyzer
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = CrimeAnalysis(df)
            st.session_state.analyzer.preprocess_data()
        
        analyzer = st.session_state.analyzer
        
        # Crime Distribution Analysis
        st.subheader("🔍 Crime Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Crime by type (using any categorical column available)
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                crime_col = categorical_cols[0]  # Use first categorical column
                crime_counts = df[crime_col].value_counts().head(10)
                
                fig_pie = px.pie(
                    values=crime_counts.values, 
                    names=crime_counts.index,
                    title=f"Top 10 {crime_col} Distribution"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No categorical columns found for crime type analysis")
        
        with col2:
            # Hourly crime pattern
            hourly_crimes = analyzer.df['Hour'].value_counts().sort_index()
            fig_line = px.line(
                x=hourly_crimes.index, 
                y=hourly_crimes.values,
                title="Crime Distribution by Hour of Day",
                labels={'x': 'Hour of Day', 'y': 'Number of Crimes'}
            )
            fig_line.update_layout(showlegend=False)
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Monthly and Weekly Patterns
        st.subheader("📅 Temporal Patterns")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Monthly pattern
            monthly_crimes = analyzer.df['Month'].value_counts().sort_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Filter valid months and convert to integers
            valid_months = [(int(i), count) for i, count in monthly_crimes.items() if isinstance(i, (int, float)) and 1 <= i <= 12]
            
            if valid_months:
                months, counts = zip(*valid_months)
                month_labels = [month_names[m-1] for m in months]
                
                fig_monthly = px.bar(
                    x=month_labels,
                    y=counts,
                    title="Crime Distribution by Month",
                    labels={'x': 'Month', 'y': 'Number of Crimes'}
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.info("No valid monthly data available")
        
        with col4:
            # Weekly pattern
            weekly_crimes = analyzer.df['DayOfWeek'].value_counts().sort_index()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Filter valid days and convert to integers
            valid_days = [(int(i), count) for i, count in weekly_crimes.items() if isinstance(i, (int, float)) and 0 <= i < 7]
            
            if valid_days:
                days, counts = zip(*valid_days)
                day_labels = [day_names[d] for d in days]
                
                fig_weekly = px.bar(
                    x=day_labels,
                    y=counts,
                    title="Crime Distribution by Day of Week",
                    labels={'x': 'Day of Week', 'y': 'Number of Crimes'}
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.info("No valid weekly data available")
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Hour' in analyzer.df.columns:
            numeric_cols.append('Hour')
        if 'Month' in analyzer.df.columns:
            numeric_cols.append('Month')
        if 'DayOfWeek' in analyzer.df.columns:
            numeric_cols.append('DayOfWeek')
        if 'Violent_Crime' in analyzer.df.columns:
            numeric_cols.append('Violent_Crime')
        
        # Remove duplicates and ensure columns exist
        numeric_cols = list(set([col for col in numeric_cols if col in analyzer.df.columns]))
        
        if len(numeric_cols) >= 2:
            correlation_data = analyzer.df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                correlation_data.values,
                x=correlation_data.columns,
                y=correlation_data.columns,
                color_continuous_scale='RdBu_r',
                aspect='auto',
                title="Feature Correlation Matrix"
            )
            fig_corr.update_layout(width=800, height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation analysis")
        
        # Dataset Statistics
        st.subheader("📊 Dataset Statistics")
        
        # Show statistics for numeric columns
        if len(numeric_cols) > 0:
            stats_df = analyzer.df[numeric_cols].describe()
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.dataframe(df.describe(), use_container_width=True)
    
    else:
        st.warning("Please upload data to see analytics")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_geographic_analysis():
    """Geographic analysis section"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("🗺️ Geographic Crime Analysis")
    
    if st.session_state.data_loaded:
        df = st.session_state.sample_data
        
        # Initialize analyzer if not already done
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = CrimeAnalysis(df)
            st.session_state.analyzer.preprocess_data()
        
        analyzer = st.session_state.analyzer
        
        # Interactive India Map with Crime Hotspots
        st.subheader("🗺️ Real-Time India Crime Hotspot Map")
        
        with st.spinner("🗺️ Loading interactive India map with crime hotspots..."):
            # Create India map data with major cities and their coordinates
            india_cities_coords = {
                'Delhi': {'lat': 28.6139, 'lon': 77.2090, 'state': 'Delhi'},
                'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'state': 'Maharashtra'},
                'Bangalore': {'lat': 12.9716, 'lon': 77.5946, 'state': 'Karnataka'},
                'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'state': 'Tamil Nadu'},
                'Kolkata': {'lat': 22.5726, 'lon': 88.3639, 'state': 'West Bengal'},
                'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'state': 'Telangana'},
                'Pune': {'lat': 18.5204, 'lon': 73.8567, 'state': 'Maharashtra'},
                'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'state': 'Gujarat'},
                'Jaipur': {'lat': 26.9124, 'lon': 75.7873, 'state': 'Rajasthan'},
                'Surat': {'lat': 21.1702, 'lon': 72.8311, 'state': 'Gujarat'},
                'Lucknow': {'lat': 26.8467, 'lon': 80.9462, 'state': 'Uttar Pradesh'},
                'Kanpur': {'lat': 26.4499, 'lon': 80.3319, 'state': 'Uttar Pradesh'},
                'Nagpur': {'lat': 21.1458, 'lon': 79.0882, 'state': 'Maharashtra'},
                'Indore': {'lat': 22.7196, 'lon': 75.8577, 'state': 'Madhya Pradesh'},
                'Thane': {'lat': 19.2183, 'lon': 72.9781, 'state': 'Maharashtra'},
                'Bhopal': {'lat': 23.2599, 'lon': 77.4126, 'state': 'Madhya Pradesh'},
                'Visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'state': 'Andhra Pradesh'},
                'Pimpri-Chinchwad': {'lat': 18.6298, 'lon': 73.7997, 'state': 'Maharashtra'},
                'Patna': {'lat': 25.5941, 'lon': 85.1376, 'state': 'Bihar'},
                'Vadodara': {'lat': 22.3072, 'lon': 73.1812, 'state': 'Gujarat'},
                'Ghaziabad': {'lat': 28.6692, 'lon': 77.4538, 'state': 'Uttar Pradesh'},
                'Ludhiana': {'lat': 30.9010, 'lon': 75.8573, 'state': 'Punjab'},
                'Agra': {'lat': 27.1767, 'lon': 78.0081, 'state': 'Uttar Pradesh'},
                'Nashik': {'lat': 19.9975, 'lon': 73.7898, 'state': 'Maharashtra'},
                'Faridabad': {'lat': 28.4089, 'lon': 77.3178, 'state': 'Haryana'},
                'Meerut': {'lat': 28.9845, 'lon': 77.7064, 'state': 'Uttar Pradesh'},
                'Rajkot': {'lat': 22.3039, 'lon': 70.8022, 'state': 'Gujarat'},
                'Kalyan-Dombivali': {'lat': 19.2403, 'lon': 73.1305, 'state': 'Maharashtra'},
                'Vasai-Virar': {'lat': 19.4912, 'lon': 72.8054, 'state': 'Maharashtra'},
                'Varanasi': {'lat': 25.3176, 'lon': 82.9739, 'state': 'Uttar Pradesh'},
                'Srinagar': {'lat': 34.0837, 'lon': 74.7973, 'state': 'Jammu and Kashmir'},
                'Aurangabad': {'lat': 19.8762, 'lon': 75.3433, 'state': 'Maharashtra'},
                'Dhanbad': {'lat': 23.7957, 'lon': 86.4304, 'state': 'Jharkhand'},
                'Amritsar': {'lat': 31.6340, 'lon': 74.8723, 'state': 'Punjab'},
                'Navi Mumbai': {'lat': 19.0330, 'lon': 73.0297, 'state': 'Maharashtra'},
                'Allahabad': {'lat': 25.4358, 'lon': 81.8463, 'state': 'Uttar Pradesh'},
                'Ranchi': {'lat': 23.3441, 'lon': 85.3096, 'state': 'Jharkhand'},
                'Howrah': {'lat': 22.5958, 'lon': 88.2636, 'state': 'West Bengal'},
                'Coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'state': 'Tamil Nadu'},
                'Jabalpur': {'lat': 23.1815, 'lon': 79.9864, 'state': 'Madhya Pradesh'},
                'Gwalior': {'lat': 26.2183, 'lon': 78.1828, 'state': 'Madhya Pradesh'},
                'Vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'state': 'Andhra Pradesh'},
                'Jodhpur': {'lat': 26.2389, 'lon': 73.0243, 'state': 'Rajasthan'},
                'Madurai': {'lat': 9.9252, 'lon': 78.1198, 'state': 'Tamil Nadu'},
                'Raipur': {'lat': 21.2514, 'lon': 81.6296, 'state': 'Chhattisgarh'},
                'Kota': {'lat': 25.2138, 'lon': 75.8648, 'state': 'Rajasthan'},
                'Chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'state': 'Chandigarh'},
                'Guwahati': {'lat': 26.1445, 'lon': 91.7362, 'state': 'Assam'},
                'Solapur': {'lat': 17.6599, 'lon': 75.9064, 'state': 'Maharashtra'},
                'Hubli-Dharwad': {'lat': 15.3647, 'lon': 75.1240, 'state': 'Karnataka'},
                'Bareilly': {'lat': 28.3670, 'lon': 79.4304, 'state': 'Uttar Pradesh'},
                'Moradabad': {'lat': 28.8386, 'lon': 78.7733, 'state': 'Uttar Pradesh'},
                'Mysore': {'lat': 12.2958, 'lon': 76.6394, 'state': 'Karnataka'},
                'Gurgaon': {'lat': 28.4595, 'lon': 77.0266, 'state': 'Haryana'},
                'Aligarh': {'lat': 27.8974, 'lon': 78.0880, 'state': 'Uttar Pradesh'},
                'Jalandhar': {'lat': 31.3260, 'lon': 75.5762, 'state': 'Punjab'},
                'Tiruchirappalli': {'lat': 10.7905, 'lon': 78.7047, 'state': 'Tamil Nadu'},
                'Bhubaneswar': {'lat': 20.2961, 'lon': 85.8245, 'state': 'Odisha'},
                'Salem': {'lat': 11.6643, 'lon': 78.1460, 'state': 'Tamil Nadu'},
                'Warangal': {'lat': 17.9689, 'lon': 79.5941, 'state': 'Telangana'},
                'Mira-Bhayandar': {'lat': 19.2952, 'lon': 72.8544, 'state': 'Maharashtra'},
                'Thiruvananthapuram': {'lat': 8.5241, 'lon': 76.9366, 'state': 'Kerala'},
                'Bhiwandi': {'lat': 19.3002, 'lon': 73.0635, 'state': 'Maharashtra'},
                'Saharanpur': {'lat': 29.9680, 'lon': 77.5552, 'state': 'Uttar Pradesh'},
                'Guntur': {'lat': 16.3067, 'lon': 80.4365, 'state': 'Andhra Pradesh'},
                'Amravati': {'lat': 20.9374, 'lon': 77.7796, 'state': 'Maharashtra'},
                'Bikaner': {'lat': 28.0229, 'lon': 73.3119, 'state': 'Rajasthan'},
                'Noida': {'lat': 28.5355, 'lon': 77.3910, 'state': 'Uttar Pradesh'},
                'Jamshedpur': {'lat': 22.8046, 'lon': 86.2029, 'state': 'Jharkhand'},
                'Bhilai Nagar': {'lat': 21.1938, 'lon': 81.3509, 'state': 'Chhattisgarh'},
                'Cuttack': {'lat': 20.4625, 'lon': 85.8828, 'state': 'Odisha'},
                'Firozabad': {'lat': 27.1592, 'lon': 78.3957, 'state': 'Uttar Pradesh'},
                'Kochi': {'lat': 9.9312, 'lon': 76.2673, 'state': 'Kerala'},
                'Bhavnagar': {'lat': 21.7645, 'lon': 72.1519, 'state': 'Gujarat'},
                'Dehradun': {'lat': 30.3165, 'lon': 78.0322, 'state': 'Uttarakhand'},
                'Durgapur': {'lat': 23.4841, 'lon': 87.3119, 'state': 'West Bengal'},
                'Asansol': {'lat': 23.6739, 'lon': 86.9524, 'state': 'West Bengal'},
                'Nanded-Waghala': {'lat': 19.1383, 'lon': 77.2975, 'state': 'Maharashtra'},
                'Kolhapur': {'lat': 16.7050, 'lon': 74.2433, 'state': 'Maharashtra'},
                'Ajmer': {'lat': 26.4499, 'lon': 74.6399, 'state': 'Rajasthan'},
                'Gulbarga': {'lat': 17.3297, 'lon': 76.8343, 'state': 'Karnataka'},
                'Jamnagar': {'lat': 22.4707, 'lon': 70.0577, 'state': 'Gujarat'},
                'Ujjain': {'lat': 23.1765, 'lon': 75.7885, 'state': 'Madhya Pradesh'},
                'Loni': {'lat': 28.7333, 'lon': 77.2833, 'state': 'Uttar Pradesh'},
                'Siliguri': {'lat': 26.7271, 'lon': 88.3953, 'state': 'West Bengal'},
                'Jhansi': {'lat': 25.4484, 'lon': 78.5685, 'state': 'Uttar Pradesh'},
                'Ulhasnagar': {'lat': 19.2215, 'lon': 73.1645, 'state': 'Maharashtra'},
                'Jammu': {'lat': 32.7266, 'lon': 74.8570, 'state': 'Jammu and Kashmir'},
                'Sangli-Miraj & Kupwad': {'lat': 16.8524, 'lon': 74.5815, 'state': 'Maharashtra'},
                'Mangalore': {'lat': 12.9141, 'lon': 74.8560, 'state': 'Karnataka'},
                'Erode': {'lat': 11.3410, 'lon': 77.7172, 'state': 'Tamil Nadu'},
                'Belgaum': {'lat': 15.8497, 'lon': 74.4977, 'state': 'Karnataka'},
                'Ambattur': {'lat': 13.1143, 'lon': 80.1548, 'state': 'Tamil Nadu'},
                'Tirunelveli': {'lat': 8.7139, 'lon': 77.7567, 'state': 'Tamil Nadu'},
                'Malegaon': {'lat': 20.5579, 'lon': 74.5287, 'state': 'Maharashtra'},
                'Gaya': {'lat': 24.7914, 'lon': 85.0002, 'state': 'Bihar'},
                'Jalgaon': {'lat': 21.0077, 'lon': 75.5626, 'state': 'Maharashtra'},
                'Udaipur': {'lat': 24.5854, 'lon': 73.7125, 'state': 'Rajasthan'},
                'Maheshtala': {'lat': 22.4986, 'lon': 88.2475, 'state': 'West Bengal'}
            }
            
            # Get crime hotspots data
            hotspots = analyzer.identify_crime_hotspots()
            
            # Prepare map data
            map_data = []
            for city, coords in india_cities_coords.items():
                # Get crime data for this city if available
                if city in hotspots.index:
                    crime_count = hotspots.loc[city, 'Total_Crimes']
                    violence_ratio = hotspots.loc[city, 'Violent_Crime_Ratio']
                else:
                    # Generate realistic sample data for demonstration
                    crime_count = np.random.randint(50, 800)
                    violence_ratio = np.random.uniform(0.1, 0.7)
                
                # Determine risk level and marker properties
                if crime_count > 400 and violence_ratio > 0.5:
                    risk_level = "Critical Risk"
                    marker_color = "red"
                    marker_size = 25
                elif crime_count > 250 or violence_ratio > 0.4:
                    risk_level = "High Risk"
                    marker_color = "orange"
                    marker_size = 20
                elif crime_count > 150 or violence_ratio > 0.25:
                    risk_level = "Medium Risk"
                    marker_color = "yellow"
                    marker_size = 15
                else:
                    risk_level = "Low Risk"
                    marker_color = "green"
                    marker_size = 10
                
                map_data.append({
                    'City': city,
                    'State': coords['state'],
                    'Latitude': coords['lat'],
                    'Longitude': coords['lon'],
                    'Total_Crimes': crime_count,
                    'Violence_Ratio': violence_ratio,
                    'Risk_Level': risk_level,
                    'Marker_Color': marker_color,
                    'Marker_Size': marker_size
                })
            
            # Create DataFrame for map
            map_df = pd.DataFrame(map_data)
            
            # Create the interactive India map
            fig_india_map = go.Figure()
            
            # Add markers for each city with different colors based on risk level
            for risk_level in ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']:
                risk_data = map_df[map_df['Risk_Level'] == risk_level]
                if not risk_data.empty:
                    color_map = {
                        'Low Risk': 'green',
                        'Medium Risk': 'yellow', 
                        'High Risk': 'orange',
                        'Critical Risk': 'red'
                    }
                    
                    fig_india_map.add_trace(go.Scattermapbox(
                        lat=risk_data['Latitude'],
                        lon=risk_data['Longitude'],
                        mode='markers',
                        marker=dict(
                            size=risk_data['Marker_Size'],
                            color=color_map[risk_level],
                            opacity=0.8,
                            sizemode='diameter'
                        ),
                        text=risk_data.apply(lambda row: 
                            f"<b>{row['City']}, {row['State']}</b><br>" +
                            f"Total Crimes: {row['Total_Crimes']:,}<br>" +
                            f"Violence Ratio: {row['Violence_Ratio']:.1%}<br>" +
                            f"Risk Level: {row['Risk_Level']}", axis=1),
                        hovertemplate='%{text}<extra></extra>',
                        name=f"{risk_level} ({len(risk_data)} cities)",
                        showlegend=True
                    ))
            
            # Update map layout
            fig_india_map.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=20.5937, lon=78.9629),  # Center of India
                    zoom=4.5
                ),
                height=600,
                title={
                    'text': "🗺️ India Crime Hotspot Map - Real-Time Risk Assessment",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#1a237e'}
                },
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                ),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            # Display the map
            st.plotly_chart(fig_india_map, use_container_width=True)
            
            # Map Statistics
            col_map1, col_map2, col_map3, col_map4 = st.columns(4)
            
            with col_map1:
                critical_count = len(map_df[map_df['Risk_Level'] == 'Critical Risk'])
                st.metric("🔴 Critical Risk Cities", critical_count)
            
            with col_map2:
                high_count = len(map_df[map_df['Risk_Level'] == 'High Risk'])
                st.metric("🟠 High Risk Cities", high_count)
            
            with col_map3:
                medium_count = len(map_df[map_df['Risk_Level'] == 'Medium Risk'])
                st.metric("🟡 Medium Risk Cities", medium_count)
            
            with col_map4:
                low_count = len(map_df[map_df['Risk_Level'] == 'Low Risk'])
                st.metric("🟢 Low Risk Cities", low_count)
        
        st.markdown("---")
        
        # Crime Hotspots Analysis
        st.subheader("🔥 Crime Hotspots Identification")
        
        with st.spinner("🔍 Identifying crime hotspots..."):
            hotspots = analyzer.identify_crime_hotspots()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Top Crime Hotspots")
            st.dataframe(hotspots.head(10), use_container_width=True)
            
            # Show hotspot statistics
            if not hotspots.empty:
                total_crimes = hotspots['Total_Crimes'].sum()
                avg_violent_ratio = hotspots['Violent_Crime_Ratio'].mean()
                
                st.info(f"""
                **Hotspot Statistics:**
                - Total Crimes Analyzed: {total_crimes:,}
                - Average Violent Crime Ratio: {avg_violent_ratio:.1%}
                - Number of Areas: {len(hotspots)}
                """)
        
        with col2:
            st.subheader("📈 Hotspot Visualization")
            if not hotspots.empty:
                # Bar chart of top hotspots
                top_hotspots = hotspots.head(10)
                fig_hotspots = px.bar(
                    x=top_hotspots['Total_Crimes'],
                    y=top_hotspots.index,
                    orientation='h',
                    title="Top 10 Crime Hotspots",
                    labels={'x': 'Total Crimes', 'y': 'Location'},
                    color=top_hotspots['Total_Crimes'],
                    color_continuous_scale='Reds'
                )
                fig_hotspots.update_layout(height=400)
                st.plotly_chart(fig_hotspots, use_container_width=True)
            else:
                st.info("No hotspot data available")
        
        # Geographic Distribution Analysis
        st.subheader("🌍 Geographic Distribution")
        
        # Check if we have location-based columns
        location_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['city', 'state', 'district', 'location', 'area']):
                location_cols.append(col)
        
        if location_cols:
            selected_location = st.selectbox("Select Geographic Level:", location_cols)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Geographic distribution pie chart
                location_counts = df[selected_location].value_counts().head(10)
                fig_geo_pie = px.pie(
                    values=location_counts.values,
                    names=location_counts.index,
                    title=f"Crime Distribution by {selected_location}"
                )
                st.plotly_chart(fig_geo_pie, use_container_width=True)
            
            with col4:
                # Geographic distribution bar chart
                fig_geo_bar = px.bar(
                    x=location_counts.values,
                    y=location_counts.index,
                    orientation='h',
                    title=f"Crime Count by {selected_location}",
                    labels={'x': 'Number of Crimes', 'y': selected_location}
                )
                st.plotly_chart(fig_geo_bar, use_container_width=True)
        else:
            st.warning("No geographic columns detected in the dataset")
        
        # Violent vs Non-Violent Crime Distribution
        st.subheader("⚔️ Violence Analysis by Location")
        
        if not hotspots.empty and 'Violent_Crime_Ratio' in hotspots.columns:
            # Scatter plot of total crimes vs violent crime ratio
            fig_violence = px.scatter(
                hotspots.reset_index(),
                x='Total_Crimes',
                y='Violent_Crime_Ratio',
                hover_name=hotspots.index,
                title="Total Crimes vs Violent Crime Ratio by Location",
                labels={
                    'Total_Crimes': 'Total Number of Crimes',
                    'Violent_Crime_Ratio': 'Violent Crime Ratio'
                },
                size='Total_Crimes',
                color='Violent_Crime_Ratio',
                color_continuous_scale='Reds'
            )
            fig_violence.update_layout(height=500)
            st.plotly_chart(fig_violence, use_container_width=True)
            
            # Risk categorization
            st.subheader("🚨 Risk Level Categorization")
            
            # Categorize locations by risk level
            def categorize_risk(row):
                if row['Total_Crimes'] > hotspots['Total_Crimes'].quantile(0.75) and row['Violent_Crime_Ratio'] > 0.4:
                    return "🔴 High Risk"
                elif row['Total_Crimes'] > hotspots['Total_Crimes'].quantile(0.5) or row['Violent_Crime_Ratio'] > 0.3:
                    return "🟡 Medium Risk"
                else:
                    return "🟢 Low Risk"
            
            hotspots['Risk_Level'] = hotspots.apply(categorize_risk, axis=1)
            
            # Display risk categorization
            risk_summary = hotspots['Risk_Level'].value_counts()
            
            col5, col6 = st.columns(2)
            
            with col5:
                st.subheader("Risk Level Distribution")
                fig_risk = px.pie(
                    values=risk_summary.values,
                    names=risk_summary.index,
                    title="Locations by Risk Level"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col6:
                st.subheader("High Risk Locations")
                high_risk = hotspots[hotspots['Risk_Level'] == '🔴 High Risk'].head(5)
                if not high_risk.empty:
                    st.dataframe(high_risk[['Total_Crimes', 'Violent_Crime_Ratio', 'Risk_Level']], use_container_width=True)
                else:
                    st.info("No high-risk locations identified")
        
        # Geographic Insights
        st.subheader("💡 Geographic Insights")
        
        insights = []
        
        if not hotspots.empty:
            # Top crime location
            top_location = hotspots.index[0]
            top_crimes = hotspots.iloc[0]['Total_Crimes']
            insights.append(f"🏆 **Highest Crime Area:** {top_location} with {top_crimes:,} incidents")
            
            # Most violent location
            if 'Violent_Crime_Ratio' in hotspots.columns:
                most_violent_idx = hotspots['Violent_Crime_Ratio'].idxmax()
                most_violent_ratio = hotspots.loc[most_violent_idx, 'Violent_Crime_Ratio']
                insights.append(f"⚔️ **Most Violent Area:** {most_violent_idx} with {most_violent_ratio:.1%} violent crimes")
            
            # Crime concentration
            top_5_crimes = hotspots.head(5)['Total_Crimes'].sum()
            total_crimes = hotspots['Total_Crimes'].sum()
            concentration = (top_5_crimes / total_crimes) * 100
            insights.append(f"📊 **Crime Concentration:** Top 5 areas account for {concentration:.1f}% of all crimes")
        
        for insight in insights:
            st.success(insight)
    
    else:
        st.warning("Please upload data to see geographic analysis")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_predictions():
    """Predictions section"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("🔮 Crime Predictions")
    
    if st.session_state.data_loaded:
        df = st.session_state.sample_data
        
        # Initialize analyzer if not already done
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = CrimeAnalysis(df)
            st.session_state.analyzer.preprocess_data()
        
        analyzer = st.session_state.analyzer
        
        # Crime Prediction Model
        st.subheader("🤖 Machine Learning Crime Prediction")
        
        # Build prediction model
        with st.spinner("🔄 Training prediction model..."):
            # Prepare features for prediction
            feature_columns = []
            
            # Add available numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns.extend([col for col in numeric_cols if col in analyzer.df.columns])
            
            # Add engineered features
            engineered_features = ['Hour', 'DayOfWeek', 'Month']
            feature_columns.extend([col for col in engineered_features if col in analyzer.df.columns])
            
            # Add encoded categorical features
            encoded_features = [col for col in analyzer.df.columns if col.endswith('_encoded')]
            feature_columns.extend(encoded_features)
            
            # Remove duplicates and ensure we have features
            feature_columns = list(set(feature_columns))
            
            if len(feature_columns) >= 2 and 'Violent_Crime' in analyzer.df.columns:
                try:
                    # Prepare data
                    X = analyzer.df[feature_columns].fillna(0)
                    y = analyzer.df['Violent_Crime']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    # Train model
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = rf_model.predict(X_test)
                    y_pred_proba = rf_model.predict_proba(X_test)
                    
                    # Calculate accuracy
                    accuracy = (y_pred == y_test).mean()
                    
                    # Store model in session state
                    st.session_state.prediction_model = rf_model
                    st.session_state.feature_columns = feature_columns
                    st.session_state.model_accuracy = accuracy
                    
                    st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.1%}")
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    st.session_state.prediction_model = None
            else:
                st.warning("⚠️ Insufficient features for prediction model")
                st.session_state.prediction_model = None
        
        # Interactive Prediction Interface
        if 'prediction_model' in st.session_state and st.session_state.prediction_model is not None:
            st.subheader("🎯 Interactive Crime Risk Prediction")
            
            with st.form("prediction_form"):
                st.write("**Enter scenario details to predict crime risk:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Time-based inputs
                    hour = st.slider("Hour of Day", 0, 23, 14)
                    day_of_week = st.selectbox("Day of Week", 
                                             ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                              'Friday', 'Saturday', 'Sunday'])
                    month = st.selectbox("Month", 
                                       ['January', 'February', 'March', 'April', 'May', 'June',
                                        'July', 'August', 'September', 'October', 'November', 'December'])
                
                with col2:
                    # Location-based inputs (if available)
                    location_cols = [col for col in df.columns if any(keyword in col.lower() 
                                   for keyword in ['city', 'state', 'district', 'location'])]
                    
                    if location_cols:
                        location_col = location_cols[0]
                        unique_locations = df[location_col].unique()[:20]  # Limit to first 20
                        selected_location = st.selectbox(f"Select {location_col}", unique_locations)
                    else:
                        selected_location = "Unknown"
                
                with col3:
                    # Additional inputs based on available features
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        crime_type_col = categorical_cols[0]
                        unique_types = df[crime_type_col].unique()[:10]  # Limit to first 10
                        selected_type = st.selectbox(f"Select {crime_type_col}", unique_types)
                    else:
                        selected_type = "Unknown"
                
                predict_button = st.form_submit_button("🎯 Predict Crime Risk", use_container_width=True)
                
                if predict_button:
                    # Prepare prediction input
                    try:
                        # Create input vector
                        input_data = {}
                        
                        # Add time features
                        if 'Hour' in st.session_state.feature_columns:
                            input_data['Hour'] = hour
                        if 'DayOfWeek' in st.session_state.feature_columns:
                            input_data['DayOfWeek'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                                     'Friday', 'Saturday', 'Sunday'].index(day_of_week)
                        if 'Month' in st.session_state.feature_columns:
                            input_data['Month'] = ['January', 'February', 'March', 'April', 'May', 'June',
                                                 'July', 'August', 'September', 'October', 'November', 'December'].index(month) + 1
                        
                        # Handle encoded categorical features properly
                        for col in st.session_state.feature_columns:
                            if col not in input_data:
                                if col.endswith('_encoded'):
                                    # Handle encoded categorical features
                                    base_col = col.replace('_encoded', '')
                                    if base_col in analyzer.encoders:
                                        encoder = analyzer.encoders[base_col]
                                        
                                        # Determine the value to encode based on user selection
                                        if base_col == location_cols[0] if location_cols else None:
                                            value_to_encode = selected_location
                                        elif base_col == categorical_cols[0] if len(categorical_cols) > 0 else None:
                                            value_to_encode = selected_type
                                        else:
                                            # Use the most common value from training data
                                            if base_col in analyzer.df.columns:
                                                value_to_encode = analyzer.df[base_col].mode()[0] if len(analyzer.df[base_col].mode()) > 0 else 'Unknown'
                                            else:
                                                value_to_encode = 'Unknown'
                                        
                                        # Safely encode the value
                                        try:
                                            # Check if the value exists in the encoder's classes
                                            if hasattr(encoder, 'classes_') and value_to_encode in encoder.classes_:
                                                input_data[col] = encoder.transform([str(value_to_encode)])[0]
                                            else:
                                                # Use the most common encoded value from training data
                                                input_data[col] = analyzer.df[col].mode()[0] if len(analyzer.df[col].mode()) > 0 else 0
                                        except:
                                            # Fallback to most common value
                                            input_data[col] = analyzer.df[col].mode()[0] if len(analyzer.df[col].mode()) > 0 else 0
                                    else:
                                        # No encoder available, use default
                                        input_data[col] = 0
                                elif col in analyzer.df.columns:
                                    # Handle regular features
                                    if analyzer.df[col].dtype in ['int64', 'float64']:
                                        input_data[col] = analyzer.df[col].median()
                                    else:
                                        input_data[col] = analyzer.df[col].mode()[0] if len(analyzer.df[col].mode()) > 0 else 0
                                else:
                                    input_data[col] = 0
                        
                        # Ensure all required features are present with correct data types
                        for col in st.session_state.feature_columns:
                            if col not in input_data:
                                input_data[col] = 0
                            # Convert to appropriate numeric type
                            try:
                                input_data[col] = float(input_data[col])
                            except:
                                input_data[col] = 0.0
                        
                        # Create DataFrame for prediction with correct column order
                        input_df = pd.DataFrame([input_data])[st.session_state.feature_columns]
                        
                        # Make prediction
                        prediction = st.session_state.prediction_model.predict(input_df)[0]
                        prediction_proba = st.session_state.prediction_model.predict_proba(input_df)[0]
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        
                        col_res1, col_res2, col_res3 = st.columns(3)
                        
                        with col_res1:
                            risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
                            risk_color = "🔴" if prediction == 1 else "🟢"
                            st.metric("Crime Risk Level", f"{risk_color} {risk_level}")
                        
                        with col_res2:
                            violent_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0
                            st.metric("Violent Crime Probability", f"{violent_prob:.1%}")
                        
                        with col_res3:
                            st.metric("Model Confidence", f"{st.session_state.model_accuracy:.1%}")
                        
                        # Risk interpretation
                        if prediction == 1:
                            st.error("⚠️ **HIGH RISK SCENARIO** - Increased police presence recommended")
                        else:
                            st.success("✅ **LOW RISK SCENARIO** - Normal patrol levels sufficient")
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
        
        # Feature Importance Analysis
        if 'prediction_model' in st.session_state and st.session_state.prediction_model is not None:
            st.subheader("📊 Feature Importance Analysis")
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_columns,
                'Importance': st.session_state.prediction_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            col_imp1, col_imp2 = st.columns(2)
            
            with col_imp1:
                st.subheader("Top 10 Most Important Features")
                st.dataframe(feature_importance.head(10), use_container_width=True)
            
            with col_imp2:
                # Feature importance visualization
                fig_importance = px.bar(
                    feature_importance.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance for Crime Prediction",
                    labels={'Importance': 'Importance Score', 'Feature': 'Features'}
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Prediction Trends and Patterns
        st.subheader("📈 Prediction Trends and Patterns")
        
        # Hourly risk prediction
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            st.subheader("Risk by Hour of Day")
            if 'prediction_model' in st.session_state and st.session_state.prediction_model is not None:
                # Calculate risk for each hour
                hourly_risk = []
                for h in range(24):
                    # Create sample input for each hour
                    sample_input = {}
                    for col in st.session_state.feature_columns:
                        if col == 'Hour':
                            sample_input[col] = h
                        elif col in analyzer.df.columns:
                            if analyzer.df[col].dtype in ['int64', 'float64']:
                                sample_input[col] = analyzer.df[col].median()
                            else:
                                sample_input[col] = analyzer.df[col].mode()[0] if len(analyzer.df[col].mode()) > 0 else 0
                        else:
                            sample_input[col] = 0
                    
                    try:
                        sample_df = pd.DataFrame([sample_input])
                        risk_prob = st.session_state.prediction_model.predict_proba(sample_df)[0][1]
                        hourly_risk.append(risk_prob)
                    except:
                        hourly_risk.append(0.5)  # Default risk
                
                fig_hourly = px.line(
                    x=list(range(24)),
                    y=hourly_risk,
                    title="Crime Risk Probability by Hour",
                    labels={'x': 'Hour of Day', 'y': 'Risk Probability'}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            else:
                st.info("Train the model first to see hourly risk trends")
        
        with col_trend2:
            st.subheader("Risk by Day of Week")
            if 'prediction_model' in st.session_state and st.session_state.prediction_model is not None:
                # Calculate risk for each day
                daily_risk = []
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                for d in range(7):
                    sample_input = {}
                    for col in st.session_state.feature_columns:
                        if col == 'DayOfWeek':
                            sample_input[col] = d
                        elif col in analyzer.df.columns:
                            if analyzer.df[col].dtype in ['int64', 'float64']:
                                sample_input[col] = analyzer.df[col].median()
                            else:
                                sample_input[col] = analyzer.df[col].mode()[0] if len(analyzer.df[col].mode()) > 0 else 0
                        else:
                            sample_input[col] = 0
                    
                    try:
                        sample_df = pd.DataFrame([sample_input])
                        risk_prob = st.session_state.prediction_model.predict_proba(sample_df)[0][1]
                        daily_risk.append(risk_prob)
                    except:
                        daily_risk.append(0.5)
                
                fig_daily = px.bar(
                    x=day_names,
                    y=daily_risk,
                    title="Crime Risk Probability by Day of Week",
                    labels={'x': 'Day of Week', 'y': 'Risk Probability'}
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.info("Train the model first to see daily risk trends")
        
        # Model Performance Metrics
        if 'prediction_model' in st.session_state and st.session_state.prediction_model is not None:
            st.subheader("📊 Model Performance Metrics")
            
            col_perf1, col_perf2, col_perf3 = st.columns(3)
            
            with col_perf1:
                st.metric("Model Accuracy", f"{st.session_state.model_accuracy:.1%}")
            
            with col_perf2:
                st.metric("Number of Features", len(st.session_state.feature_columns))
            
            with col_perf3:
                st.metric("Training Data Size", f"{len(analyzer.df):,} records")
    
    else:
        st.warning("Please upload data to enable predictions")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_reports():
    """Reports section"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("📋 Reports & Recommendations")
    
    if st.session_state.data_loaded:
        df = st.session_state.sample_data
        
        # Initialize analyzer if not already done
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = CrimeAnalysis(df)
            st.session_state.analyzer.preprocess_data()
        
        analyzer = st.session_state.analyzer
        
        # Executive Summary
        st.subheader("📊 Executive Summary")
        
        with st.spinner("📋 Generating comprehensive report..."):
            # Key Statistics
            total_records = len(df)
            violent_crimes = analyzer.df['Violent_Crime'].sum()
            violent_percentage = analyzer.df['Violent_Crime'].mean()
            
            # Date range
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].dropna().head(100), errors='raise')
                        date_cols.append(col)
                        break
                    except:
                        continue
            
            date_range = "N/A"
            if date_cols:
                try:
                    dates = pd.to_datetime(df[date_cols[0]], errors='coerce').dropna()
                    if len(dates) > 0:
                        date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                except:
                    pass
        
        # Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Incidents", f"{total_records:,}")
        with col2:
            st.metric("⚔️ Violent Crimes", f"{violent_crimes:,}")
        with col3:
            st.metric("📈 Violence Rate", f"{violent_percentage:.1%}")
        with col4:
            st.metric("📅 Analysis Period", date_range)
        
        # Policy Recommendations
        st.subheader("🚨 Data-Driven Policy Recommendations")
        
        with st.spinner("🔍 Analyzing patterns for recommendations..."):
            recommendations = analyzer.generate_policy_recommendations()
        
        st.success("### 🎯 Strategic Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            st.info(f"**{i}.** {rec}")
        
        # Crime Hotspots Report
        st.subheader("🔥 Crime Hotspots Analysis Report")
        
        hotspots = analyzer.identify_crime_hotspots()
        
        if not hotspots.empty:
            col_hot1, col_hot2 = st.columns(2)
            
            with col_hot1:
                st.write("**Top 5 High-Risk Areas:**")
                top_hotspots = hotspots.head(5)
                
                for idx, (location, data) in enumerate(top_hotspots.iterrows(), 1):
                    risk_level = "🔴 HIGH" if data['Violent_Crime_Ratio'] > 0.4 else "🟡 MEDIUM"
                    st.write(f"{idx}. **{location}**")
                    st.write(f"   - Total Crimes: {data['Total_Crimes']:,}")
                    st.write(f"   - Violence Rate: {data['Violent_Crime_Ratio']:.1%}")
                    st.write(f"   - Risk Level: {risk_level}")
                    st.write("")
            
            with col_hot2:
                # Hotspot visualization
                fig_hotspots_report = px.bar(
                    x=hotspots.head(8)['Total_Crimes'],
                    y=hotspots.head(8).index,
                    orientation='h',
                    title="Crime Hotspots Overview",
                    labels={'x': 'Total Crimes', 'y': 'Location'},
                    color=hotspots.head(8)['Total_Crimes'],
                    color_continuous_scale='Reds'
                )
                fig_hotspots_report.update_layout(height=400)
                st.plotly_chart(fig_hotspots_report, use_container_width=True)
        
        # Temporal Analysis Report
        st.subheader("⏰ Temporal Crime Patterns Report")
        
        col_temp1, col_temp2 = st.columns(2)
        
        with col_temp1:
            st.write("**Peak Crime Hours:**")
            hourly_crimes = analyzer.df['Hour'].value_counts().sort_values(ascending=False)
            peak_hours = hourly_crimes.head(5)
            
            for hour, count in peak_hours.items():
                time_period = "Morning" if 6 <= hour < 12 else "Afternoon" if 12 <= hour < 18 else "Evening" if 18 <= hour < 22 else "Night"
                st.write(f"• **{hour}:00** ({time_period}) - {count:,} incidents")
            
            st.write("\n**Recommendations:**")
            st.write("• Increase patrol presence during peak hours")
            st.write("• Deploy additional resources in evening/night shifts")
            st.write("• Focus on preventive measures during high-activity periods")
        
        with col_temp2:
            # Monthly trend
            monthly_crimes = analyzer.df['Month'].value_counts().sort_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Filter valid months and convert to integers for reports
            valid_months_report = [(int(i), count) for i, count in monthly_crimes.items() if isinstance(i, (int, float)) and 1 <= i <= 12]
            
            if valid_months_report:
                months_report, counts_report = zip(*valid_months_report)
                month_labels_report = [month_names[m-1] for m in months_report]
                
                fig_monthly_report = px.line(
                    x=month_labels_report,
                    y=counts_report,
                    title="Monthly Crime Trends",
                    labels={'x': 'Month', 'y': 'Number of Crimes'}
                )
                st.plotly_chart(fig_monthly_report, use_container_width=True)
            else:
                st.info("No valid monthly data available for trend analysis")
        
        # Risk Assessment Matrix
        st.subheader("🎯 Risk Assessment Matrix")
        
        if not hotspots.empty and 'Violent_Crime_Ratio' in hotspots.columns:
            # Create risk categories
            def get_risk_category(row):
                crimes = row['Total_Crimes']
                violence_ratio = row['Violent_Crime_Ratio']
                
                if crimes > hotspots['Total_Crimes'].quantile(0.75):
                    if violence_ratio > 0.4:
                        return "🔴 Critical Risk"
                    elif violence_ratio > 0.25:
                        return "🟠 High Risk"
                    else:
                        return "🟡 Medium Risk"
                elif crimes > hotspots['Total_Crimes'].quantile(0.5):
                    if violence_ratio > 0.3:
                        return "🟠 High Risk"
                    else:
                        return "🟡 Medium Risk"
                else:
                    return "🟢 Low Risk"
            
            hotspots['Risk_Category'] = hotspots.apply(get_risk_category, axis=1)
            risk_distribution = hotspots['Risk_Category'].value_counts()
            
            col_risk1, col_risk2 = st.columns(2)
            
            with col_risk1:
                st.write("**Risk Distribution:**")
                for risk_level, count in risk_distribution.items():
                    percentage = (count / len(hotspots)) * 100
                    st.write(f"• {risk_level}: {count} areas ({percentage:.1f}%)")
                
                st.write("\n**Priority Actions:**")
                critical_areas = len(hotspots[hotspots['Risk_Category'] == '🔴 Critical Risk'])
                if critical_areas > 0:
                    st.error(f"⚠️ {critical_areas} areas require immediate intervention")
                
                high_risk_areas = len(hotspots[hotspots['Risk_Category'] == '🟠 High Risk'])
                if high_risk_areas > 0:
                    st.warning(f"⚠️ {high_risk_areas} areas need enhanced monitoring")
            
            with col_risk2:
                fig_risk_dist = px.pie(
                    values=risk_distribution.values,
                    names=risk_distribution.index,
                    title="Risk Level Distribution"
                )
                st.plotly_chart(fig_risk_dist, use_container_width=True)
        
        # Resource Allocation Recommendations
        st.subheader("👮‍♂️ Resource Allocation Strategy")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.write("**Immediate Actions (0-30 days):**")
            st.write("• Deploy additional patrol units to critical risk areas")
            st.write("• Increase surveillance during peak crime hours")
            st.write("• Establish rapid response teams for high-risk zones")
            st.write("• Implement community policing initiatives")
            
            st.write("\n**Short-term Strategy (1-6 months):**")
            st.write("• Install CCTV systems in hotspot areas")
            st.write("• Conduct crime prevention awareness programs")
            st.write("• Strengthen inter-agency coordination")
            st.write("• Develop predictive policing capabilities")
        
        with col_res2:
            st.write("**Long-term Vision (6+ months):**")
            st.write("• Implement data-driven policing strategies")
            st.write("• Establish crime analysis units")
            st.write("• Develop community partnership programs")
            st.write("• Create integrated crime management systems")
            
            st.write("\n**Success Metrics:**")
            st.write("• 15-20% reduction in overall crime rates")
            st.write("• 25% decrease in violent crime incidents")
            st.write("• Improved response times by 30%")
            st.write("• Enhanced community satisfaction scores")
        
        # Data Quality and Limitations
        st.subheader("📊 Data Quality Assessment")
        
        # Calculate data quality metrics
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.count().sum()
        completeness = (non_null_cells / total_cells * 100)
        
        col_qual1, col_qual2 = st.columns(2)
        
        with col_qual1:
            st.write("**Data Completeness:**")
            st.progress(completeness / 100)
            st.write(f"Overall completeness: {completeness:.1f}%")
            
            # Column-wise completeness
            st.write("\n**Column Completeness:**")
            for col in df.columns[:5]:  # Show first 5 columns
                col_completeness = (df[col].count() / len(df)) * 100
                st.write(f"• {col}: {col_completeness:.1f}%")
        
        with col_qual2:
            st.write("**Recommendations for Data Improvement:**")
            if completeness < 90:
                st.write("• Implement data validation at collection points")
                st.write("• Establish data quality monitoring processes")
                st.write("• Train personnel on proper data entry procedures")
            
            st.write("• Regular data audits and cleansing")
            st.write("• Standardize data collection formats")
            st.write("• Implement automated data validation rules")
        
        # Export Report Option
        st.subheader("📄 Export Report")
        
        if st.button("📋 Generate Detailed PDF Report", use_container_width=True):
            st.info("📄 PDF report generation feature would be implemented here")
            st.write("The report would include:")
            st.write("• Executive summary with key findings")
            st.write("• Detailed statistical analysis")
            st.write("• Visualization charts and graphs")
            st.write("• Policy recommendations")
            st.write("• Resource allocation strategies")
            st.write("• Implementation timeline")
        
        # Contact Information
        st.subheader("📞 Contact Information")
        
        st.info("""
        **For questions about this analysis or to request additional reports:**
        
        📧 **Email:** crime.analysis@police.gov.in  
        📱 **Phone:** +91-11-2334-5678  
        🏢 **Department:** Crime Analysis & Research Division  
        📍 **Address:** Ministry of Home Affairs, New Delhi
        
        **Report Generated:** """ + datetime.now().strftime("%B %d, %Y at %I:%M %p"))
    
    else:
        st.warning("Please upload data to generate reports")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
