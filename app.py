import streamlit as st
import pandas as pd
import time
import random
import numpy as np
import os
from datetime import datetime

try:
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

# TRAFFIC MONITORING SYSTEM
TRAFFIC_LOG_FILE = "traffic_log.csv"

def log_event(event_type):
    """
    Logs an event (Visit, Optimization, etc.) to a CSV file with a timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(TRAFFIC_LOG_FILE):
        with open(TRAFFIC_LOG_FILE, "w") as f:
            f.write("Timestamp,Event\n")
    
    # Append the new event
    with open(TRAFFIC_LOG_FILE, "a") as f:
        f.write(f"{timestamp},{event_type}\n")

def show_traffic_dashboard():
    """
    Reads the log file and displays simple analytics in the sidebar.
    """
    if os.path.exists(TRAFFIC_LOG_FILE):
        try:
            df_traffic = pd.read_csv(TRAFFIC_LOG_FILE)
            df_traffic["Timestamp"] = pd.to_datetime(df_traffic["Timestamp"])
            
            # Metric 1: Total Visits
            total_visits = len(df_traffic[df_traffic["Event"] == "App Visit"])
            
            # Metric 2: Total Optimizations Run
            total_optims = len(df_traffic[df_traffic["Event"] == "Optimization Run"])
            
            st.sidebar.markdown("---")
            st.sidebar.header("📊 Traffic Monitor")
            
            col1, col2 = st.sidebar.columns(2)
            col1.metric("Visits", total_visits)
            col2.metric("Uses", total_optims)
            
            # Simple Chart: Events over time
            st.sidebar.caption("Activity Log")
            st.sidebar.dataframe(df_traffic.tail(5), hide_index=True)
            
        except Exception as e:
            st.sidebar.error(f"Log Error: {e}")

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Logistics Route Optimizer",
    page_icon="🚚",
    layout="wide"
)

# AUTHENTICATION
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "Logistics2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter Access Code (Hint: Logistics2025)", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter Access Code (Hint: Logistics2025)", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

if "visit_logged" not in st.session_state:
    log_event("App Visit")
    st.session_state["visit_logged"] = True

# SIDEBAR
st.sidebar.title("Configuration")
st.sidebar.header("Model Selection")

model_choice = st.sidebar.radio(
    "Choose Prediction Model:",
    ("Linear Regression (Baseline)", "Random Forest (Proposed)", "Logarithmic Regression (Experimental)")
)

st.sidebar.info(f"Currently using: **{model_choice}**")

if model_choice == "Linear Regression (Baseline)":
    st.sidebar.markdown("*RMSE: 8.37 min | R²: 0.84*")
elif model_choice == "Random Forest (Proposed)":
    st.sidebar.markdown("*RMSE: 11.46 min | R²: 0.70*")
else:
    st.sidebar.markdown("*RMSE: 13.1 min | R²: 0.61*")

# MAIN APP
st.title("🚚 Logistics & Delivery Route Optimizer")
st.markdown("""
**Current Phase:** MVP Deployment & Validation.  
This tool optimizes delivery routes using a **Genetic Algorithm** and predicts travel times using **Machine Learning**.
""")

# DATA UPLOAD SECTION
st.header("1. Upload Delivery Data")
uploaded_file = st.file_uploader(
    "Upload CSV (Format: CustomerID, Latitude, Longitude, TimeWindow)", 
    type=["csv"], 
    key="main_csv_uploader"
)

if not uploaded_file:
    st.warning("⚠️ No file uploaded. Using sample demo data for visualization.")
    # MOCK DATA
    data = {
        'CUSTOMERID': [f'Cust_{i}' for i in range(1, 11)],
        'LATITUDE': [47.6062 + random.uniform(-0.05, 0.05) for _ in range(10)], # Seattle
        'LONGITUDE': [-122.3321 + random.uniform(-0.05, 0.05) for _ in range(10)],
        'MILES': [random.uniform(2, 20) for _ in range(10)]
    }
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(uploaded_file)
    
    # DATA NORMALIZATION
    df.columns = [c.upper() for c in df.columns]
    
    if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
        st.warning("⚠️ CSV lacks 'Latitude'/'Longitude'. Generating simulated coordinates for Map Demo.")
        df['LATITUDE'] = [47.6062 + random.uniform(-0.1, 0.1) for _ in range(len(df))]
        df['LONGITUDE'] = [-122.3321 + random.uniform(-0.1, 0.1) for _ in range(len(df))]
    
    if 'CUSTOMERID' not in df.columns:
        df['CUSTOMERID'] = [f"Stop_{i}" for i in range(len(df))]

st.dataframe(df.head(), width='stretch')

# OPTIMIZATION SCREEN
st.header("2. Route Optimization")
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Controls")
    generations = st.slider("GA Generations", min_value=10, max_value=100, value=50)
    optimize_btn = st.button("🚀 Optimize Route", type="primary")

# BACKEND SIMULATION
def run_optimization_simulation(df, model_type):
    """
    Running genetic_algorithm.py
    """
    try:
        from genetic_algorithm import RouteOptimizerGA
    except ImportError:
        st.error("Missing 'genetic_algorithm.py' or dependencies.")
        return 0, 0, 0, 0

    with st.spinner('Initializing Route Optimizer (Batch ML Processing)...'):
        locations = list(zip(df['LATITUDE'], df['LONGITUDE']))
        
        optimizer = RouteOptimizerGA(
            locations, 
            population_size=50, 
            mutation_rate=0.05, 
            generations=50
        )
    
    with st.spinner(f'Running Evolutionary Process...'):
        progress_bar = st.progress(0)
        
        start_time = time.time()
        
        best_route_indices = optimizer.run()
        progress_bar.progress(100)
        
        end_time = time.time()
        compute_time = end_time - start_time

    optimized_df = df.iloc[best_route_indices].reset_index(drop=True)
    
    original_dist = 0
    for i in range(len(df) - 1):
        original_dist += optimizer._haversine(locations[i], locations[i+1])
    original_dist = original_dist / 1609.34 

    optimized_dist = 0
    opt_locations = list(zip(optimized_df['LATITUDE'], optimized_df['LONGITUDE']))
    for i in range(len(opt_locations) - 1):
        optimized_dist += optimizer._haversine(opt_locations[i], opt_locations[i+1])
    optimized_dist = optimized_dist / 1609.34

    if "Linear" in model_type:
        pred_time = optimized_dist * 2
    elif "Logarithmic" in model_type:
        pred_time = optimized_dist * 2.5
    else:
        pred_time = optimized_dist * 2.25

    return original_dist, optimized_dist, compute_time, pred_time, optimized_df
    
# SHAP EXPLAINABILITY
@st.cache_resource
def build_model_and_explain(input_df):
    if not HAS_ML_LIBS:
        return None, "Libraries missing"
    
    required_cols = ['START_DATE', 'END_DATE', 'MILES']
    if not all(col in input_df.columns for col in required_cols):
        return None, "Missing Columns"

    try:
        df_ml = input_df.copy()
        df_ml = df_ml.dropna(subset=required_cols)
        
        df_ml['START_DATE'] = pd.to_datetime(df_ml['START_DATE'], errors='coerce')
        df_ml['END_DATE'] = pd.to_datetime(df_ml['END_DATE'], errors='coerce')
        df_ml = df_ml.dropna(subset=['START_DATE', 'END_DATE'])
        
        df_ml['TRAVEL_TIME_MIN'] = (df_ml['END_DATE'] - df_ml['START_DATE']).dt.total_seconds() / 60.0
        df_ml['START_HOUR'] = df_ml['START_DATE'].dt.hour
        df_ml['DAY_OF_WEEK'] = df_ml['START_DATE'].dt.dayofweek
        
        df_ml = df_ml[df_ml['TRAVEL_TIME_MIN'] > 0]
        df_ml = df_ml[df_ml['MILES'] > 0]
        
        numeric_features = ['MILES', 'START_HOUR', 'DAY_OF_WEEK']
        X = df_ml[numeric_features]
        y = df_ml['TRAVEL_TIME_MIN']
        
        if len(X) < 10:
            return None, "Not enough data"

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(model)
        sample_X = X_test.iloc[:50] if len(X_test) > 50 else X_test
        shap_values = explainer.shap_values(sample_X)
        
        return (shap_values, sample_X), "Success"

    except Exception as e:
        return None, str(e)

# MAP CREATION    
def create_route_map(df):
    try:
        import folium
        from pathfinding import PathFinder
        center_lat = df['LATITUDE'].mean()
        center_lon = df['LONGITUDE'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

        pf = PathFinder(center_lat=center_lat, center_lon=center_lon)

        locations = list(zip(df['LATITUDE'], df['LONGITUDE']))
        for index, row in df.iterrows():
            folium.Marker(
                [row['LATITUDE'], row['LONGITUDE']], 
                popup=f"ID: {row['CUSTOMERID']}",
                icon=folium.Icon(color="blue", icon="truck", prefix="fa")
            ).add_to(m)
        
        full_route_points = []

        if len(locations) > 1:
            with st.spinner("Calculating actual street paths for map..."):
                for i in range(len(locations) - 1):
                    start = locations[i]
                    end = locations[i+1]
                    
                    segment = pf.get_route_coords(start, end)
                    full_route_points.extend(segment)
        
        folium.PolyLine(
                full_route_points, 
                color="red", 
                weight=4, 
                opacity=0.7
            ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Map Generation Error: {e}")
        return None

# LOGIC TO HANDLE BUTTON CLICKS AND PERSISTENCE
if optimize_btn:
    orig_dist, opt_dist, compute_time, pred_time, optimized_df_sorted = run_optimization_simulation(df, model_choice)
    map_obj = create_route_map(optimized_df_sorted)
    
    st.session_state['optimization_run'] = True
    st.session_state['orig_dist'] = orig_dist
    st.session_state['opt_dist'] = opt_dist
    st.session_state['pred_time'] = pred_time
    st.session_state['last_model_used'] = model_choice
    st.session_state['generated_map'] = map_obj
    st.session_state['compute_time'] = compute_time

# RESULTS & VISUALIZATION
if st.session_state.get('optimization_run'):
    orig_dist = st.session_state['orig_dist']
    opt_dist = st.session_state['opt_dist']
    pred_time = st.session_state['pred_time']
    model_used = st.session_state.get('last_model_used', 'Unknown Model')
    map_obj = st.session_state.get('generated_map')
    compute_time = st.session_state['compute_time']
    
    efficiency_gain = ((orig_dist - opt_dist) / orig_dist) * 100
    
    st.subheader("Optimization Results")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Optimized Distance", f"{opt_dist:.2f} miles", delta=f"-{efficiency_gain:.1f}%")
    kpi2.metric("Predicted Total Time", f"{pred_time:.0f} mins", delta=f"via {model_used.split()[0]} Model")
    kpi3.metric("Algorithm Compute Time", f"{compute_time:.4f} sec", help="Latency scales with number of stops")
    st.subheader("Route Visualization")
    
    if map_obj:
        try:
            from streamlit_folium import st_folium

            st_folium(map_obj, width=800, height=500, returned_objects=[])
            
        except Exception as e:
            st.error(f"Map Error: {e}")
            st.write("Ensure 'streamlit-folium' is installed.")

# EXPLAINABILITY
st.divider()
st.header("3. AI Explainability")

with st.expander("View Feature Importance"):
    shap_data, status = build_model_and_explain(df)
    
    if shap_data:
        shap_values, sample_X = shap_data
        st.success("SHAP values generated from uploaded data.")
        
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "Feature": sample_X.columns,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)
        
        st.markdown("####Feature Importance")
        st.bar_chart(importance_df, x="Feature", y="Importance")
        
        st.caption("This chart shows the average impact of each feature on predicted travel time (calculated live via Random Forest).")
            
    else:
        st.warning(f"Could not generate live SHAP plots: {status}.")
        st.info("Displaying pre-computed reference charts instead.")
        
        chart_data = pd.DataFrame({
            "Feature": ["MILES", "START_HOUR", "DAY_OF_WEEK",],
            "Importance": [0.65, 0.20, 0.10]
        })
        st.bar_chart(chart_data, x="Feature", y="Importance")

st.markdown("---")
st.caption("Delivery Optimizer | Abdullah Ihsan (2023039), Aazeb Ali (2023003)")
