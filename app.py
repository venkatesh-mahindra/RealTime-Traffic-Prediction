import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import pickle
from datetime import datetime, timedelta
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Set page configuration
st.set_page_config(
    page_title="Real-time Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Define model classes (same as in the notebook)
class TimeSeriesNN(nn.Module):
    def __init__(self, input_dim):
        super(TimeSeriesNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=1):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.linear(x)
        return x

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try to load the actual data file
        df = pd.read_csv('traffic.csv')
        print("Successfully loaded traffic.csv")
    except FileNotFoundError:
        # If the file doesn't exist, create a synthetic dataset with the same structure
        print("Creating synthetic dataset for demonstration")
        
        # Create a synthetic dataset with the specified columns
        np.random.seed(42)
        
        # Generate dates for the past 30 days with hourly readings
        dates = []
        for day in range(30):
            for hour in range(24):
                dates.append(datetime(2023, 1, 1) + timedelta(days=day, hours=hour))
        
        # Generate data for 10 junctions
        junctions = list(range(1, 11))
        
        # Create the dataframe
        data = []
        id_counter = 1
        
        for dt in dates:
            for junction in junctions:
                # Base traffic pattern with time-of-day variation
                hour = dt.hour
                day_of_week = dt.weekday()
                
                # Create time-based patterns
                if 7 <= hour <= 9:  # Morning rush hour
                    base_vehicles = 100 + np.random.normal(0, 10)
                elif 16 <= hour <= 18:  # Evening rush hour
                    base_vehicles = 120 + np.random.normal(0, 15)
                elif 0 <= hour <= 5:  # Night time
                    base_vehicles = 20 + np.random.normal(0, 5)
                else:  # Regular hours
                    base_vehicles = 60 + np.random.normal(0, 8)
                
                # Add day-of-week effect
                if day_of_week >= 5:  # Weekend
                    base_vehicles *= 0.7  # Less traffic on weekends
                
                # Add junction-specific patterns
                junction_factor = 0.8 + (junction / 10)
                vehicles = int(base_vehicles * junction_factor)
                vehicles = max(1, vehicles)  # Ensure positive count
                
                data.append({
                    'DateTime': dt,
                    'Junction': junction,
                    'Vehicles': vehicles,
                    'ID': id_counter
                })
                id_counter += 1
        
        df = pd.DataFrame(data)
    
    # Ensure DateTime is in datetime format
    if df['DateTime'].dtype == 'object':
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Create time-based features
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Month'] = df['DateTime'].dt.month
    df['DayOfMonth'] = df['DateTime'].dt.day
    df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['IsRushHour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9) | 
                        (df['Hour'] >= 16) & (df['Hour'] <= 18)).astype(int)
    
    # Create lag features
    def create_lag_features(df, group_col, target_col, lag_periods):
        df_copy = df.copy()
        df_copy = df_copy.sort_values(['Junction', 'DateTime'])
        
        for lag in lag_periods:
            df_copy[f'{target_col}_lag_{lag}'] = df_copy.groupby(group_col)[target_col].shift(lag)
        
        return df_copy
    
    lag_periods = [1, 2, 3, 6, 12, 24]
    df = create_lag_features(df, 'Junction', 'Vehicles', lag_periods)
    
    # Create rolling window features
    df = df.sort_values(['Junction', 'DateTime'])
    df['Vehicles_rolling_mean_3h'] = df.groupby('Junction')['Vehicles'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['Vehicles_rolling_mean_6h'] = df.groupby('Junction')['Vehicles'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    df['Vehicles_rolling_std_3h'] = df.groupby('Junction')['Vehicles'].transform(lambda x: x.rolling(window=3, min_periods=1).std())
    
    # Drop rows with NaN values
    df_clean = df.dropna()
    
    return df_clean

# Function to load models (in a real app, you would load saved models)
@st.cache_resource
def load_models(df):
    # For demonstration, we'll create and "train" simple models
    
    # Time Series Model
    # Prepare features
    time_series_features = [
        'Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'WeekOfYear', 'IsWeekend', 'IsRushHour',
        'Vehicles_lag_1', 'Vehicles_lag_2', 'Vehicles_lag_3', 'Vehicles_lag_6', 'Vehicles_lag_12', 'Vehicles_lag_24',
        'Vehicles_rolling_mean_3h', 'Vehicles_rolling_mean_6h', 'Vehicles_rolling_std_3h'
    ]
    
    from sklearn.preprocessing import StandardScaler
    
    X = df[time_series_features].values
    y = df['Vehicles'].values
    
    # Scale the features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Create a simple model
    ts_model = TimeSeriesNN(input_dim=len(time_series_features))
    
    # In a real app, you would load the saved model
    # ts_model.load_state_dict(torch.load('time_series_model.pth'))
    
    # GNN Model
    # Create a simple network where junctions are connected based on their ID proximity
    junctions = sorted(df['Junction'].unique())
    
    # Create edge index
    edge_index = []
    for i, junction in enumerate(junctions[:-2]):
        # Connect to next two junctions
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])  # Bidirectional
        
        edge_index.append([i, i+2])
        edge_index.append([i+2, i])  # Bidirectional
    
    # Connect the last two junctions
    if len(junctions) >= 2:
        edge_index.append([len(junctions)-2, len(junctions)-1])
        edge_index.append([len(junctions)-1, len(junctions)-2])  # Bidirectional
    
    # Connect the last junction to the first (cycle)
    if len(junctions) > 2:
        edge_index.append([len(junctions)-1, 0])
        edge_index.append([0, len(junctions)-1])  # Bidirectional
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Prepare node features
    node_features = []
    for junction in junctions:
        junction_data = df[df['Junction'] == junction]
        
        # Use the most recent data for each junction
        latest_data = junction_data.sort_values('DateTime').iloc[-1]
        
        # Create a feature vector for this junction
        features = [
            latest_data['Vehicles'],
            latest_data['Hour'],
            latest_data['DayOfWeek'],
            latest_data['IsWeekend'],
            latest_data['IsRushHour'],
            latest_data['Vehicles_rolling_mean_3h'],
            latest_data['Vehicles_rolling_std_3h']
        ]
        
        node_features.append(features)
    
    # Convert to tensor and normalize
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Normalize features
    mean = node_features.mean(dim=0, keepdim=True)
    std = node_features.std(dim=0, keepdim=True)
    std[std == 0] = 1  # Avoid division by zero
    node_features = (node_features - mean) / std
    
    # Create target values (current traffic volume)
    target = []
    for junction in junctions:
        junction_data = df[df['Junction'] == junction]
        target.append(junction_data.sort_values('DateTime')['Vehicles'].iloc[-1])
    
    target = torch.tensor(target, dtype=torch.float)
    
    # Create PyTorch Geometric data object
    gnn_data = Data(x=node_features, edge_index=edge_index, y=target)
    
    # Create a simple GNN model
    gnn_model = GNNModel(input_dim=gnn_data.num_node_features)
    
    # In a real app, you would load the saved model
    # gnn_model.load_state_dict(torch.load('gnn_model.pth'))
    
    return ts_model, gnn_model, gnn_data, scaler_X, scaler_y, time_series_features, junctions

# Function to predict future traffic
def predict_future_traffic(model, scaler_X, scaler_y, df, junction, hours_ahead=3, features=None):
    if features is None:
        features = [
            'Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'WeekOfYear', 'IsWeekend', 'IsRushHour',
            'Vehicles_lag_1', 'Vehicles_lag_2', 'Vehicles_lag_3', 'Vehicles_lag_6', 'Vehicles_lag_12', 'Vehicles_lag_24',
            'Vehicles_rolling_mean_3h', 'Vehicles_rolling_mean_6h', 'Vehicles_rolling_std_3h'
        ]
    
    # Get the most recent data for this junction
    junction_data = df[df['Junction'] == junction].sort_values('DateTime')
    
    predictions = []
    current_data = junction_data.iloc[-1:].copy()
    
    for hour in range(hours_ahead):
        # Prepare features
        X = current_data[features].values
        X_scaled = scaler_X.transform(X)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            pred_scaled = model(X_tensor).item()
            pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        
        predictions.append(pred)
        
        # Update current data for next prediction
        next_time = current_data['DateTime'].iloc[0] + timedelta(hours=1)
        new_row = current_data.copy()
        new_row['DateTime'] = next_time
        new_row['Hour'] = next_time.hour
        new_row['DayOfWeek'] = next_time.weekday()
        new_row['Month'] = next_time.month
        new_row['DayOfMonth'] = next_time.day
        new_row['WeekOfYear'] = next_time.isocalendar()[1]
        new_row['IsWeekend'] = 1 if next_time.weekday() >= 5 else 0
        new_row['IsRushHour'] = 1 if (7 <= next_time.hour <= 9 or 16 <= next_time.hour <= 18) else 0
        new_row['Vehicles'] = pred
        
        # Update lag features
        for lag in range(1, 25):
            if f'Vehicles_lag_{lag}' in new_row.columns:
                if lag == 1:
                    new_row[f'Vehicles_lag_{lag}'] = current_data['Vehicles'].values[0]
                else:
                    lag_col = f'Vehicles_lag_{lag-1}'
                    if lag_col in current_data.columns:
                        new_row[f'Vehicles_lag_{lag}'] = current_data[lag_col].values[0]
        
        # Update rolling features (simplified)
        new_row['Vehicles_rolling_mean_3h'] = pred
        new_row['Vehicles_rolling_mean_6h'] = pred
        new_row['Vehicles_rolling_std_3h'] = 0
        
        current_data = new_row
    
    return predictions

# Function to find alternative routes
def find_alternative_routes(G, source, target, num_alternatives=2):
    # Create a copy of the graph for path finding
    G_copy = G.copy()
    
    # Set edge weights based on predicted traffic (higher traffic = higher weight)
    for u, v in G_copy.edges():
        # Use the average predicted traffic of the two nodes as the edge weight
        if 'predicted' in G_copy.nodes[u] and 'predicted' in G_copy.nodes[v]:
            traffic_u = G_copy.nodes[u]['predicted']
            traffic_v = G_copy.nodes[v]['predicted']
            weight = (traffic_u + traffic_v) / 2
            G_copy[u][v]['weight'] = weight
    
    # Find the shortest path
    try:
        shortest_path = nx.shortest_path(G_copy, source=source, target=target, weight='weight')
        
        # Find alternative paths
        alternative_paths = []
        temp_graph = G_copy.copy()
        
        for _ in range(num_alternatives):
            # Remove a random edge from the previous shortest path
            if len(shortest_path) > 2:
                # Choose a random edge from the path
                edge_idx = np.random.randint(0, len(shortest_path) - 1)
                u, v = shortest_path[edge_idx], shortest_path[edge_idx + 1]
                
                # Temporarily increase the weight of this edge
                if temp_graph.has_edge(u, v):
                    temp_graph[u][v]['weight'] *= 10
                
                try:
                    # Find a new shortest path
                    alt_path = nx.shortest_path(temp_graph, source=source, target=target, weight='weight')
                    if alt_path != shortest_path and alt_path not in alternative_paths:
                        alternative_paths.append(alt_path)
                except nx.NetworkXNoPath:
                    continue
        
        return shortest_path, alternative_paths
    
    except nx.NetworkXNoPath:
        return None, []

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🚦 Real-time Traffic Prediction System</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading traffic data..."):
        df = load_data()
    
    # Load models
    with st.spinner("Loading prediction models..."):
        ts_model, gnn_model, gnn_data, scaler_X, scaler_y, time_series_features, junctions = load_models(df)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Traffic Analysis", "Route Recommendation", "About"])
    
    # Dashboard page
    if page == "Dashboard":
        st.markdown('<h2 class="sub-header">Traffic Dashboard</h2>', unsafe_allow_html=True)
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Average Vehicles</p>', unsafe_allow_html=True)
            avg_vehicles = df['Vehicles'].mean()
            st.markdown(f'<p class="metric-value">{avg_vehicles:.0f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Total Junctions</p>', unsafe_allow_html=True)
            total_junctions = df['Junction'].nunique()
            st.markdown(f'<p class="metric-value">{total_junctions}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Congestion Index</p>', unsafe_allow_html=True)
            # Calculate a simple congestion index (percentage of max observed traffic)
            max_vehicles = df['Vehicles'].max()
            current_avg = df.groupby('DateTime')['Vehicles'].mean().iloc[-1]
            congestion = (current_avg / max_vehicles) * 100
            st.markdown(f'<p class="metric-value">{congestion:.1f}%</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Time series plot
        st.markdown('<h3 class="sub-header">Traffic Volume Over Time</h3>', unsafe_allow_html=True)
        
        # Filter by junction
        selected_junction = st.selectbox("Select Junction", sorted(df['Junction'].unique()), index=0)
        
        # Filter data for the selected junction
        junction_data = df[df['Junction'] == selected_junction].sort_values('DateTime')
        
        # Plot time series
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(junction_data['DateTime'], junction_data['Vehicles'], label='Actual')
        
        # Add some "predicted" data for demonstration
        last_timestamp = junction_data['DateTime'].max()
        future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, 13)]
        
        # Generate predictions
        hours_ahead = 12
        predictions = predict_future_traffic(
            ts_model, scaler_X, scaler_y, df, selected_junction, hours_ahead, time_series_features
        )
        
        # Plot predictions
        ax.plot(future_timestamps, predictions, 'r--', label='Predicted')
        
        ax.set_title(f'Traffic Volume at Junction {selected_junction}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Vehicles')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Traffic heatmap
        st.markdown('<h3 class="sub-header">Traffic Heatmap</h3>', unsafe_allow_html=True)
        
        # Create a pivot table for the heatmap
        junction_data['Hour'] = junction_data['DateTime'].dt.hour
        junction_data['DayOfWeek'] = junction_data['DateTime'].dt.dayofweek
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = junction_data.pivot_table(values='Vehicles', index='Hour', columns='DayOfWeek', aggfunc='mean')
        pivot.columns = [day_names[i] for i in pivot.columns]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f', ax=ax)
        ax.set_title(f'Average Traffic Volume by Hour and Day at Junction {selected_junction}')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Hour of Day')
        
        st.pyplot(fig)
    
    # Traffic Analysis page
    elif page == "Traffic Analysis":
        st.markdown('<h2 class="sub-header">Traffic Network Analysis</h2>', unsafe_allow_html=True)
        
        # Run GNN predictions
        gnn_model.eval()
        with torch.no_grad():
            pred = gnn_model(gnn_data).squeeze()
        
        # Create a graph for visualization
        G = nx.Graph()
        
        # Add nodes with predictions as attributes
        for i, junction in enumerate(junctions):
            actual = gnn_data.y[i].item()
            predicted = pred[i].item()
            G.add_node(junction, actual=actual, predicted=predicted)
        
        # Add edges based on our simple network
        for i, junction in enumerate(junctions[:-2]):
            G.add_edge(junction, junctions[i+1])
            G.add_edge(junction, junctions[i+2])
        
        # Connect the last two junctions
        if len(junctions) >= 2:
            G.add_edge(junctions[-2], junctions[-1])
        
        # Connect the last junction to the first (cycle)
        if len(junctions) > 2:
            G.add_edge(junctions[-1], junctions[0])
        
        # Visualization options
        viz_option = st.radio("Visualization Type", ["Network Graph", "Junction Comparison"])
        
        if viz_option == "Network Graph":
            # Draw the network
            fig, ax = plt.subplots(figsize=(12, 10))
            
            pos = nx.spring_layout(G, seed=42)
            
            # Color nodes based on traffic volume
            node_colors = []
            for node in G.nodes():
                traffic = G.nodes[node]['predicted']
                # Normalize traffic to [0,1] for color mapping
                max_traffic = 150  # Assuming max traffic around 150 vehicles
                normalized_traffic = min(1, max(0, traffic / max_traffic))
                # Use a color gradient: green (low traffic) to red (high traffic)
                node_colors.append((normalized_traffic, 0.8-normalized_traffic, 0))
            
            # Size nodes based on traffic volume
            node_sizes = [G.nodes[node]['predicted'] * 5 for node in G.nodes()]
            
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, 
                    font_size=10, font_weight='bold', ax=ax)
            
            ax.set_title('Traffic Network with Current Traffic Volumes')
            
            # Add a colorbar legend
            import matplotlib.patches as mpatches
            legend_patches = [
                mpatches.Patch(color='green', label='Low Traffic'),
                mpatches.Patch(color='yellow', label='Medium Traffic'),
                mpatches.Patch(color='red', label='High Traffic')
            ]
            ax.legend(handles=legend_patches, loc='upper right')
            
            st.pyplot(fig)
            
        else:  # Junction Comparison
            # Create a bar chart comparing actual vs predicted traffic
            fig, ax = plt.subplots(figsize=(12, 6))
            
            junctions_list = list(G.nodes())
            actual_values = [G.nodes[j]['actual'] for j in junctions_list]
            predicted_values = [G.nodes[j]['predicted'] for j in junctions_list]
            
            x = np.arange(len(junctions_list))
            width = 0.35
            
            ax.bar(x - width/2, actual_values, width, label='Actual')
            ax.bar(x + width/2, predicted_values, width, label='Predicted')
            
            ax.set_xlabel('Junction')
            ax.set_ylabel('Number of Vehicles')
            ax.set_title('Actual vs Predicted Traffic Volume by Junction')
            ax.set_xticks(x)
            ax.set_xticklabels(junctions_list)
            ax.legend()
            
            st.pyplot(fig)
        
        # Traffic statistics
        st.markdown('<h3 class="sub-header">Traffic Statistics</h3>', unsafe_allow_html=True)
        
        # Calculate traffic statistics
        traffic_stats = []
        for node in G.nodes():
            actual = G.nodes[node]['actual']
            predicted = G.nodes[node]['predicted']
            error = abs(actual - predicted)
            error_percent = (error / actual) * 100 if actual > 0 else 0
            
            traffic_stats.append({
                'Junction': node,
                'Actual Vehicles': int(actual),
                'Predicted Vehicles': int(predicted),
                'Error': int(error),
                'Error (%)': f"{error_percent:.1f}%"
            })
        
        traffic_df = pd.DataFrame(traffic_stats)
        traffic_df = traffic_df.sort_values('Actual Vehicles', ascending=False)
        
        # Display busiest junctions
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown('#### Busiest Junctions')
        st.dataframe(traffic_df.head(5))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display least busy junctions
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown('#### Least Busy Junctions')
        st.dataframe(traffic_df.tail(5))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Route Recommendation page
    elif page == "Route Recommendation":
        st.markdown('<h2 class="sub-header">Route Recommendation</h2>', unsafe_allow_html=True)
        
        # Run GNN predictions for the graph
        gnn_model.eval()
        with torch.no_grad():
            pred = gnn_model(gnn_data).squeeze()
        
        # Create a graph for route finding
        G = nx.Graph()
        
        # Add nodes with predictions as attributes
        for i, junction in enumerate(junctions):
            actual = gnn_data.y[i].item()
            predicted = pred[i].item()
            G.add_node(junction, actual=actual, predicted=predicted)
        
        # Add edges based on our simple network
        for i, junction in enumerate(junctions[:-2]):
            G.add_edge(junction, junctions[i+1])
            G.add_edge(junction, junctions[i+2])
        
        # Connect the last two junctions
        if len(junctions) >= 2:
            G.add_edge(junctions[-2], junctions[-1])
        
        # Connect the last junction to the first (cycle)
        if len(junctions) > 2:
            G.add_edge(junctions[-1], junctions[0])
        
        # Route selection
        col1, col2 = st.columns(2)
        
        with col1:
            source = st.selectbox("Start Junction", sorted(G.nodes()), index=0)
        
        with col2:
            target = st.selectbox("Destination Junction", sorted(G.nodes()), index=len(G.nodes())-1)
        
        if st.button("Find Routes"):
            if source == target:
                st.warning("Start and destination junctions must be different.")
            else:
                # Find routes
                main_route, alternative_routes = find_alternative_routes(G, source, target)
                
                if main_route:
                    # Display routes
                    st.markdown('<div class="highlight">', unsafe_allow_html=True)
                    st.markdown("### Recommended Routes")
                    
                    # Main route
                    st.markdown("#### Main Route")
                    route_str = " → ".join(map(str, main_route))
                    st.markdown(f"**Path:** {route_str}")
                    
                    # Calculate estimated travel time
                    total_distance = len(main_route) - 1  # Simplified: each segment has distance 1
                    avg_traffic = sum(G.nodes[node]['predicted'] for node in main_route) / len(main_route)
                    
                    # Simple travel time model: higher traffic = longer travel time
                    base_time = total_distance * 5  # 5 minutes per segment with no traffic
                    traffic_factor = 1 + (avg_traffic / 100)  # Traffic increases travel time
                    travel_time = base_time * traffic_factor
                    
                    st.markdown(f"**Estimated Travel Time:** {travel_time:.1f} minutes")
                    st.markdown(f"**Average Traffic Volume:** {avg_traffic:.1f} vehicles")
                    
                    # Alternative routes
                    if alternative_routes:
                        st.markdown("#### Alternative Routes")
                        
                        for i, route in enumerate(alternative_routes):
                            st.markdown(f"**Alternative {i+1}**")
                            route_str = " → ".join(map(str, route))
                            st.markdown(f"Path: {route_str}")
                            
                            # Calculate estimated travel time
                            total_distance = len(route) - 1
                            avg_traffic = sum(G.nodes[node]['predicted'] for node in route) / len(route)
                            
                            # Simple travel time model
                            base_time = total_distance * 5
                            traffic_factor = 1 + (avg_traffic / 100)
                            travel_time = base_time * traffic_factor
                            
                            st.markdown(f"Estimated Travel Time: {travel_time:.1f} minutes")
                            st.markdown(f"Average Traffic Volume: {avg_traffic:.1f} vehicles")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualize the routes
                    st.markdown("### Route Visualization")
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Draw the base network
                    pos = nx.spring_layout(G, seed=42)
                    nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=500, 
                            font_size=10, font_weight='bold', edge_color='lightgray', ax=ax)
                    
                    # Draw the main route
                    main_route_edges = list(zip(main_route[:-1], main_route[1:]))
                    nx.draw_networkx_edges(G, pos, edgelist=main_route_edges, width=3, edge_color='blue', ax=ax)
                    
                    # Draw alternative routes with different colors
                    colors = ['green', 'red', 'purple']
                    for i, route in enumerate(alternative_routes):
                        if i < len(colors):
                            alt_route_edges = list(zip(route[:-1], route[1:]))
                            nx.draw_networkx_edges(G, pos, edgelist=alt_route_edges, width=2, 
                                                edge_color=colors[i], style='dashed', ax=ax)
                    
                    # Add a legend
                    from matplotlib.lines import Line2D
                    legend_elements = [Line2D([0], [0], color='blue', lw=3, label='Main Route')]
                    for i, color in enumerate(colors[:len(alternative_routes)]):
                        legend_elements.append(Line2D([0], [0], color=color, lw=2, ls='--', 
                                                    label=f'Alternative {i+1}'))
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    ax.set_title(f'Route Recommendations from Junction {source} to Junction {target}')
                    
                    st.pyplot(fig)
                else:
                    st.error(f"No path found between Junction {source} and Junction {target}")
    
    # About page
    else:
        st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-text">
        <p>This real-time traffic prediction system uses advanced machine learning techniques to predict traffic conditions and recommend optimal routes between junctions.</p>
        
        <h3>Key Features:</h3>
        <ul>
            <li><strong>Time Series Analysis:</strong> Predicts future traffic volumes based on historical patterns</li>
            <li><strong>Graph Neural Networks:</strong> Models the junction network to capture spatial relationships between junctions</li>
            <li><strong>Route Recommendation:</strong> Suggests optimal routes based on predicted traffic volumes</li>
            <li><strong>Interactive Visualization:</strong> Provides intuitive visualizations of traffic conditions</li>
        </ul>
        
        <h3>Technologies Used:</h3>
        <ul>
            <li>Python</li>
            <li>Streamlit</li>
            <li>PyTorch & PyTorch Geometric</li>
            <li>NetworkX</li>
            <li>Pandas & NumPy</li>
            <li>Matplotlib & Seaborn</li>
        </ul>
        
        <h3>Use Cases:</h3>
        <ul>
            <li>Urban Planning</li>
            <li>Logistics Optimization</li>
            <li>Traffic Management</li>
            <li>Navigation Systems</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()