"""
In-Place Inclinometer (IPI) Cumulative Displacement Dashboard
==============================================================
A comprehensive Streamlit web application for visualizing IPI monitoring data.

Features:
- Auto-detection of Campbell Scientific TOA5 format and generic CSV/DAT files
- Configurable gauge length and depth settings
- Base reading correction (initial reading subtraction)
- Cumulative displacement calculation (bottom-up summation)
- Interactive Plotly charts for profile and trend analysis
- Resultant displacement calculation
- Temperature monitoring visualization

Author: Geotechnical Data Analysis Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="IPI Displacement Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def parse_toa5_file(file_content: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse Campbell Scientific TOA5 format file.
    
    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    lines = file_content.strip().split('\n')
    
    # Parse header line (line 0)
    header_info = lines[0].replace('"', '').split(',')
    metadata = {
        'format': header_info[0] if len(header_info) > 0 else 'Unknown',
        'station_name': header_info[1] if len(header_info) > 1 else 'Unknown',
        'logger_model': header_info[2] if len(header_info) > 2 else 'Unknown',
        'serial_number': header_info[3] if len(header_info) > 3 else 'Unknown',
        'program_name': header_info[5] if len(header_info) > 5 else 'Unknown',
        'table_name': header_info[7] if len(header_info) > 7 else 'Unknown'
    }
    
    # Column names (line 1)
    columns = [col.replace('"', '') for col in lines[1].split(',')]
    
    # Units (line 2) and processing info (line 3) - skip for data parsing
    
    # Data starts from line 4
    data_lines = '\n'.join(lines[4:])
    
    df = pd.read_csv(io.StringIO(data_lines), names=columns, na_values=['NAN', 'NaN', 'nan', ''])
    
    # Parse timestamp
    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        df = df.dropna(subset=['TIMESTAMP'])
    
    return df, metadata


def detect_ipi_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect IPI-related columns in the dataframe.
    
    Returns:
        Dictionary with detected column groups
    """
    columns = df.columns.tolist()
    
    detected = {
        'timestamp': None,
        'tilt_a': [],
        'tilt_b': [],
        'def_a': [],
        'def_b': [],
        'therm': [],
        'battery': None,
        'panel_temp': None,
        'num_sensors': 0
    }
    
    for col in columns:
        col_lower = col.lower()
        
        if 'timestamp' in col_lower or col_lower == 'ts':
            detected['timestamp'] = col
        elif 'battv' in col_lower or 'batt_v' in col_lower:
            detected['battery'] = col
        elif 'ptemp' in col_lower or 'panel' in col_lower:
            detected['panel_temp'] = col
        elif 'tilt_a' in col_lower or 'tilta' in col_lower:
            detected['tilt_a'].append(col)
        elif 'tilt_b' in col_lower or 'tiltb' in col_lower:
            detected['tilt_b'].append(col)
        elif 'def_a' in col_lower or 'defa' in col_lower:
            detected['def_a'].append(col)
        elif 'def_b' in col_lower or 'defb' in col_lower:
            detected['def_b'].append(col)
        elif 'therm' in col_lower or 'temp' in col_lower:
            if 'panel' not in col_lower and 'ptemp' not in col_lower:
                detected['therm'].append(col)
    
    # Sort columns by sensor number
    def extract_number(col_name):
        match = re.search(r'\((\d+)\)|\[(\d+)\]|_(\d+)$', col_name)
        if match:
            return int(next(g for g in match.groups() if g is not None))
        return 0
    
    for key in ['tilt_a', 'tilt_b', 'def_a', 'def_b', 'therm']:
        detected[key] = sorted(detected[key], key=extract_number)
    
    # Determine number of sensors
    detected['num_sensors'] = max(
        len(detected['tilt_a']),
        len(detected['tilt_b']),
        len(detected['def_a']),
        len(detected['def_b'])
    )
    
    return detected


def calculate_incremental_displacement(tilt_sin: float, gauge_length: float) -> float:
    """
    Calculate incremental displacement from tilt reading.
    
    Formula: displacement = L × sin(θ)
    Since the data is already sin(θ), we just multiply by gauge length.
    
    Args:
        tilt_sin: Sine of tilt angle (raw sensor reading)
        gauge_length: Gauge length in meters
        
    Returns:
        Incremental displacement in mm
    """
    if pd.isna(tilt_sin):
        return np.nan
    return tilt_sin * gauge_length * 1000  # Convert to mm


def calculate_cumulative_displacement(incremental_displacements: np.ndarray, from_bottom: bool = True) -> np.ndarray:
    """
    Calculate cumulative displacement from incremental values.
    
    Args:
        incremental_displacements: Array of incremental displacements (sensor 1 = top, sensor N = bottom)
        from_bottom: If True, sum from bottom up (assuming stable toe)
        
    Returns:
        Array of cumulative displacements
    """
    if from_bottom:
        # Reverse, cumsum, then reverse back
        return np.flip(np.nancumsum(np.flip(incremental_displacements)))
    else:
        return np.nancumsum(incremental_displacements)


def process_ipi_data(
    df: pd.DataFrame,
    detected_cols: dict,
    gauge_length: float,
    top_depth: float,
    use_raw_tilt: bool = True,
    base_reading_idx: int = 0
) -> pd.DataFrame:
    """
    Process IPI data to calculate cumulative displacements.
    
    Args:
        df: Raw dataframe
        detected_cols: Detected column mapping
        gauge_length: Gauge length in meters
        top_depth: Depth of topmost sensor in meters
        use_raw_tilt: If True, use tilt values; if False, use pre-calculated deflection
        base_reading_idx: Index of reading to use as base (0 = first reading)
        
    Returns:
        Processed dataframe with cumulative displacements
    """
    num_sensors = detected_cols['num_sensors']
    timestamps = df[detected_cols['timestamp']].values
    
    # Generate depth array (sensor 1 = top, sensor N = bottom)
    depths = np.array([top_depth + i * gauge_length for i in range(num_sensors)])
    
    results = []
    
    for idx, row in df.iterrows():
        timestamp = row[detected_cols['timestamp']]
        
        # Extract A-axis and B-axis data
        if use_raw_tilt and detected_cols['tilt_a'] and detected_cols['tilt_b']:
            tilt_a = np.array([row[col] for col in detected_cols['tilt_a']])
            tilt_b = np.array([row[col] for col in detected_cols['tilt_b']])
            
            # Calculate incremental displacement
            inc_a = np.array([calculate_incremental_displacement(t, gauge_length) for t in tilt_a])
            inc_b = np.array([calculate_incremental_displacement(t, gauge_length) for t in tilt_b])
        elif detected_cols['def_a'] and detected_cols['def_b']:
            # Use pre-calculated deflection values
            inc_a = np.array([row[col] for col in detected_cols['def_a']])
            inc_b = np.array([row[col] for col in detected_cols['def_b']])
        else:
            continue
        
        # Get temperature if available
        if detected_cols['therm']:
            temps = np.array([row[col] for col in detected_cols['therm'][:num_sensors]])
        else:
            temps = np.full(num_sensors, np.nan)
        
        for i in range(num_sensors):
            results.append({
                'timestamp': timestamp,
                'record_idx': idx,
                'sensor_num': i + 1,
                'depth': depths[i],
                'inc_disp_a': inc_a[i] if i < len(inc_a) else np.nan,
                'inc_disp_b': inc_b[i] if i < len(inc_b) else np.nan,
                'temperature': temps[i] if i < len(temps) else np.nan
            })
    
    processed_df = pd.DataFrame(results)
    
    if processed_df.empty:
        return processed_df
    
    # Apply base reading correction
    base_data = processed_df[processed_df['record_idx'] == base_reading_idx].copy()
    base_data = base_data.set_index('sensor_num')[['inc_disp_a', 'inc_disp_b']].rename(
        columns={'inc_disp_a': 'base_a', 'inc_disp_b': 'base_b'}
    )
    
    processed_df = processed_df.merge(base_data, left_on='sensor_num', right_index=True, how='left')
    processed_df['inc_disp_a_corr'] = processed_df['inc_disp_a'] - processed_df['base_a']
    processed_df['inc_disp_b_corr'] = processed_df['inc_disp_b'] - processed_df['base_b']
    
    # Calculate cumulative displacement for each timestamp
    cum_disp_a_list = []
    cum_disp_b_list = []
    
    for timestamp in processed_df['timestamp'].unique():
        mask = processed_df['timestamp'] == timestamp
        inc_a = processed_df.loc[mask, 'inc_disp_a_corr'].values
        inc_b = processed_df.loc[mask, 'inc_disp_b_corr'].values
        
        cum_a = calculate_cumulative_displacement(inc_a, from_bottom=True)
        cum_b = calculate_cumulative_displacement(inc_b, from_bottom=True)
        
        cum_disp_a_list.extend(cum_a)
        cum_disp_b_list.extend(cum_b)
    
    processed_df['cum_disp_a'] = cum_disp_a_list
    processed_df['cum_disp_b'] = cum_disp_b_list
    
    # Calculate resultant displacement
    processed_df['cum_disp_resultant'] = np.sqrt(
        processed_df['cum_disp_a']**2 + processed_df['cum_disp_b']**2
    )
    
    return processed_df


def create_profile_plot(
    processed_df: pd.DataFrame,
    selected_timestamps: list,
    axis: str = 'A',
    show_resultant: bool = False
) -> go.Figure:
    """
    Create displacement profile plot (Displacement vs Depth).
    """
    fig = go.Figure()
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    for i, timestamp in enumerate(selected_timestamps):
        mask = processed_df['timestamp'] == timestamp
        data = processed_df[mask].sort_values('depth')
        
        color = colors[i % len(colors)]
        ts_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        
        if show_resultant:
            fig.add_trace(go.Scatter(
                x=data['cum_disp_resultant'],
                y=data['depth'],
                mode='lines+markers',
                name=f'Resultant - {ts_str}',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate='<b>Depth:</b> %{y:.2f} m<br><b>Displacement:</b> %{x:.3f} mm<extra></extra>'
            ))
        else:
            disp_col = 'cum_disp_a' if axis == 'A' else 'cum_disp_b'
            fig.add_trace(go.Scatter(
                x=data[disp_col],
                y=data['depth'],
                mode='lines+markers',
                name=f'{axis}-Axis - {ts_str}',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate='<b>Depth:</b> %{y:.2f} m<br><b>Displacement:</b> %{x:.3f} mm<extra></extra>'
            ))
    
    # Add zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=dict(
            text='<b>IPI Cumulative Displacement Profile</b>',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Cumulative Displacement (mm)',
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        yaxis=dict(
            title='Depth (m)',
            autorange='reversed',  # Depth increases downward
            gridcolor='lightgray'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
        ),
        template='plotly_white',
        hovermode='closest',
        height=600
    )
    
    return fig


def create_trend_plot(
    processed_df: pd.DataFrame,
    selected_depths: list,
    axis: str = 'A',
    show_resultant: bool = False
) -> go.Figure:
    """
    Create displacement trend plot (Displacement vs Time).
    """
    fig = go.Figure()
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    all_depths = sorted(processed_df['depth'].unique())
    
    for i, depth in enumerate(selected_depths):
        # Find closest depth
        closest_depth = min(all_depths, key=lambda x: abs(x - depth))
        mask = processed_df['depth'] == closest_depth
        data = processed_df[mask].sort_values('timestamp')
        
        color = colors[i % len(colors)]
        
        if show_resultant:
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['cum_disp_resultant'],
                mode='lines+markers',
                name=f'Resultant @ {closest_depth:.1f}m',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Time:</b> %{x}<br><b>Displacement:</b> %{y:.3f} mm<extra></extra>'
            ))
        else:
            disp_col = 'cum_disp_a' if axis == 'A' else 'cum_disp_b'
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data[disp_col],
                mode='lines+markers',
                name=f'{axis}-Axis @ {closest_depth:.1f}m',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Time:</b> %{x}<br><b>Displacement:</b> %{y:.3f} mm<extra></extra>'
            ))
    
    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=dict(
            text='<b>IPI Displacement Time History</b>',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Date/Time',
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Cumulative Displacement (mm)',
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
        ),
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_temperature_plot(processed_df: pd.DataFrame, selected_depths: list) -> go.Figure:
    """
    Create temperature trend plot.
    """
    fig = go.Figure()
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    all_depths = sorted(processed_df['depth'].unique())
    
    for i, depth in enumerate(selected_depths):
        closest_depth = min(all_depths, key=lambda x: abs(x - depth))
        mask = processed_df['depth'] == closest_depth
        data = processed_df[mask].sort_values('timestamp')
        
        if data['temperature'].notna().any():
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['temperature'],
                mode='lines',
                name=f'{closest_depth:.1f}m',
                line=dict(color=colors[i % len(colors)], width=1.5),
                hovertemplate='<b>Time:</b> %{x}<br><b>Temperature:</b> %{y:.1f} °C<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text='<b>Sensor Temperature History</b>',
            font=dict(size=16)
        ),
        xaxis=dict(title='Date/Time', gridcolor='lightgray'),
        yaxis=dict(title='Temperature (°C)', gridcolor='lightgray'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        template='plotly_white',
        height=350
    )
    
    return fig


def create_polar_plot(processed_df: pd.DataFrame, timestamp) -> go.Figure:
    """
    Create polar/vector plot showing A vs B displacement.
    """
    mask = processed_df['timestamp'] == timestamp
    data = processed_df[mask].sort_values('depth')
    
    fig = go.Figure()
    
    colors = [f'rgb({int(255*(1-i/len(data)))}, {int(100+50*i/len(data))}, {int(200*i/len(data))})' 
              for i in range(len(data))]
    
    for i, (_, row) in enumerate(data.iterrows()):
        fig.add_trace(go.Scatter(
            x=[0, row['cum_disp_a']],
            y=[0, row['cum_disp_b']],
            mode='lines+markers',
            name=f"{row['depth']:.1f}m",
            line=dict(color=colors[i], width=2),
            marker=dict(size=[4, 8]),
            hovertemplate=f"<b>Depth:</b> {row['depth']:.1f}m<br><b>A-Axis:</b> {row['cum_disp_a']:.3f}mm<br><b>B-Axis:</b> {row['cum_disp_b']:.3f}mm<extra></extra>"
        ))
    
    # Add reference circles
    max_disp = max(
        data['cum_disp_a'].abs().max(),
        data['cum_disp_b'].abs().max()
    ) * 1.2
    
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [max_disp/3, 2*max_disp/3, max_disp]:
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta),
            y=r * np.sin(theta),
            mode='lines',
            line=dict(color='lightgray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>Displacement Vector Plot</b><br><sub>{pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M")}</sub>',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='A-Axis Displacement (mm)',
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray',
            scaleanchor='y'
        ),
        yaxis=dict(
            title='B-Axis Displacement (mm)',
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray'
        ),
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02)
    )
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">📊 IPI Displacement Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">In-Place Inclinometer Cumulative Displacement Visualization</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("1. Data Upload")
        uploaded_file = st.file_uploader(
            "Upload IPI Data File",
            type=['dat', 'csv', 'txt'],
            help="Upload Campbell Scientific TOA5 format or generic CSV file"
        )
        
        st.divider()
        
        # Sensor configuration
        st.subheader("2. Sensor Parameters")
        gauge_length = st.number_input(
            "Gauge Length (m)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Distance between sensor nodes (default: 1.0m)"
        )
        
        top_depth = st.number_input(
            "Top Sensor Depth (m)",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.5,
            help="Depth of the topmost sensor from ground surface"
        )
        
        data_source = st.radio(
            "Data Source",
            options=['Raw Tilt (sin θ)', 'Pre-calculated Deflection'],
            index=0,
            help="Select whether to use raw tilt values or pre-calculated deflection"
        )
        use_raw_tilt = data_source == 'Raw Tilt (sin θ)'
        
        st.divider()
        
        # Display options
        st.subheader("3. Display Options")
        axis_select = st.radio(
            "Axis Selection",
            options=['A-Axis', 'B-Axis', 'Resultant'],
            index=0
        )
        show_resultant = axis_select == 'Resultant'
        axis = 'A' if axis_select == 'A-Axis' else 'B'
    
    # Main content area
    if uploaded_file is None:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload an IPI data file using the sidebar to begin analysis.")
        
        with st.expander("📖 About This Dashboard", expanded=True):
            st.markdown("""
            ### Features
            
            This dashboard provides comprehensive visualization for In-Place Inclinometer (IPI) monitoring data:
            
            - **Auto-detection** of Campbell Scientific TOA5 format files
            - **Flexible configuration** for gauge length and sensor depths
            - **Base reading correction** (current reading - initial reading)
            - **Bottom-up cumulative summation** assuming stable toe
            - **Interactive Plotly charts** with zoom, pan, and hover
            
            ### Calculation Methods
            
            **Incremental Displacement:**
            $$d_{inc} = L \\times \\sin(\\theta)$$
            
            Where $L$ is the gauge length and $\\theta$ is the tilt angle.
            
            **Cumulative Displacement (Bottom-Up):**
            $$D_i = \\sum_{j=i}^{n} d_j$$
            
            Where $n$ is the bottom sensor (assumed stable).
            
            ### Supported File Formats
            - Campbell Scientific TOA5 (.dat, .csv)
            - Generic CSV with timestamp and sensor columns
            """)
        
        return
    
    # Process uploaded file
    try:
        file_content = uploaded_file.read().decode('utf-8')
        
        # Detect file format
        if file_content.startswith('"TOA5"'):
            df, metadata = parse_toa5_file(file_content)
            st.success(f"✅ TOA5 file loaded: **{metadata['station_name']}** ({metadata['table_name']})")
        else:
            # Generic CSV parsing
            df = pd.read_csv(io.StringIO(file_content), na_values=['NAN', 'NaN', 'nan', ''])
            metadata = {'station_name': 'Unknown', 'table_name': 'Generic CSV'}
            st.success("✅ CSV file loaded successfully")
        
        # Detect columns
        detected_cols = detect_ipi_columns(df)
        
        if detected_cols['num_sensors'] == 0:
            st.error("❌ Could not detect IPI sensor columns in the uploaded file.")
            return
        
        # Display detection results
        with st.expander("📋 Detected Data Structure", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Sensors", detected_cols['num_sensors'])
            with col2:
                st.metric("Total Records", len(df))
            with col3:
                if detected_cols['timestamp']:
                    date_range = f"{df[detected_cols['timestamp']].min().strftime('%Y-%m-%d')} to {df[detected_cols['timestamp']].max().strftime('%Y-%m-%d')}"
                    st.metric("Date Range", date_range)
        
        # Base reading selection
        with st.sidebar:
            st.divider()
            st.subheader("4. Base Reading")
            
            timestamps = df[detected_cols['timestamp']].unique()
            base_options = {f"{i}: {pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M')}": i 
                          for i, ts in enumerate(timestamps[:50])}  # Limit to first 50
            
            selected_base = st.selectbox(
                "Select Base Reading",
                options=list(base_options.keys()),
                index=0,
                help="Initial reading for correction (Current - Base)"
            )
            base_reading_idx = base_options[selected_base]
        
        # Process data
        with st.spinner("Processing IPI data..."):
            processed_df = process_ipi_data(
                df,
                detected_cols,
                gauge_length,
                top_depth,
                use_raw_tilt,
                base_reading_idx
            )
        
        if processed_df.empty:
            st.error("❌ No data could be processed. Please check the file format.")
            return
        
        # Summary metrics
        st.subheader("📊 Summary Statistics")
        
        # Get latest data
        latest_timestamp = processed_df['timestamp'].max()
        latest_data = processed_df[processed_df['timestamp'] == latest_timestamp]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_disp_a = latest_data['cum_disp_a'].abs().max()
            st.metric("Max A-Axis Disp.", f"{max_disp_a:.2f} mm")
        with col2:
            max_disp_b = latest_data['cum_disp_b'].abs().max()
            st.metric("Max B-Axis Disp.", f"{max_disp_b:.2f} mm")
        with col3:
            max_resultant = latest_data['cum_disp_resultant'].max()
            st.metric("Max Resultant", f"{max_resultant:.2f} mm")
        with col4:
            depth_at_max = latest_data.loc[latest_data['cum_disp_resultant'].idxmax(), 'depth']
            st.metric("Depth at Max", f"{depth_at_max:.1f} m")
        
        st.divider()
        
        # Plot tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Profile Plot", "📉 Trend Plot", "🎯 Vector Plot", "🌡️ Temperature"])
        
        with tab1:
            st.subheader("Displacement Profile")
            
            # Date selection for profile plot
            available_dates = sorted(processed_df['timestamp'].unique())
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_dates = st.multiselect(
                    "Select Timestamps to Compare",
                    options=available_dates,
                    default=[available_dates[0], available_dates[-1]] if len(available_dates) > 1 else [available_dates[0]],
                    format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d %H:%M'),
                    max_selections=10
                )
            with col2:
                st.write("")  # Spacing
                if st.button("Select Latest", key="latest_profile"):
                    selected_dates = [available_dates[-1]]
            
            if selected_dates:
                fig_profile = create_profile_plot(processed_df, selected_dates, axis, show_resultant)
                st.plotly_chart(fig_profile, use_container_width=True)
        
        with tab2:
            st.subheader("Displacement Time History")
            
            # Depth selection for trend plot
            available_depths = sorted(processed_df['depth'].unique())
            
            selected_depths = st.multiselect(
                "Select Depths to Monitor",
                options=available_depths,
                default=[available_depths[0], available_depths[len(available_depths)//2], available_depths[-1]],
                format_func=lambda x: f"{x:.1f} m",
                max_selections=10
            )
            
            if selected_depths:
                fig_trend = create_trend_plot(processed_df, selected_depths, axis, show_resultant)
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with tab3:
            st.subheader("Vector Plot (A vs B Displacement)")
            
            selected_vector_date = st.select_slider(
                "Select Timestamp",
                options=available_dates,
                value=available_dates[-1],
                format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d %H:%M')
            )
            
            fig_polar = create_polar_plot(processed_df, selected_vector_date)
            st.plotly_chart(fig_polar, use_container_width=True)
        
        with tab4:
            st.subheader("Temperature Monitoring")
            
            if processed_df['temperature'].notna().any():
                temp_depths = st.multiselect(
                    "Select Depths for Temperature",
                    options=available_depths,
                    default=[available_depths[0], available_depths[-1]],
                    format_func=lambda x: f"{x:.1f} m",
                    key="temp_depths"
                )
                
                if temp_depths:
                    fig_temp = create_temperature_plot(processed_df, temp_depths)
                    st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.info("No temperature data available in this file.")
        
        # Data export
        st.divider()
        with st.expander("💾 Export Processed Data"):
            export_df = processed_df[['timestamp', 'sensor_num', 'depth', 
                                      'inc_disp_a_corr', 'inc_disp_b_corr',
                                      'cum_disp_a', 'cum_disp_b', 'cum_disp_resultant',
                                      'temperature']].copy()
            export_df.columns = ['Timestamp', 'Sensor', 'Depth (m)', 
                                'Inc. Disp A (mm)', 'Inc. Disp B (mm)',
                                'Cum. Disp A (mm)', 'Cum. Disp B (mm)', 'Resultant (mm)',
                                'Temperature (°C)']
            
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Processed Data (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"IPI_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
