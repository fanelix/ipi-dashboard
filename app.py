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

# Custom CSS for professional white theme
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a365d;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 8px;
        border-bottom: 3px solid #2563eb;
    }
    
    /* Sub header */
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #1e293b;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 6px;
        color: #1e293b;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563eb;
        color: #ffffff;
    }
    
    /* Data editor / table styling */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 6px;
    }
    
    /* Selectbox and input styling */
    .stSelectbox, .stNumberInput, .stDateInput {
        color: #1e293b;
    }
    
    /* Gauge length table header */
    .gauge-table-header {
        background-color: #2563eb;
        color: white;
        padding: 0.5rem;
        border-radius: 4px 4px 0 0;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def clean_and_split_lines(file_content: str) -> list:
    """
    Clean file content and handle concatenated lines (common data corruption).
    
    Some Campbell Scientific files have lines that are concatenated together
    (missing newline characters). This function detects and splits them.
    
    Returns:
        List of cleaned lines
    """
    # Replace Windows line endings and handle carriage returns
    content = file_content.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split by newlines first
    lines = content.split('\n')
    
    cleaned_lines = []
    
    for line in lines:
        # Check if line contains multiple timestamps (concatenated records)
        # Pattern: ends with number/quote, immediately followed by quote + timestamp
        # Example: ...26.0625"2025-12-05 07:00:00",...
        
        # Find all timestamp patterns in the line
        timestamp_pattern = r'"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"'
        timestamps = list(re.finditer(timestamp_pattern, line))
        
        if len(timestamps) > 1:
            # Multiple timestamps found - need to split
            # Split at each timestamp (except the first one)
            last_end = 0
            for i, match in enumerate(timestamps):
                if i == 0:
                    continue  # Skip first timestamp
                
                # Find the split point (just before the quote of the timestamp)
                split_point = match.start()
                
                # Extract the segment
                segment = line[last_end:split_point].strip()
                if segment:
                    cleaned_lines.append(segment)
                
                last_end = split_point
            
            # Add the last segment
            if last_end < len(line):
                segment = line[last_end:].strip()
                if segment:
                    cleaned_lines.append(segment)
        else:
            # Normal line
            if line.strip():
                cleaned_lines.append(line.strip())
    
    return cleaned_lines


def parse_toa5_file(file_content: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse Campbell Scientific TOA5 format file with robust error handling.
    
    Handles:
    - Concatenated lines (missing newlines)
    - Inconsistent field counts
    - Various data corruption issues
    
    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    # Clean and split lines properly
    lines = clean_and_split_lines(file_content)
    
    if len(lines) < 5:
        raise ValueError("File too short - expected at least 5 lines (4 header + 1 data)")
    
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
    expected_fields = len(columns)
    
    # Data starts from line 4 (skip lines 2 and 3 which are units and processing)
    data_lines = lines[4:]
    
    # Parse each line individually, handling field count mismatches
    valid_rows = []
    skipped_rows = 0
    
    for i, line in enumerate(data_lines):
        try:
            # Parse the line
            # Handle quoted fields properly
            fields = []
            in_quote = False
            current_field = ""
            
            for char in line:
                if char == '"':
                    in_quote = not in_quote
                elif char == ',' and not in_quote:
                    fields.append(current_field.strip().strip('"'))
                    current_field = ""
                else:
                    current_field += char
            
            # Don't forget the last field
            fields.append(current_field.strip().strip('"'))
            
            # Check field count
            if len(fields) == expected_fields:
                valid_rows.append(fields)
            elif len(fields) > expected_fields:
                # Too many fields - truncate to expected
                valid_rows.append(fields[:expected_fields])
                skipped_rows += 1
            else:
                # Too few fields - pad with NaN
                fields.extend([np.nan] * (expected_fields - len(fields)))
                valid_rows.append(fields)
                skipped_rows += 1
                
        except Exception as e:
            skipped_rows += 1
            continue
    
    if not valid_rows:
        raise ValueError("No valid data rows found in file")
    
    # Create DataFrame
    df = pd.DataFrame(valid_rows, columns=columns)
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ['TIMESTAMP']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse timestamp
    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        df = df.dropna(subset=['TIMESTAMP'])
    
    # Sort by timestamp
    if 'TIMESTAMP' in df.columns:
        df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    
    # Store skipped rows info in metadata
    metadata['skipped_rows'] = skipped_rows
    metadata['total_rows'] = len(valid_rows)
    
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
    gauge_lengths: np.ndarray,
    top_depth: float,
    use_raw_tilt: bool = True,
    base_reading_idx: int = 0
) -> pd.DataFrame:
    """
    Process IPI data to calculate cumulative displacements.
    
    Args:
        df: Raw dataframe
        detected_cols: Detected column mapping
        gauge_lengths: Array of gauge lengths per sensor in meters
        top_depth: Depth of topmost sensor in meters
        use_raw_tilt: If True, use tilt values; if False, use pre-calculated deflection
        base_reading_idx: Index of reading to use as base (0 = first reading)
        
    Returns:
        Processed dataframe with cumulative displacements
    """
    num_sensors = detected_cols['num_sensors']
    timestamps = df[detected_cols['timestamp']].values
    
    # Ensure gauge_lengths is the right size
    if len(gauge_lengths) != num_sensors:
        # Pad or truncate to match num_sensors
        if len(gauge_lengths) < num_sensors:
            gauge_lengths = np.concatenate([gauge_lengths, np.full(num_sensors - len(gauge_lengths), gauge_lengths[-1] if len(gauge_lengths) > 0 else 1.0)])
        else:
            gauge_lengths = gauge_lengths[:num_sensors]
    
    # Generate depth array based on cumulative gauge lengths (sensor 1 = top, sensor N = bottom)
    depths = np.zeros(num_sensors)
    depths[0] = top_depth
    for i in range(1, num_sensors):
        depths[i] = depths[i-1] + gauge_lengths[i-1]
    
    results = []
    
    for idx, row in df.iterrows():
        timestamp = row[detected_cols['timestamp']]
        
        # Extract A-axis and B-axis data
        if use_raw_tilt and detected_cols['tilt_a'] and detected_cols['tilt_b']:
            tilt_a = np.array([row[col] for col in detected_cols['tilt_a']])
            tilt_b = np.array([row[col] for col in detected_cols['tilt_b']])
            
            # Calculate incremental displacement with per-sensor gauge length
            inc_a = np.array([calculate_incremental_displacement(tilt_a[i], gauge_lengths[i]) 
                            for i in range(min(len(tilt_a), num_sensors))])
            inc_b = np.array([calculate_incremental_displacement(tilt_b[i], gauge_lengths[i]) 
                            for i in range(min(len(tilt_b), num_sensors))])
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
                'gauge_length': gauge_lengths[i],
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


def create_profile_plot_dual(
    processed_df: pd.DataFrame,
    selected_timestamps: list
) -> go.Figure:
    """
    Create side-by-side displacement profile plots for A-axis and B-axis.
    
    Args:
        processed_df: Processed dataframe with displacement data
        selected_timestamps: List of timestamps to display
        
    Returns:
        Plotly figure with two subplots (A-axis and B-axis)
    """
    # Create subplots - 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>A-Axis Displacement</b>', '<b>B-Axis Displacement</b>'),
        shared_yaxes=True,
        horizontal_spacing=0.10
    )
    
    # High contrast colors for white background
    colors = [
        '#2563eb',  # Blue
        '#dc2626',  # Red
        '#16a34a',  # Green
        '#9333ea',  # Purple
        '#ea580c',  # Orange
        '#0891b2',  # Cyan
        '#c026d3',  # Magenta
        '#4f46e5',  # Indigo
        '#059669',  # Emerald
        '#d97706',  # Amber
    ]
    
    for i, timestamp in enumerate(selected_timestamps):
        mask = processed_df['timestamp'] == timestamp
        data = processed_df[mask].sort_values('depth')
        
        color = colors[i % len(colors)]
        ts_str = pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        
        # A-Axis plot (left)
        fig.add_trace(
            go.Scatter(
                x=data['cum_disp_a'],
                y=data['depth'],
                mode='lines+markers',
                name=f'{ts_str}',
                line=dict(color=color, width=2.5),
                marker=dict(size=7, symbol='circle'),
                hovertemplate='<b>Depth:</b> %{y:.2f} m<br><b>A-Axis:</b> %{x:.3f} mm<extra></extra>',
                legendgroup=f'group{i}',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # B-Axis plot (right)
        fig.add_trace(
            go.Scatter(
                x=data['cum_disp_b'],
                y=data['depth'],
                mode='lines+markers',
                name=f'{ts_str}',
                line=dict(color=color, width=2.5),
                marker=dict(size=7, symbol='circle'),
                hovertemplate='<b>Depth:</b> %{y:.2f} m<br><b>B-Axis:</b> %{x:.3f} mm<extra></extra>',
                legendgroup=f'group{i}',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Add zero reference lines to both subplots
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=2)
    
    # Update layout - Professional white theme
    fig.update_layout(
        title=dict(
            text='<b>IPI Cumulative Displacement Profile</b>',
            font=dict(size=18, color='#1e293b', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top'
        ),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.12,
            xanchor='center',
            x=0.5,
            title=dict(text='<b>Timestamp:</b> ', font=dict(size=11, color='#374151')),
            bgcolor='#f8fafc',
            bordercolor='#cbd5e1',
            borderwidth=1,
            font=dict(size=10, color='#1e293b')
        ),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        hovermode='closest',
        height=650,
        margin=dict(t=60, b=80, l=70, r=50)
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['y'] = 1.02
        annotation['font'] = dict(size=14, color='#1e293b', family='Arial, sans-serif')
    
    # Update x-axes with professional styling
    axis_style = dict(
        title_font=dict(size=12, color='#374151', family='Arial, sans-serif'),
        tickfont=dict(size=10, color='#4b5563'),
        gridcolor='#e5e7eb',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='#9ca3af',
        zerolinewidth=1.5,
        linecolor='#d1d5db',
        linewidth=1,
        showline=True,
        mirror=True
    )
    
    fig.update_xaxes(title_text='Cumulative Displacement (mm)', **axis_style, row=1, col=1)
    fig.update_xaxes(title_text='Cumulative Displacement (mm)', **axis_style, row=1, col=2)
    
    # Update y-axes (reversed for depth)
    fig.update_yaxes(
        title_text='Depth (m)',
        autorange='reversed',
        **axis_style,
        row=1, col=1
    )
    fig.update_yaxes(
        autorange='reversed',
        **axis_style,
        row=1, col=2
    )
    
    return fig


def create_profile_plot_resultant(
    processed_df: pd.DataFrame,
    selected_timestamps: list
) -> go.Figure:
    """
    Create resultant displacement profile plot.
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
        
        fig.add_trace(go.Scatter(
            x=data['cum_disp_resultant'],
            y=data['depth'],
            mode='lines+markers',
            name=f'{ts_str}',
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate='<b>Depth:</b> %{y:.2f} m<br><b>Resultant:</b> %{x:.3f} mm<extra></extra>'
        ))
    
    # Add zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=dict(
            text='<b>IPI Resultant Displacement Profile</b>',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Resultant Displacement (mm)',
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        yaxis=dict(
            title='Depth (m)',
            autorange='reversed',
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


def create_trend_plot_dual(
    processed_df: pd.DataFrame,
    selected_depths: list
) -> go.Figure:
    """
    Create side-by-side displacement trend plots for A-axis and B-axis.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>A-Axis Time History</b>', '<b>B-Axis Time History</b>'),
        shared_yaxes=False,
        horizontal_spacing=0.10
    )
    
    # High contrast colors for white background
    colors = [
        '#2563eb',  # Blue
        '#dc2626',  # Red
        '#16a34a',  # Green
        '#9333ea',  # Purple
        '#ea580c',  # Orange
        '#0891b2',  # Cyan
        '#c026d3',  # Magenta
        '#4f46e5',  # Indigo
        '#059669',  # Emerald
        '#d97706',  # Amber
    ]
    
    all_depths = sorted(processed_df['depth'].unique())
    
    for i, depth in enumerate(selected_depths):
        closest_depth = min(all_depths, key=lambda x: abs(x - depth))
        mask = processed_df['depth'] == closest_depth
        data = processed_df[mask].sort_values('timestamp')
        
        color = colors[i % len(colors)]
        
        # A-Axis trend (left)
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['cum_disp_a'],
                mode='lines+markers',
                name=f'{closest_depth:.1f}m',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Time:</b> %{x}<br><b>A-Axis:</b> %{y:.3f} mm<extra></extra>',
                legendgroup=f'depth{i}',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # B-Axis trend (right)
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['cum_disp_b'],
                mode='lines+markers',
                name=f'{closest_depth:.1f}m',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate='<b>Time:</b> %{x}<br><b>B-Axis:</b> %{y:.3f} mm<extra></extra>',
                legendgroup=f'depth{i}',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Add zero reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1.5, row=1, col=2)
    
    # Professional white theme layout
    fig.update_layout(
        title=dict(
            text='<b>IPI Displacement Time History</b>',
            font=dict(size=18, color='#1e293b', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top'
        ),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.15,
            xanchor='center',
            x=0.5,
            title=dict(text='<b>Depth:</b> ', font=dict(size=11, color='#374151')),
            bgcolor='#f8fafc',
            bordercolor='#cbd5e1',
            borderwidth=1,
            font=dict(size=10, color='#1e293b')
        ),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        hovermode='x unified',
        height=500,
        margin=dict(t=60, b=80, l=70, r=50)
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['y'] = 1.02
        annotation['font'] = dict(size=14, color='#1e293b', family='Arial, sans-serif')
    
    # Professional axis styling
    axis_style = dict(
        title_font=dict(size=12, color='#374151', family='Arial, sans-serif'),
        tickfont=dict(size=10, color='#4b5563'),
        gridcolor='#e5e7eb',
        gridwidth=1,
        linecolor='#d1d5db',
        linewidth=1,
        showline=True,
        mirror=True
    )
    
    fig.update_xaxes(title_text='Date/Time', **axis_style, row=1, col=1)
    fig.update_xaxes(title_text='Date/Time', **axis_style, row=1, col=2)
    fig.update_yaxes(
        title_text='Cumulative Displacement (mm)',
        zeroline=True,
        zerolinecolor='#9ca3af',
        zerolinewidth=1.5,
        **axis_style,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Cumulative Displacement (mm)',
        zeroline=True,
        zerolinecolor='#9ca3af',
        zerolinewidth=1.5,
        **axis_style,
        row=1, col=2
    )
    
    return fig


def create_trend_plot_resultant(
    processed_df: pd.DataFrame,
    selected_depths: list
) -> go.Figure:
    """
    Create resultant displacement trend plot.
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
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['cum_disp_resultant'],
            mode='lines+markers',
            name=f'{closest_depth:.1f}m',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate='<b>Time:</b> %{x}<br><b>Resultant:</b> %{y:.3f} mm<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=dict(
            text='<b>IPI Resultant Displacement Time History</b>',
            font=dict(size=18)
        ),
        xaxis=dict(title='Date/Time', gridcolor='lightgray'),
        yaxis=dict(
            title='Resultant Displacement (mm)',
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            title=dict(text='Depth:')
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
        display_mode = st.radio(
            "Display Mode",
            options=['A-Axis & B-Axis (Side by Side)', 'Resultant Only'],
            index=0,
            help="Show both axes side-by-side or resultant displacement only"
        )
        show_resultant = display_mode == 'Resultant Only'
    
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
            
            # Show data quality warning if rows were skipped
            if metadata.get('skipped_rows', 0) > 0:
                st.warning(f"⚠️ Data Quality: {metadata['skipped_rows']} rows had issues and were corrected/skipped. Total valid rows: {metadata['total_rows']}")
        else:
            # Generic CSV parsing with error handling
            try:
                df = pd.read_csv(
                    io.StringIO(file_content), 
                    na_values=['NAN', 'NaN', 'nan', ''],
                    on_bad_lines='skip'  # Skip malformed lines
                )
            except TypeError:
                # Older pandas version
                df = pd.read_csv(
                    io.StringIO(file_content), 
                    na_values=['NAN', 'NaN', 'nan', ''],
                    error_bad_lines=False
                )
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
        
        # Gauge Length Configuration
        st.subheader("📏 Gauge Length Configuration")
        
        num_sensors = detected_cols['num_sensors']
        
        # Initialize gauge lengths in session state if not exists
        if 'gauge_lengths' not in st.session_state or len(st.session_state.gauge_lengths) != num_sensors:
            st.session_state.gauge_lengths = [3.0] * num_sensors  # Default 3m for all sensors
        
        # Quick set options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Set All 1m", use_container_width=True):
                st.session_state.gauge_lengths = [1.0] * num_sensors
                st.rerun()
        with col2:
            if st.button("Set All 2m", use_container_width=True):
                st.session_state.gauge_lengths = [2.0] * num_sensors
                st.rerun()
        with col3:
            if st.button("Set All 3m", use_container_width=True):
                st.session_state.gauge_lengths = [3.0] * num_sensors
                st.rerun()
        with col4:
            show_gauge_config = st.checkbox("Edit Individual", value=False)
        
        # Show individual gauge length configuration
        if show_gauge_config:
            with st.expander("⚙️ Per-Sensor Gauge Length (click to expand)", expanded=True):
                st.caption("Configure gauge length for each sensor (1m, 2m, or 3m)")
                
                # Create columns for sensor configuration
                cols_per_row = 5
                for row_start in range(0, num_sensors, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for i, col in enumerate(cols):
                        sensor_idx = row_start + i
                        if sensor_idx < num_sensors:
                            with col:
                                new_val = st.selectbox(
                                    f"S{sensor_idx + 1}",
                                    options=[1.0, 2.0, 3.0],
                                    index=[1.0, 2.0, 3.0].index(st.session_state.gauge_lengths[sensor_idx]) if st.session_state.gauge_lengths[sensor_idx] in [1.0, 2.0, 3.0] else 2,
                                    key=f"gauge_{sensor_idx}",
                                    format_func=lambda x: f"{int(x)}m"
                                )
                                st.session_state.gauge_lengths[sensor_idx] = new_val
        
        # Display current gauge configuration summary
        gauge_summary = {}
        for gl in st.session_state.gauge_lengths:
            gauge_summary[f"{int(gl)}m"] = gauge_summary.get(f"{int(gl)}m", 0) + 1
        summary_text = " | ".join([f"{k}: {v} sensors" for k, v in sorted(gauge_summary.items())])
        st.caption(f"📊 Current config: {summary_text}")
        
        # Convert to numpy array for processing
        gauge_lengths = np.array(st.session_state.gauge_lengths)
        
        st.divider()
        
        # Base reading selection
        with st.sidebar:
            st.divider()
            st.subheader("4. Base Reading")
            
            # Get all timestamps and create a proper selection
            all_timestamps = df[detected_cols['timestamp']].sort_values().unique()
            
            # Create a date selector first to narrow down options
            available_dates = pd.to_datetime(all_timestamps).date
            unique_dates = sorted(list(set(available_dates)))
            
            # Date picker for base reading
            selected_date = st.date_input(
                "Select Base Date",
                value=unique_dates[0],
                min_value=unique_dates[0],
                max_value=unique_dates[-1],
                help="Select the date for base reading"
            )
            
            # Filter timestamps for the selected date
            timestamps_on_date = [ts for ts in all_timestamps 
                                  if pd.Timestamp(ts).date() == selected_date]
            
            if timestamps_on_date:
                # Show time options for selected date
                time_options = {pd.Timestamp(ts).strftime('%H:%M:%S'): ts 
                               for ts in timestamps_on_date}
                
                selected_time = st.selectbox(
                    "Select Base Time",
                    options=list(time_options.keys()),
                    index=0,
                    help="Select the time for base reading"
                )
                
                # Get the actual timestamp and find its index
                selected_timestamp = time_options[selected_time]
                base_reading_idx = df[df[detected_cols['timestamp']] == selected_timestamp].index[0]
                
                # Show selected base reading info
                st.caption(f"📌 Base: {pd.Timestamp(selected_timestamp).strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning("No data available for selected date")
                base_reading_idx = 0
        
        # Process data
        with st.spinner("Processing IPI data..."):
            processed_df = process_ipi_data(
                df,
                detected_cols,
                gauge_lengths,
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
                if show_resultant:
                    fig_profile = create_profile_plot_resultant(processed_df, selected_dates)
                else:
                    fig_profile = create_profile_plot_dual(processed_df, selected_dates)
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
                if show_resultant:
                    fig_trend = create_trend_plot_resultant(processed_df, selected_depths)
                else:
                    fig_trend = create_trend_plot_dual(processed_df, selected_depths)
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
