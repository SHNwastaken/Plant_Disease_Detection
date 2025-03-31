import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Constants ---
# Define the path relative to the main script location
CSV_FILENAME = 'results/detection_results.csv'

# --- Helper Function for Map Color ---
def get_color(condition):
    """Assigns color based on condition string for the map."""
    condition_lower = str(condition).lower()
    if 'healthy' in condition_lower:
        return '#00C000' # Green
    elif 'error' in condition_lower or 'unknown' in condition_lower or 'n/a' in condition_lower:
        return '#808080' # Grey
    else:
        return '#FF0000' # Red

# --- Main Page Function ---
def display_results():
    st.set_page_config(layout="wide", page_title="Detection Results")
    st.title("📊 Detection Results Analysis")

    # --- Read CSV Data ---
    df_results = pd.DataFrame() # Initialize empty
    if os.path.exists(CSV_FILENAME):
        try:
            df_results = pd.read_csv(CSV_FILENAME, encoding='utf-8', on_bad_lines='warn')
            # Convert timestamp and confidence columns early
            if 'timestamp' in df_results.columns:
                 df_results['timestamp'] = pd.to_datetime(df_results['timestamp'], errors='coerce')
            if 'confidence' in df_results.columns:
                 df_results['confidence'] = pd.to_numeric(df_results['confidence'], errors='coerce')
            df_results.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed

        except pd.errors.EmptyDataError:
            st.info("Detection log file is empty.")
        except Exception as e:
            st.error(f"Error reading or processing CSV: {e}")
    else:
        st.warning(f"Detection log file not found at: {CSV_FILENAME}")
        st.info("Run the 'Live Feed' page first to generate detections.")
        st.stop() # Stop execution if no file exists

    if df_results.empty and os.path.exists(CSV_FILENAME):
         st.info("Detection log file exists but contains no valid data.")
         st.stop()
    elif df_results.empty:
         st.stop() # Already warned above if file doesn't exist

    # --- Display Options ---
    st.sidebar.header("Display Options")
    show_log = st.sidebar.checkbox("Show Full Detection Log", value=True)
    show_stats = st.sidebar.checkbox("Show Statistics", value=True)
    show_map = st.sidebar.checkbox("Show Detection Map", value=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear() # Clear pandas read_csv cache if used implicitly
        st.rerun()

    # --- Display Full Log ---
    if show_log:
        st.header("📜 Full Detection Log")
        st.dataframe(df_results, use_container_width=True)
        st.download_button(
             label="Download Log as CSV",
             data=df_results.to_csv(index=False).encode('utf-8'),
             file_name='detection_results_export.csv',
             mime='text/csv',
         )
        st.markdown("---")


    # --- Display Statistics ---
    if show_stats:
        st.header("📈 Statistics")
        col_stat1, col_stat2 = st.columns(2)

        with col_stat1:
            st.subheader("Overall Summary")
            st.metric("Total Detections Logged", len(df_results))
            if 'gps_valid' in df_results.columns:
                 try:
                     # Ensure boolean conversion handles strings robustly
                     df_results['gps_valid_bool'] = df_results['gps_valid'].astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)
                     valid_gps_count = df_results['gps_valid_bool'].sum()
                     st.metric("Detections with Valid GPS", f"{int(valid_gps_count)}")
                 except Exception as e:
                     st.caption(f"Could not calculate GPS validity: {e}")


            st.subheader("Plant Types Detected")
            plant_counts = df_results['plant_type'].value_counts()
            st.bar_chart(plant_counts)

            st.subheader("Confidence Score Distribution")
            if 'confidence' in df_results.columns and pd.api.types.is_numeric_dtype(df_results['confidence']):
                hist_values = np.histogram(df_results['confidence'].dropna(), bins=10, range=(0,1))[0]
                st.bar_chart(hist_values)
            else:
                st.caption("Confidence data not available or not numeric.")

        with col_stat2:
            st.subheader("Conditions Detected")
            condition_counts = df_results['condition'].value_counts()
            st.bar_chart(condition_counts)

            st.subheader("Detections Over Time (Daily)")
            if pd.api.types.is_datetime64_any_dtype(df_results['timestamp']):
                 df_results.set_index('timestamp', inplace=True)
                 detections_per_day = df_results.resample('D').size()
                 st.line_chart(detections_per_day)
                 df_results.reset_index(inplace=True) # Reset index
            else:
                 st.caption("Timestamp data not available or not in datetime format.")


            st.subheader("Condition Breakdown by Plant Type")
            try:
                condition_by_plant = df_results.groupby(['plant_type', 'condition']).size().unstack(fill_value=0)
                st.bar_chart(condition_by_plant) # Creates a stacked bar chart
            except Exception as e:
                st.caption(f"Could not generate condition breakdown: {e}")

        st.markdown("---")

    # --- Display Map ---
    if show_map:
        st.header("🗺️ Detection Locations Map")
        if 'latitude' in df_results.columns and 'longitude' in df_results.columns:
            map_df = df_results.copy()
            map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
            map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
            map_df.dropna(subset=['latitude', 'longitude'], inplace=True)
            map_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

            if not map_df.empty:
                st.info(f"Plotting {len(map_df)} detections with valid GPS coordinates.")
                map_df['color'] = map_df['condition'].apply(get_color)
                st.map(map_df, color='color', size=10, use_container_width=True)
                # Add a legend
                st.caption("Map Legend:")
                st.markdown("- <span style='color:#FF0000'>● Red:</span> Disease/Issue Detected", unsafe_allow_html=True)
                st.markdown("- <span style='color:#00C000'>● Green:</span> Healthy Detected", unsafe_allow_html=True)
                st.markdown("- <span style='color:#808080'>● Grey:</span> Error/Unknown/N/A", unsafe_allow_html=True)

            else:
                st.info("No valid GPS coordinates found in the detection log to display on the map.")
        else:
            st.warning("Required 'latitude' or 'longitude' columns not found in the detection log.")

# --- Run the App ---
if __name__ == "__main__":
    display_results()