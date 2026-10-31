import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd

from theme import setup
from src.common import load_data

# PAGE THEME
setup(
    "DATA & GOVERNANCE / 06",
    "Dataset Explorer",
    "Inspect, filter and understand the diabetes dataset."
)

# LOAD DATA
data = load_data()

if data is None:
    st.error("Dataset not found. Check data/diabetes.csv.")
    st.stop()

# DATASET OVERVIEW
st.markdown("## DATASET OVERVIEW")

c1, c2, c3 = st.columns(3)

c1.metric("Total Records", f"{len(data):,}")
c2.metric("Total Columns", len(data.columns))
c3.metric("Missing Values", int(data.isnull().sum().sum()))

st.markdown("---")

# FILTER OPTIONS
st.markdown("## FILTER DATA")

col1, col2 = st.columns(2)

with col1:
    outcome_filter = st.selectbox(
        "Filter by Outcome",
        ["All", "Negative (0)", "Positive (1)"]
    )

with col2:
    search = st.text_input(
        "Search records",
        placeholder="Enter a value to search..."
    )

filtered_data = data.copy()

if outcome_filter == "Negative (0)":
    filtered_data = filtered_data[
        filtered_data["Outcome"] == 0
    ]

elif outcome_filter == "Positive (1)":
    filtered_data = filtered_data[
        filtered_data["Outcome"] == 1
    ]

if search.strip():
    mask = filtered_data.astype(str).apply(
        lambda column: column.str.contains(
            search.strip(),
            case=False,
            na=False
        )
    ).any(axis=1)

    filtered_data = filtered_data[mask]

# ROW LIMIT
row_limit = st.slider(
    "Maximum rows to display",
    min_value=10,
    max_value=min(500, max(10, len(filtered_data))),
    value=min(100, max(10, len(filtered_data)))
)

st.markdown("---")

# DISPLAY DATA
st.markdown("## DATA RECORDS")

st.caption(
    f"Showing {min(row_limit, len(filtered_data))} "
    f"of {len(filtered_data)} matching records."
)

st.dataframe(
    filtered_data.head(row_limit),
    use_container_width=True,
    hide_index=True
)

# DOWNLOAD
csv_data = filtered_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Filtered Dataset",
    data=csv_data,
    file_name="sparkers_filtered_dataset.csv",
    mime="text/csv",
    use_container_width=True
)

# STATISTICAL SUMMARY
st.markdown("---")
st.markdown("## STATISTICAL SUMMARY")

st.dataframe(
    filtered_data.describe().T,
    use_container_width=True
)

# DATA QUALITY
st.markdown("---")
st.markdown("## DATA QUALITY REPORT")

quality = pd.DataFrame({
    "Column": data.columns,
    "Missing Values": [
        data[column].isnull().sum()
        for column in data.columns
    ],
    "Unique Values": [
        data[column].nunique()
        for column in data.columns
    ]
})

st.dataframe(
    quality,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "This explorer displays the source dataset. "
    "It does not modify the original CSV file."
)