

import osmnx as ox
import matplotlib.pyplot as plt
import warnings
import math
warnings.filterwarnings("ignore", category=FutureWarning, module='osmnx')
"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

def get_osm_datapoints(latitude, longitude, box_size_km=2, poi_tags=None):
    """
    Get OSM data points within a bounding box around given coordinates.
    
    Args:
        latitude (float): Center latitude
        longitude (float): Center longitude  
        box_size_km (float): Size of bounding box in km
        poi_tags (dict): OSM tags to filter points of interest
        
    Returns:
        geopandas.GeoDataFrame: OSM features within the bounding box
    """
       # Convert km to degrees (approx 1° = 111km)
    box_width = box_size_km / 111
    box_height = box_size_km / 111
    
    # Create bounding box
    north = latitude + box_height/2
    south = latitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    bbox = (west, south, east, north)
    
    if poi_tags is None:
        poi_tags = {
            "amenity": True,
            "building": True,
            "historic": True,
            "leisure": True,
            "shop": True,
            "tourism": True,
            "religion": True,
             "memorial": True,
            "aeroway": ["runway", "aerodrome"],
           "natural": True,
           "highway": True,
           "waterway": True,
           
        }
    

    # Download OSM data
    pois = ox.features_from_bbox(bbox, poi_tags)
    
    return pois
from typing import Any, Union
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None

"""
access.py
Data loading and preprocessing helpers for the maize yield notebook.
Refactored from the notebook into reusable functions so you can import them.
"""

from typing import Dict, Tuple
import os
import pandas as pd
import numpy as np

def _read_file(path: str) -> pd.DataFrame:
    """Read CSV or Excel into a dataframe (basic auto-detect)."""
    path = str(path)
    if path.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(path)
    else:
        return pd.read_csv(path)

def load_datafile_dict(data_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Given a dict of name -> path, read all files and return a dict of DataFrames.
    Example:
      data_files = {'maize': 'data/Maize-Production.xlsx', 'population': 'data/pop.csv'}
    """
    dfs = {}
    for key, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File for key '{key}' not found: {path}")
        dfs[key] = _read_file(path)
    return dfs

def melt_year_columns(df: pd.DataFrame, id_vars: list = None, year_column_name: str = 'Year') -> pd.DataFrame:
    """
    Transform a wide table with year columns (e.g., columns '2012','2013',...) into long format.
    Heuristics: treat any column whose name is a 4-digit year as a year column.
    """
    if id_vars is None:
        # assume non-year columns are id_vars
        id_vars = [c for c in df.columns if not (isinstance(c, str) and c.strip().isdigit() and len(c.strip()) == 4)]
    # detect year-like columns
    year_cols = [c for c in df.columns if isinstance(c, str) and c.strip().isdigit() and len(c.strip()) == 4]
    if not year_cols:
        # fallback: detect columns that look like '20xx' anywhere in the name
        year_cols = [c for c in df.columns if isinstance(c, str) and any(ch.isdigit() for ch in c)]
    # melt
    melted = pd.melt(df, id_vars=id_vars, value_vars=year_cols, var_name=year_column_name, value_name='value')
    # coerce year to int where possible
    try:
        melted[year_column_name] = melted[year_column_name].astype(int)
    except Exception:
        pass
    return melted

def clean_numeric_columns(df: pd.DataFrame, columns: list):
    """
    Attempt to coerce listed columns to numeric; strips commas and non-numeric chars.
    Returns the modified DataFrame.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col].replace(['', 'nan', 'None'], np.nan), errors='coerce')
    return df

def compute_yield(df: pd.DataFrame, production_col: str, area_col: str, yield_col: str = 'Yield_t_per_ha'):
    """
    Compute yield = production / area. Attempt to handle units gracefully.
    By default expects production in tonnes and area in hectares so result is t/ha.
    """
    if production_col not in df.columns or area_col not in df.columns:
        raise KeyError('production or area column not found in DataFrame')
    df = df.copy()
    # avoid division by zero
    df[area_col] = df[area_col].replace({0: np.nan})
    df[yield_col] = df[production_col] / df[area_col]
    return df

def merge_on_common_keys(base_df: pd.DataFrame, other_df: pd.DataFrame, on: list = None, how: str='left'):
    """
    Simple wrapper for pd.merge with a safety check.
    """
    if on is None:
        # try automatic common columns
        on = list(set(base_df.columns).intersection(set(other_df.columns)))
        if not on:
            raise ValueError("No common columns to merge on; please provide 'on' list.")
    return base_df.merge(other_df, on=on, how=how)

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add commonly used targets/features seen in the notebook:
      - log transforms of yield and production
      - year-over-year growth per County
      - z-score of yield per year (or globally)
      - low_yield_flag (below 25th percentile)
      - anomalies relative to county mean
    This function expects columns: 'County', 'Year', 'Yield_t_per_ha', 'Total_Production' (optional)
    """
    df = df.copy()
    # log transforms (handle non-positive values)
    if 'Yield_t_per_ha' in df.columns:
        df['target_log_yield'] = np.where(df['Yield_t_per_ha'] > 0, np.log(df['Yield_t_per_ha']), np.nan)
        df['target_yield_t_per_ha'] = df['Yield_t_per_ha']

    if 'Total_Production' in df.columns:
        df['target_log_production'] = np.where(df['Total_Production'] > 0, np.log(df['Total_Production']), np.nan)
        df['target_total_production'] = df['Total_Production']

    # county-year mean and anomaly
    if {'County', 'Year'}.issubset(df.columns) and 'Yield_t_per_ha' in df.columns:
        county_mean = df.groupby('County')['Yield_t_per_ha'].transform('mean')
        df['target_yield_anomaly'] = df['Yield_t_per_ha'] - county_mean
        df['target_yield_zscore'] = (df['Yield_t_per_ha'] - county_mean) / df['Yield_t_per_ha'].std(ddof=0)

        # year-over-year growth
        df = df.sort_values(['County', 'Year'])
        df['target_yield_growth'] = df.groupby('County')['Yield_t_per_ha'].pct_change()

        # low yield flag
        low_threshold = df['Yield_t_per_ha'].quantile(0.25)
        df['target_low_yield_flag'] = (df['Yield_t_per_ha'] < low_threshold).astype(int)

    return df

# helper that loads maize + population and returns merged df
def load_and_prepare_all(data_files: Dict[str,str], production_col_guess='Production', area_col_guess='Area'):
    """
    Convenience wrapper that attempts to load known datasets and produce a finished dataframe.
    data_files should be a dict containing at least keys 'maize' and optionally 'population'.
    """
    dfs = load_datafile_dict(data_files)
    maize = dfs.get('maize')
    if maize is None:
        raise KeyError("data_files must include 'maize' key pointing to the maize file path.")
    # try to detect production/area columns
    prod_cols = [c for c in maize.columns if production_col_guess.lower() in c.lower() or 'production' in c.lower()]
    area_cols = [c for c in maize.columns if area_col_guess.lower() in c.lower() or 'area' in c.lower()]
    # if the file has year columns, melt them
    melted = melt_year_columns(maize)
    # rename common columns if present
    if 'County' not in melted.columns:
        # heuristics: find a column that looks like a county/name
        for candidate in ['County', 'county', 'District', 'Region', 'CountyName']:
            if candidate in maize.columns:
                melted = melted.rename(columns={candidate: 'County'})
                break
    # attempt to map production/area/value columns
    # prefer explicit columns if present; otherwise assume 'value' is production
    if 'value' in melted.columns and not prod_cols:
        melted = melted.rename(columns={'value': 'Total_Production'})
    elif prod_cols:
        melted = melted.rename(columns={prod_cols[0]: 'Total_Production'})

    if area_cols:
        melted = melted.rename(columns={area_cols[0]: 'Area'})
    # coerce numeric
    melted = clean_numeric_columns(melted, ['Total_Production', 'Area'])
    # compute yield
    if 'Total_Production' in melted.columns and 'Area' in melted.columns:
        melted = compute_yield(melted, 'Total_Production', 'Area')

    # merge population if available
    if 'population' in dfs:
        pop = dfs['population']
        # try to rename/pop columns to Year and County
        if 'Year' not in pop.columns and 'year' in pop.columns:
            pop = pop.rename(columns={'year': 'Year'})
        if 'County' not in pop.columns:
            for c in pop.columns:
                if 'county' in c.lower():
                    pop = pop.rename(columns={c: 'County'})
                    break
        try:
            merged = merge_on_common_keys(melted, pop, on=['County', 'Year'], how='left')
        except Exception:
            merged = melted
    else:
        merged = melted

    final = basic_feature_engineering(merged)
    return final






# access.py
import pandas as pd

def load_maize_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def load_population_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

import pandas as pd

# Load data with better handling of Excel structure
def load_maize_data(file_path):
    """Load and preprocess maize production data"""
    maize_df = pd.read_excel(file_path)

    # Identify year columns and their positions
    year_columns = {}
    current_year = None

    for idx, col in enumerate(maize_df.columns):
        cell_value = str(maize_df.iloc[0, idx])
        if cell_value.isdigit():
            current_year = int(cell_value)
            year_columns[current_year] = idx
        elif current_year is not None:
            # This column belongs to the current year
            pass

    # Create a new structured dataframe
    structured_data = []

    # Process each county row
    for row_idx in range(2, len(maize_df)):
        county = maize_df.iloc[row_idx, 0]
        if pd.isna(county):
            continue

        for year, col_idx in year_columns.items():
            # Extract data for this year
            area_col = col_idx + 1
            production_col = col_idx + 2
            yield_col = col_idx + 3

            # Check if we have these columns
            if yield_col >= len(maize_df.columns):
                continue

            area = maize_df.iloc[row_idx, area_col]
            production = maize_df.iloc[row_idx, production_col]
            yield_val = maize_df.iloc[row_idx, yield_col]

            # Only add if we have valid data
            if pd.notna(area) and pd.notna(production) and pd.notna(yield_val):
                structured_data.append({
                    'County': county,
                    'Year': year,
                    'Harvested_Area_Ha': area,
                    'Production_Tons': production,
                    'Yield_t_per_ha': yield_val
                })

    return pd.DataFrame(structured_data)

# Load data with improved function
maize_df = load_maize_data('Maize-Production-2012-2020-Combined.xlsx')

# Load population data with proper cleaning
def load_population_data(file_path):
    """Load and clean population data"""
    population_df = pd.read_csv(file_path)

    # Clean numeric columns (remove commas and convert to numeric)
    numeric_columns = ['Total_Population19', 'Male populatio 2019',
                       'Female population 2019', 'Households', 'LandArea',
                       'Population Density', 'Population in 2009', 'Pop_change']

    for col in numeric_columns:
        if col in population_df.columns:
            population_df[col] = population_df[col].astype(str).str.replace(',', '').str.replace(' ', '')
            # Convert to numeric, coercing errors to NaN
            population_df[col] = pd.to_numeric(population_df[col], errors='coerce')

    # Add year column for merging
    population_df['Year'] = 2019

    return population_df

population_df = load_population_data('2019-population_census-report-per-county.csv')


import pandas as pd

# Clean maize data
def clean_maize_data(maize_df):
    """Clean maize production data"""
    # Convert to appropriate data types
    maize_df['Harvested_Area_Ha'] = pd.to_numeric(maize_df['Harvested_Area_Ha'], errors='coerce')
    maize_df['Production_Tons'] = pd.to_numeric(maize_df['Production_Tons'], errors='coerce')
    maize_df['Yield_t_per_ha'] = pd.to_numeric(maize_df['Yield_t_per_ha'], errors='coerce')

    # Remove rows with missing critical values
    maize_df = maize_df.dropna(subset=['Yield_t_per_ha', 'Harvested_Area_Ha', 'Production_Tons'])

    return maize_df

# Merge data with improved handling
def merge_datasets(maize_df, population_df):
    """Merge maize and population data"""
    # Merge on County and Year
    merged_df = pd.merge(
        maize_df,
        population_df[['County', 'Total_Population19', 'LandArea', 'Population Density']],
        on='County',
        how='left'
    )

    # Create new features
    merged_df['Yield_per_capita'] = merged_df['Production_Tons'] / (merged_df['Total_Population19'] + 1e-6)
    merged_df['Harvested_Area_per_capita'] = merged_df['Harvested_Area_Ha'] / (merged_df['Total_Population19'] + 1e-6)
    merged_df['Production_per_capita'] = merged_df['Production_Tons'] / (merged_df['Total_Population19'] + 1e-6)
    merged_df['Area_per_land'] = merged_df['Harvested_Area_Ha'] / (merged_df['LandArea'] + 1e-6)

    # Drop rows with missing population data
    merged_df = merged_df.dropna(subset=['Total_Population19', 'LandArea'])

    return merged_df


import pandas as pd

def load_maize_production(path: str) -> pd.DataFrame:
    """
    Load maize production data from an Excel file.

    Parameters
    ----------
    path : str
        Path to the Excel file containing maize production data.

    Returns
    -------
    pd.DataFrame
        DataFrame with county names and yearly production values.
    """
    # Load maize production data
    df = pd.read_excel(path, skiprows=1, header=[0, 1])

    # Rename the county column
    df = df.rename(columns={'Unnamed: 0_level_0': 'County'})

    return df

import pandas as pd

def restructure_maize_data(maize_production_df):
    """
    Restructure the maize production dataset by unpivoting yearly yield columns
    into a tidy long-format DataFrame.

    Parameters
    ----------
    maize_production_df : pd.DataFrame
        Original maize production DataFrame with county info and yearly yield columns.

    Returns
    -------
    pd.DataFrame
        Cleaned and restructured DataFrame with columns:
        ['County', 'Year', 'Yield (MT/HA)']
    """
    # Identify the County column name (multi-index assumed: ('County', 'COUNTY'))
    county_col = ('County', 'COUNTY')

    # Identify the columns containing the yearly yield data
    yield_cols = [col for col in maize_production_df.columns if col[1] == 'Yield (MT/HA)']

    # Create an empty list to store the unpivoted data
    unpivoted_data = []

    # Iterate through each row in the original DataFrame
    for _, row in maize_production_df.iterrows():
        county = row[county_col]
        # Iterate through each yearly yield column
        for col in yield_cols:
            year = col[0]
            yield_value = row[col]
            unpivoted_data.append({'County': county, 'Year': year, 'Yield (MT/HA)': yield_value})

    # Create a new DataFrame from the unpivoted data
    maize_yield = pd.DataFrame(unpivoted_data)

    # Convert Year to numeric and clean invalid years
    maize_yield['Year'] = pd.to_numeric(maize_yield['Year'], errors='coerce')
    maize_yield.dropna(subset=['Year'], inplace=True)
    maize_yield['Year'] = maize_yield['Year'].astype(int)

    # Convert Yield to numeric and clean invalid values
    maize_yield['Yield (MT/HA)'] = pd.to_numeric(maize_yield['Yield (MT/HA)'], errors='coerce')
    maize_yield.dropna(subset=['Yield (MT/HA)'], inplace=True)

    return maize_yield


import geopandas as gpd

def load_kenya_map(geojson_url=None):
    """
    Load the Kenya counties GeoJSON as a GeoDataFrame.
    """
    if geojson_url is None:
        geojson_url = "https://open.africa/dataset/a8f8b195-aafd-449b-9b1a-ab337fd9925f/resource/4fb2e27e-c001-4b7f-b71d-4fee4a96a0f8/download/kenyan-counties.geojson"

    kenya_map = gpd.read_file(geojson_url)

    # Standardize county names
    if "COUNTY" in kenya_map.columns:
        kenya_map["County"] = kenya_map["COUNTY"].str.strip().str.title()

    return kenya_map


