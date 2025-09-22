import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import osmnx as ox
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='osmnx')

# assess.py  (analysis + merging)
import pandas as pd

def merge_datasets(maize_df: pd.DataFrame, population_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(
        maize_df,
        population_df[['County', 'Total_Population19', 'LandArea', 'Population Density']],
        on='County',
        how='left'
    )
    merged_df['Yield_per_capita'] = merged_df['Production_Tons'] / (merged_df['Total_Population19'] + 1e-6)
    merged_df['Harvested_Area_per_capita'] = merged_df['Harvested_Area_Ha'] / (merged_df['Total_Population19'] + 1e-6)
    merged_df['Production_per_capita'] = merged_df['Production_Tons'] / (merged_df['Total_Population19'] + 1e-6)
    merged_df['Area_per_land'] = merged_df['Harvested_Area_Ha'] / (merged_df['LandArea'] + 1e-6)
    return merged_df.dropna(subset=['Total_Population19', 'LandArea'])

def plot_city_map(place_name, latitude, longitude, box_size_km=2, poi_tags=None):
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

    """
    Visualize geographic data on a map.
    
    Args:
        place_name (str): Name of the location
        latitude (float): Center latitude
        longitude (float): Center longitude
        box_size_km (float): Size of bounding box in km
        poi_tags (dict): OSM tags for points of interest ,fwi i abandonded dict :)
    """
    
    graph  = ox.graph_from_bbox(bbox)
    area  = ox.geocode_to_gdf(place_name)
    nodes, edges =ox.graph_to_gdfs(graph)
    buildings = ox.features_from_bbox(bbox, tags={"building": True})
    amenities = ox.features_from_bbox(bbox, tags={"amenity": True})
    shops = ox.features_from_bbox(bbox, tags={"shop": True})
    roads = ox.features_from_bbox(bbox, tags={"highway": True})
    natural = ox.features_from_bbox(bbox, tags={"natural": True})
    tourism = ox.features_from_bbox(bbox, tags={"tourism": True})


    fig, ax = plt.subplots(figsize=(6,6))
    area.plot(ax=ax, color="tan", alpha=0.5)
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor="gray", edgecolor="gray", alpha=0.6)
    if not amenities.empty:
        amenities.plot(ax=ax, color="cornsilk", markersize=5)
    if not shops.empty:
        shops.plot(ax=ax, color="purple", markersize=5)
    if not natural.empty:
        natural.plot(ax=ax,facecolor="lightgreen", edgecolor="gray", alpha=0.6)
    edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
    nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_title(place_name, fontsize=14)
    plt.show()




def get_osm_features(latitude, longitude, box_size_km=2, tags=None):
    """
    Get raw OSM data as a GeoDataFrame.
    
    Args:
        latitude (float): Center latitude
        longitude (float): Center longitude
        box_size_km (float): Size of bounding box in km
        tags (dict): OSM tags to filter features
        
    Returns:
        geopandas.GeoDataFrame: Raw OSM features
    """
    return get_osm_datapoints(latitude, longitude, box_size_km, tags)

def get_feature_vector(latitude, longitude, box_size_km=2, features=None):
    """
    Quantify geographic features into a numerical vector.
    
    Args:
        latitude (float): Center latitude
        longitude (float): Center longitude
        box_size_km (float): Size of bounding box in km
        features (list): List of feature types to count
        
    Returns:
        numpy.ndarray: Feature vector
    """
    if features is None:
        features = ['amenity', 'building', 'shop', 'tourism', 'leisure']
    
    # Get OSM data
    pois = get_osm_datapoints(latitude, longitude, box_size_km)
    
    # Create feature vector
    feature_vector = []
    
    for feature in features:
        # Count occurrences of each feature type
        if feature in pois.columns:
            count = pois[feature].notna().sum()
        else:
            count = 0
        feature_vector.append(count)
    
    # Add total count as additional feature
    feature_vector.append(len(pois))
    
    return np.array(feature_vector)

def visualize_feature_space(X, y, method='PCA'):
    """
    Visualize data distribution and separability using dimensionality reduction.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Labels
        method (str): Dimensionality reduction method ('PCA' or 't-SNE')
    """
    if method == 'PCA':
        reducer = PCA(n_components=2)
        title = "PCA Visualization"
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42)
        title = "t-SNE Visualization"
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")
    
    # Reduce dimensionality
    X_reduced = reducer.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot each class with different color
    unique_labels = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7)
    
    plt.title(title)
    plt.xlabel(f"{method} Component 1")
    plt.ylabel(f"{method} Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access

# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


"""
assess.py
Exploratory data analysis and plotting helpers.
Refactored plotting/EDA from the notebook.
"""

from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def describe_df(df: pd.DataFrame, columns: List[str] = None):
    """Print and return descriptive stats for selected columns (or all numeric columns)."""
    if columns is None:
        stats = df.describe(include='all')
    else:
        stats = df[columns].describe(include='all')
    print(stats)
    return stats

def plot_correlation_matrix(df: pd.DataFrame, features: List[str], figsize=(8,6)):
    """Plot correlation heatmap for provided features."""
    corr = df[features].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", square=True, cmap='coolwarm')
    plt.title("Feature correlation matrix")
    plt.show()

def plot_histograms(df: pd.DataFrame, columns: List[str], bins=30, figsize=(10,6)):
    """Plot histograms for a list of columns."""
    n = len(columns)
    cols = min(3, n)
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=figsize)
    for i, col in enumerate(columns, 1):
        plt.subplot(rows, cols, i)
        df[col].dropna().hist(bins=bins)
        plt.title(col)
    plt.tight_layout()
    plt.show()

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str = None, figsize=(7,5)):
    """Scatter plot of x vs y; optionally color by hue."""
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.title(f"{y} vs {x}")
    plt.show()

def plot_residuals(y_true, y_pred, bins=30):
    """Plot residual distribution and residuals vs predicted."""
    residuals = (y_true - y_pred)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(residuals, bins=bins, kde=True)
    plt.title("Residual distribution")
    plt.subplot(1,2,2)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.show()

def qq_plot(residuals):
    """QQ-plot for residuals against normal distribution."""
    import pylab, scipy.stats as stats
    stats.probplot(residuals, dist="norm", plot=pylab)
    pylab.show()

import pandas as pd

def clean_maize_data(maize_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean maize production data:
    - Convert numeric columns to appropriate dtypes
    - Drop rows with missing critical values
    """
    # Convert to numeric (coerce invalid values to NaN)
    maize_df['Harvested_Area_Ha'] = pd.to_numeric(maize_df['Harvested_Area_Ha'], errors='coerce')
    maize_df['Production_Tons'] = pd.to_numeric(maize_df['Production_Tons'], errors='coerce')
    maize_df['Yield_t_per_ha'] = pd.to_numeric(maize_df['Yield_t_per_ha'], errors='coerce')

    # Drop rows missing critical values
    maize_df = maize_df.dropna(subset=['Yield_t_per_ha', 'Harvested_Area_Ha', 'Production_Tons'])

    return maize_df


def merge_datasets(maize_df: pd.DataFrame, population_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge maize and population data:
    - Merge on County
    - Add per-capita and area-based features
    - Drop rows with missing population/land data
    """
    merged_df = pd.merge(
        maize_df,
        population_df[['County', 'Total_Population19', 'LandArea', 'Population Density']],
        on='County',
        how='left'
    )

    # Create derived features
    merged_df['Yield_per_capita'] = merged_df['Production_Tons'] / (merged_df['Total_Population19'] + 1e-6)
    merged_df['Harvested_Area_per_capita'] = merged_df['Harvested_Area_Ha'] / (merged_df['Total_Population19'] + 1e-6)
    merged_df['Production_per_capita'] = merged_df['Production_Tons'] / (merged_df['Total_Population19'] + 1e-6)
    merged_df['Area_per_land'] = merged_df['Harvested_Area_Ha'] / (merged_df['LandArea'] + 1e-6)

    # Drop incomplete population/land rows
    merged_df = merged_df.dropna(subset=['Total_Population19', 'LandArea'])

    return merged_df

import numpy as np
import pandas as pd

def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed versions of selected columns.
    Modifies df in place and returns it.
    """
    df['log_Harvested_Area_Ha'] = np.log1p(df['Harvested_Area_Ha'])
    df['log_Production_Tons'] = np.log1p(df['Production_Tons'])
    df['log_Total_Population19'] = np.log1p(df['Total_Population19'])
    df['log_LandArea'] = np.log1p(df['LandArea'])
    return df



