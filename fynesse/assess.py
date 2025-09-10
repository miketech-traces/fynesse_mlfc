import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import osmnx as ox
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='osmnx')


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
