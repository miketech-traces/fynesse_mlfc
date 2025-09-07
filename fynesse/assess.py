import pandas as pd
import matplotlib.pyplot as plt
from access import get_feature_vector
import osmnx as ox

# Define the features to query (used in both access and assess)
FEATURES = [
    ("building", None),
    ("amenity", None),
    ("amenity", "school"),
    ("amenity", "hospital"),
    ("amenity", "restaurant"),
    ("amenity", "cafe"),
    ("shop", None),
    ("tourism", None),
    ("tourism", "hotel"),
    ("tourism", "museum"),
    ("leisure", None),
    ("leisure", "park"),
    ("historic", None),
    ("amenity", "place_of_worship"),
]

# Your provided city dictionaries
CITIES_KENYA = {
    "Nyeri, Kenya": {"latitude": -0.4371, "longitude": 36.9580},
    "Nairobi, Kenya": {"latitude": -1.2921, "longitude": 36.8219},
    "Mombasa, Kenya": {"latitude": -4.0435, "longitude": 39.6682},
    "Kisumu, Kenya": {"latitude": -0.0917, "longitude": 34.7680}
}

CITIES_ENGLAND = {
    "Cambridge, England": {"latitude": 52.2053, "longitude": 0.1218},
    "London, England": {"latitude": 51.5072, "longitude": -0.1276},
    "Sheffield, England": {"latitude": 53.3811, "longitude": -1.4701},
    "Oxford, England": {"latitude": 51.7520, "longitude": -1.2577},
}

def get_counts(pois):
    """
    Counts features from a GeoDataFrame and returns a feature vector dictionary.
    This is part of the "Assess" step.
    """
    poi_counts = {}
    if pois is not None:
        for key, value in FEATURES:
            feature_name = f"{key}:{value}" if value else key
            if key in pois.columns:
                if value:  # count only that value
                    poi_counts[feature_name] = (pois[key] == value).sum()
                else:  # count any non-null entry
                    poi_counts[feature_name] = pois[key].notnull().sum()
            else:
                poi_counts[feature_name] = 0
    else:
        for key, value in FEATURES:
            feature_name = f"{key}:{value}" if value else key
            poi_counts[feature_name] = 0
    return poi_counts

def prepare_data(df):
    """
    Prepares the data by fetching geospatial features and creating a DataFrame.
    This function represents the main "Assess" part of the pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with cities, latitudes, and longitudes.

    Returns
    -------
    X : pandas.DataFrame
        DataFrame of feature vectors.
    y : pandas.Series
        Series of labels (countries).
    """
    feature_vectors = []
    y = []

    for city, row in df.iterrows():
        print(f"Assessing features for {city}...")
        pois = get_feature_vector(
            latitude=row["latitude"],
            longitude=row["longitude"],
            features=FEATURES,
            box_size_km=5
        )
        vector = get_counts(pois)
        feature_vectors.append(vector)
        y.append(row["country"])

    X = pd.DataFrame(feature_vectors, index=df.index)
    y = pd.Series(y, index=df.index)
    return X, y
    
def plot_city_map(city_name, latitude, longitude, box_size_km):
    """
    Plots a street map of a given city.
    This function can be used for visual assessment of the area being queried.

    Parameters
    ----------
    city_name : str
        The name of the city to plot.
    latitude : float
        Latitude of the center point.
    longitude : float
        Longitude of the center point.
    box_size_km : float
        Size of the bounding box in kilometers.
    """
    print(f"Plotting street map for {city_name}...")
    try:
        G = ox.graph_from_point((latitude, longitude), dist=box_size_km * 1000, network_type="all")
        fig, ax = ox.plot_graph(G, show=False, close=True, bgcolor="#ffffff", edge_color="#333333")
        plt.show()
        print("Plotting complete.")
    except Exception as e:
        print(f"Error plotting map for {city_name}: {e}")

if __name__ == "__main__":
    # Combine and split city data for a demonstration
    cities_df_kenya = pd.DataFrame.from_dict(CITIES_KENYA, orient="index")
    cities_df_kenya["country"] = "Kenya"

    cities_df_england = pd.DataFrame.from_dict(CITIES_ENGLAND, orient="index")
    cities_df_england["country"] = "England"

    df_full = pd.concat([cities_df_kenya, cities_df_england])

    # Let's manually define a test/train split similar to the notebook
    train_cities = df_full.index[:4]  # First 4 cities for training
    test_cities = df_full.index[4:]   # Last 4 cities for testing

    df_train = df_full.loc[train_cities]
    df_test = df_full.loc[test_cities]

    print("--- Preparing Training Data ---")
    X_train, y_train = prepare_data(df_train)
    print("\nTraining Data (X_train) Shape:", X_train.shape)
    print("Training Labels (y_train) Shape:", y_train.shape)

    print("\n--- Preparing Test Data ---")
    X_test, y_test = prepare_data(df_test)
    print("\nTest Data (X_test) Shape:", X_test.shape)
    print("Test Labels (y_test) Shape:", y_test.shape)
    
    # Example usage of the new plot_city_map function
    print("\n" + "="*50 + "\n")
    plot_city_map(
        'Cambridge, England',
        CITIES_ENGLAND['Cambridge, England']['latitude'],
        CITIES_ENGLAND['Cambridge, England']['longitude'],
        2
    )
