import math
import osmnx as ox

def get_feature_vector(latitude, longitude, box_size_km=2, features=None):
    """
    Given a central point (latitude, longitude) and a bounding box size,
    query OpenStreetMap via OSMnx and return a feature vector.

    This function represents the "Access" part of the pipeline, as it is
    responsible for querying and retrieving the raw geospatial data.

    Parameters
    ----------
    latitude : float
        Latitude of the center point.
    longitude : float
        Longitude of the center point.
    box_size_km : float
        Size of the bounding box in kilometers
    features : list of tuples
        List of (key, value) pairs to count. Example:
        [("amenity", "school"), ("shop", None)]

    Returns
    -------
    feature_vector : GeoDataFrame
        A GeoDataFrame of the retrieved features.
    """
    # Approximate degrees per kilometer at the given latitude
    degrees_per_km_lat = 1 / 111.0
    degrees_per_km_lon = 1 / (111.0 * math.cos(math.radians(latitude)))

    box_width_deg = box_size_km * degrees_per_km_lon
    box_height_deg = box_size_km * degrees_per_km_lat

    north = latitude + box_height_deg / 2
    south = latitude - box_height_deg / 2
    west = longitude - box_width_deg / 2
    east = longitude + box_width_deg / 2
    bbox = (west, south, east, north)

    # Create tags dictionary from the features list
    tags = {}
    for key, value in features:
        if key not in tags:
            tags[key] = True
        if value and isinstance(tags.get(key), bool):
            tags[key] = [value]
        elif value and isinstance(tags.get(key), list):
            tags[key].append(value)
        elif value:
            tags[key] = [value]

    try:
        pois = ox.features_from_bbox(bbox, tags)
    except Exception as e:
        print(f"Error fetching features for ({latitude}, {longitude}): {e}")
        pois = None

    return pois

if __name__ == "__main__":
    # Example usage for the access function
    features_list = [
        ("amenity", "school"),
        ("amenity", "hospital"),
        ("leisure", "park"),
    ]
    sample_latitude = -1.2921  # Nairobi, Kenya
    sample_longitude = 36.8219
    sample_pois = get_feature_vector(
        sample_latitude,
        sample_longitude,
        box_size_km=5,
        features=features_list
    )
    if sample_pois is not None:
        print(f"Retrieved {len(sample_pois)} features from OpenStreetMap.")
