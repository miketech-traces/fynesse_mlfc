import osmnx as ox

def download_geospatial_data(place_name, latitude, longitude, box_width=0.1, box_height=0.1):
    """
    Download geospatial data from OpenStreetMap for a specified location
    """
    # Create bounding box
    north = latitude + box_height/2
    south = latitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    bbox = (west, south, east, north)

    # Define tags for points of interest
    tags = {
        "amenity": True,
        "buildings": True,
        "historic": True,
        "leisure": True,
        "shop": True,
        "tourism": True,
        "religion": True,
        "memorial": True
    }

    # Retrieve data
    pois = ox.features_from_bbox(bbox, tags)
    graph = ox.graph_from_bbox(bbox)
    area = ox.geocode_to_gdf(place_name)
    buildings = ox.features_from_bbox(bbox, tags={"building": True})

    return pois, graph, area, buildings, bbox
