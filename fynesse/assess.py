import matplotlib.pyplot as plt

def explore_data(pois):
    """
    Explore and assess the downloaded geospatial data
    """
    # Check data volume
    print(f"Number of POIs: {len(pois)}")
    
    # Display sample data
    print("\nFirst few rows of POIs data:")
    print(pois.head())
    
    return pois

def visualize_data(area, buildings, graph, bbox):
    """
    Visualize the geospatial data for assessment
    """
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 10))
    area.plot(ax=ax, facecolor='gray')
    buildings.plot(ax=ax, facecolor='red', alpha=0.3)
    ox.plot_graph(graph, ax=ax, node_size=0, edge_color='white', edge_linewidth=0.5)
    
    plt.title("Geospatial Data Visualization")
    plt.show()
    
    return fig, ax
