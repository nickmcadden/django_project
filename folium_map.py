import re
import folium
import pandas as pd
import numpy as np


def dms_string_to_decimal(dms_str):
    """
    Convert a string in DMS (Degrees-Minutes-Seconds) format to decimal degrees.
    example:
        
    lat = dms_string_to_decimal("51°28'38\"N")
    lon = dms_string_to_decimal("0°0'12\"W")

    Parameters:
    - dms_str: The string containing the DMS formatted coordinate.

    Returns:
    - Decimal representation of the DMS coordinate.
    """

    # Extracting values using regex
    match = re.match(r'(\d+)°(\d+)' + r"'([\d\.]+)\"([NSEW])", str(dms_str).upper())
    if not match:
        return np.nan

    degrees, minutes, seconds, direction = match.groups()
    degrees, minutes, seconds = float(degrees), float(minutes), float(seconds)

    decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)

    if direction in ['S', 'W']:
        decimal_degrees = -decimal_degrees

    return decimal_degrees


def read_wind_farm_data():
    wind_farms = pd.read_csv('uk_wind_farms.csv')
    print(wind_farms)
    # Splitting the coordinate column into latitude and longitude
    wind_farms[['latitude_dms', 'longitude_dms']] = wind_farms['Location'].str.split(' ', expand=True)

    print(wind_farms.dtypes)
    # Converting DMS string columns to decimal
    wind_farms['latitude'] = wind_farms['latitude_dms'].apply(dms_string_to_decimal)
    wind_farms['longitude'] = wind_farms['longitude_dms'].apply(dms_string_to_decimal)

    # Drop the intermediate columns if you want
    wind_farms = wind_farms.drop(columns=['latitude_dms', 'longitude_dms'])
    
    return wind_farms

# Wind farm map data 
wind_farms = read_wind_farm_data()

# Latitude and Longitude of a central point in the UK (around Birmingham)
uk_coords = [52.4862, -1.8904]

# Create a map centered around the UK
uk_map = folium.Map(location=uk_coords, 
                    zoom_start=5,
                    tiles=None,  
                    max_bounds=True)

# Add markets at each wind farm location
for i, row in wind_farms.iterrows():
    if row.latitude > 0 and row.Power > 0:
        print([row.latitude, row.longitude], row.Name)
        #folium.Marker( tooltip=row.Name, popup=row.Name).add_to(uk_map)
        folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=row.Power / 50,
                color="red",
                fill=True,
                fill_color="red",
                tooltip=f"Name: {row.Name}<br>Turbines: {row.Turbines}<br>Completion: {row.Completion}<br>Cost: {row.Cost}"
            ).add_to(uk_map)

        # Overlay your image
        img_bounds = [(48.9, -8.65), (60.6, 2.5)]  # Example bounds, adjust to fit your image

        folium.raster_layers.ImageOverlay(
            image="uk_map_overlay.jpg",
            bounds=img_bounds,
            opacity=0.5,
            interactive=True,                
            cross_origin=False,
            zindex=1,
            tiles=None,  # `tiles=None` to start with a blank slate
            max_bounds=True
        ).add_to(uk_map)
        
        # Restrict the viewable area to the image bounds and limit zoom
        uk_map.fit_bounds(img_bounds)
        uk_map.options['minZoom'] = 5  # Example, adjust as needed
        uk_map.options['maxZoom'] = 6  # Prevent zooming

# Save the map to an HTML file
uk_map.save('wind_farm_map.html')

