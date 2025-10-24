def get_wildlife_crossing_query(
    extent_country: str, timeout: int = 240, bridges_only: bool = False
) -> str:
    """Generate an Overpass API query to fetch wildlife crossings within a specified country extent.

    Args:
        extent_country (str): The country code to define the extent. (e.g., "DE" for Germany)
        timeout (int): The timeout for the Overpass API request in seconds. Default is 240.

    Returns:
        str: The generated Overpass API query string.
    """
    bridge_filter = '["bridge"="yes"]' if bridges_only else ""
    query = f"""
    [out:json][timeout:{timeout}];
    area["ISO3166-1"="{extent_country}"]->.searchArea;
    (
        nwr(area.searchArea)["man_made"="wildlife_crossing"]{bridge_filter};
    );
    out body geom;
    """
    return query


def get_street_query(
    extent_country: str, street_type_depth: int = 2, timeout: int = 240
) -> str:
    """Generate an Overpass API query to fetch motorways within a specified country extent.

    Args:
        extent_country (str): The country code to define the extent. (e.g., "DE" for Germany)
        street_type_depth (int): The depth of street types to include. Default is 2. Find more information at https://wiki.openstreetmap.org/wiki/Key:highway.
        timeout (int): The timeout for the Overpass API request in seconds. Default is 240.

    Returns:
        str: The generated Overpass API query string.
    """
    query = f"""
    [out:json][timeout:{timeout}];
    area["ISO3166-1"="{extent_country}"]->.searchArea;
    (
        way(area.searchArea){get_street_filter_string(street_type_depth)};
    );
    out body geom;
    >;
    out skel qt;
    """
    return query


def get_street_filter_string(street_type_depth: int) -> str:
    """Generate a filter string for street types based on the specified depth.

    Args:
        street_type_depth (int): The depth of street types to include. Find more information at https://wiki.openstreetmap.org/wiki/Key:highway.

    Returns:
        str: The generated filter string for street types.
    """
    street_types = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "living_street",
        "pedestrian",
        "track",
        "path",
    ]
    if street_type_depth < 1 or street_type_depth > len(street_types):
        raise ValueError(f"street_type_depth must be between 1 and {len(street_types)}")
    selected_street_types = street_types[:street_type_depth]
    filter_string = '["highway"~"' + "|".join(selected_street_types) + '"]'
    return filter_string


if __name__ == "__main__":
    # Example usage
    query = get_street_query("DE", street_type_depth=2, timeout=180)
    print(query)
