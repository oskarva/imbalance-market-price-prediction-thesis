curve_collections = {
    "no": {
        "sub_areas": [
            "no1", 
            "no2",
            "no3",
            "no4",
            "no5",
            ],
        "X": [
            #"vol {area} regulation up mwh cet h a",
            #"vol {area} regulation down mwh cet h a",
            "con {area} intraday mwh/h cet h a", # consumption 
            "con {area} heating % cet min15 s", # need for heat
            "con {area} cooling % cet min15 s", # need for cooling
            "pro {area} tot mwh/h cet min15 a", # total production
            "pri {area} spot €/mwh cet h a", # spotprice
            "rdl {area} mwh/h cet min15 a", 
        ],
        "y": [
            "pri {area} regulation up €/mwh cet min15 a",
            "pri {area} regulation down €/mwh cet min15 a",
        ],
        "X_to_forecast": {
            #"vol {area} regulation up mwh cet h a"   : "vol {area} regulation up mwh cet h a",
            #"vol {area} regulation down mwh cet h a" : "vol {area} regulation down mwh cet h a",
            "con {area} intraday mwh/h cet h a"      : "con {area} intraday mwh/h cet min15 f",
            "con {area} heating % cet min15 s"       : "con {area} heating % cet min15 s",
            "con {area} cooling % cet min15 s"       : "con {area} cooling % cet min15 s",
            "pro {area} tot mwh/h cet min15 a"       : "pro {area} tot mwh/h cet min15 s",
            "pri {area} spot €/mwh cet h a"          : "pri {area} spot €/mwh cet h a",
            "rdl {area} mwh/h cet min15 a"           : "rdl {area} mwh/h cet min15 a",
        }
    },

    
}

def get_curve_dicts(area:str, sub_areas=None):
    """Get lists of curves for specified area and sub-areas.
    
    Parameters:
    ----------
    area : str
        The area code (e.g., "no")
    sub_areas : list of str, optional
        List of sub-areas. If None, uses all sub-areas defined for the area.
        Only applicable for areas with defined sub-areas (e.g., "no").
        
    Returns:
    ----------
    list of dict
        Each dict contains:
        - sub_area: str - The sub-area code
        - X: list of str - Input curve names
        - y: list of str - Target curve names (varies by area)
        - X_to_forecast: dict - Mapping of actual to forecast curves
    """
    AREA_STRING = "{area}"
    collections = []

    area = area.lower()
    if sub_areas is None:
        sub_areas = curve_collections[area]["sub_areas"]
    
    for sa in sub_areas:
        X_sa = curve_collections[area]["X"].copy()
        for i in range(len(X_sa)):
            X_sa[i] = X_sa[i].replace(AREA_STRING, sa)

        y_sa = curve_collections[area]["y"].copy()
        for i in range(len(y_sa)):
            y_sa[i] = y_sa[i].replace(AREA_STRING, sa)
        
        X_f_sa = curve_collections[area]["X_to_forecast"].copy()
        X_f_sa_new = {}  # Create a new dictionary
        for key, value in X_f_sa.items():
            new_key = key.replace(AREA_STRING, sa)
            new_value = value.replace(AREA_STRING, sa)
            X_f_sa_new[new_key] = new_value

        # Replace the old dictionary with the new one
        X_f_sa = X_f_sa_new
        
        new_collection = {
            "sub_area":sa,
            "X":X_sa,
            "y":y_sa,
            "X_to_forecast":X_f_sa,
        }
        collections.append(new_collection)
    
    return collections
