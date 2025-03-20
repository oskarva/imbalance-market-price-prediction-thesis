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
            "vol {area} regulation up mwh cet h a",
            "vol {area} regulation down mwh cet h a",
            "con {area} intraday mwh/h cet h a",
            "con {area} heating % cet min15 s",
            "con {area} cooling % cet min15 s",
            "pro {area} tot mwh/h cet min15 a",
            "pri {area} spot €/mwh cet h a",
            "rdl {area} mwh/h cet min15 a",
        ],
        "y": [
            "pri {area} regulation up €/mwh cet h a",
            "pri {area} regulation down €/mwh cet h a",
        ],
        "X_to_forecast": {
            "vol {area} regulation up mwh cet h a"   : "vol {area} regulation up mwh cet h a",
            "vol {area} regulation down mwh cet h a" : "vol {area} regulation down mwh cet h a",
            "con {area} intraday mwh/h cet h a"      : "con {area} intraday mwh/h cet min15 f",
            "con {area} heating % cet min15 s"       : "con {area} heating % cet min15 s",
            "con {area} cooling % cet min15 s"       : "con {area} cooling % cet min15 s",
            "pro {area} tot mwh/h cet min15 a"       : "pro {area} tot mwh/h cet min15 s",
            "pri {area} spot €/mwh cet h a"          : "pri {area} spot €/mwh cet h a",
            "rdl {area} mwh/h cet min15 a"           : "rdl {area} mwh/h cet min15 a",
        }
    },

    "de": {
        "X": [
            "vol de imb sys mw cet min15 a", 
            "vol de imb up mw cet min15 a",  #no
            "vol de imb down mw cet min15 a", #no
            "con de intraday mwh/h cet min15 a", #no
            "con de heating % cet min15 s", #no
            "con de cooling % cet min15 s", #no
            "pro de tot mwh/h cet min15 a", #no
            "vol de cap imb up mfrr mw cet h a",
            "vol de cap imb down mfrr mw cet h a",
            "vol de cap imb up afrr mw cet h a",
            "vol de cap imb down afrr mw cet h a",
            "pri de spot €/mwh cet h a", #no
            "rdl de mwh/h cet min15 a", #no
            #"pri de cap imb fcr €/mw cet h a", #Not added simply because I do not have forecasts for it
            #"pri de imb stlmt €/mwh cet min15 a",
            
            #"exc de>se4 com mw cet h a",
            #"exc de>dk1 com mw cet h a",
            #"exc de>dk2 com mw cet h a",
            #"exc de>nl com mw cet h a",
            #"exc de>fr com mw cet h a",
            #"exc de>ch com mw cet h a",
            #"exc de>at com mw cet h a",
            #"exc de>cz com mw cet h a",
            #"exc de>pl com mw cet h a",
            #"exc de>se com mw cet h a",
            #"exc de>no2 com mw cet h a",
            #"exc de>be com mw cet h a",
            #"exc de>no com mw cet h a",
            #"exc de>dk com mw cet h a",
        ],
        "X_to_forecast": {
            "vol de imb sys mw cet min15 a"         : "vol de imb sys mw cet min15 n",
            "vol de imb up mw cet min15 a"          : "vol de imb up mw cet min15 a",
            "vol de imb down mw cet min15 a"        : "vol de imb down mw cet min15 a",
            "con de intraday mwh/h cet min15 a"     : "con de intraday mwh/h cet min15 f",
            "con de heating % cet min15 s"          : "con de heating % cet min15 s",
            "con de cooling % cet min15 s"          : "con de cooling % cet min15 s",
            "pro de tot mwh/h cet min15 a"          : "pro de tot mwh/h cet min15 s",
            "vol de cap imb up mfrr mw cet h a"     : "vol de cap imb up mfrr mw cet h a",
            "vol de cap imb down mfrr mw cet h a"   : "vol de cap imb down mfrr mw cet h a",
            "vol de cap imb up afrr mw cet h a"     : "vol de cap imb up afrr mw cet h a",
            "vol de cap imb down afrr mw cet h a"   : "vol de cap imb down afrr mw cet h a",
            "pri de spot €/mwh cet h a"             : "pri de spot €/mwh cet h a",
            "rdl de mwh/h cet min15 a"              : "rdl de mwh/h cet min15 a",
        },
        "mfrr": [
            "pri de imb up mfrr €/mwh cet min15 a",
            "pri de imb down mfrr €/mwh cet min15 a",
        ],
        "afrr": [
            "pri de imb up afrr €/mwh cet min15 a",
            "pri de imb down afrr €/mwh cet min15 a",
        ],
    },
}

def get_curve_lists(area:str, sub_areas=None):
    """Get lists of curves for specified area and sub-areas.
    
    Parameters:
    ----------
    area : str
        The area code (e.g., "no", "de")
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
        for key, value in X_f_sa.items():
            X_f_sa.pop(key)
            X_f_sa[key.replace(AREA_STRING, sa)] = value.replace(AREA_STRING, sa)
        
        new_collection = {
            "sub_area":sa,
            "X":X_sa,
            "y":y_sa,
            "X_to_forecast":X_f_sa,
        }
        collections.append(new_collection)
    
    return collections

