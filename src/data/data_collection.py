import volue_insight_timeseries 
import os
import pandas as pd
import matplotlib.pyplot as plt

session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_INI"))

basic_curves = [
"pri de imb up mfrr €/mwh cet min15 a",
"pri de imb down mfrr €/mwh cet min15 a",
"vol de imb sys mw cet min15 a",
"vol de imb up mw cet min15 a",
"vol de imb down mw cet min15 a",
"con de intraday mwh/h cet min15 a",
"con de heating % cet min15 s",
"con de cooling % cet min15 s",
"pro de tot mwh/h cet min15 a",
"vol de cap imb up mfrr mw cet h a",
"vol de cap imb down mfrr mw cet h a",
"vol de cap imb up afrr mw cet h a",
"vol de cap imb down afrr mw cet h a",
"pri de spot €/mwh cet h a",
"rdl de mwh/h cet min15 a",
"pri de imb stlmt €/mwh cet min15 a",
]

exchange_curves = [
    "exc de>se4 com mw cet h a",
    "exc de>dk1 com mw cet h a",
    "exc de>dk2 com mw cet h a",
    "exc de>nl com mw cet h a",
    "exc de>fr com mw cet h a",
    "exc de>ch com mw cet h a",
    "exc de>at com mw cet h a",
    "exc de>cz com mw cet h a",
    "exc de>pl com mw cet h a",
    "exc de>se com mw cet h a",
    "exc de>no2 com mw cet h a",
    "exc de>be com mw cet h a",
    "exc de>no com mw cet h a",
    "exc de>dk com mw cet h a"
]

combined = basic_curves + exchange_curves

start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp.today()

pandas_series = {}
for curve_name in combined:
    curve = session.get_curve(name=curve_name)
    ts = curve.get_data(data_from=start_date, data_to=end_date)
    s = ts.to_pandas()
    pandas_series[curve_name] = s