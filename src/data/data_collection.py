import volue_insight_timeseries 
import os

print(os.environ.get("WAPI_INI"))

session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_INI"))
cats = session.get_categories()
print("printing all data categories:")
for cat in cats:
    print(cat.get("name"), cat.get("key"))