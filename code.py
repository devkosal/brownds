import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
os.chdir('/Users/devsharma/Dropbox/Education/Data Science/Python Learning')

data =  pd.read_csv("data/citizen_props.csv")


#Distance to major cities
from math import radians, cos, sin, asin, sqrt

#Latitute then Longitute
cities_loc = {
        'Philadelphia':[39.9526, -75.165222],'Boston':[42.3601,-71.057083],'Pittsburg':[40.4406,-79.9959], 'Cambridge':[42.3736,-71.1097],'York':[39.9626,-76.7277], 'Reading':[40.3356,-75.9269],'Nantucket':[41.2835,-70.0995], 'Westechester':[41.1220,-73.7949], 'Providence':[41.8240,-71.4128]
}

def haversine(row,city):
    try:
        lat1, lon1, lat2, lon2 = map(radians, [row["geocode_latitude"], row["geocode_longitude"], cities_loc[city][0], cities_loc[city][1]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r * .621371
    except:
        return None

data["boston_dist"] = data.apply(lambda row: haversine(row,"Boston"),axis=1)
data["philly_dist"] = data.apply(lambda row: haversine(row,"Philadelphia"),axis=1)
data["pitt_dist"] = data.apply(lambda row: haversine(row,"Pittsburg"),axis=1)
data["prov_dist"] = data.apply(lambda row: haversine(row,"Providence"),axis=1)

# data.to_csv("data.csv")


# mapping clusters

dwelling = pd.read_csv("cluster_dwelling.csv")
city = pd.read_csv("cluster_city.csv")

dwel_dict = {}
city_dict = {}

for index, row in dwelling.iterrows():
    print(row["dwelling_type"],row["m_segments_upd"])
    dwel_dict[row["dwelling_type"]] = row["m_segments_upd"]

for index, row in city.iterrows():
    city_dict[row["prop_city"]] = row["m_segments_city"]


data["dwel_clust"] = data["dwelling_type"].map(dwel_dict)
data["city_clust"] = data["prop_city"].map(city_dict)


#Dates
from dateutil.parser import parse
data["date"] = data["transaction_date"].apply(lambda x: parse(x))


#Season

def season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    elif month in [9,10,11]:
        return "Fall"
    else:
        return None

# data.loc[0].date.month

data["season"] = data["date"].apply(lambda x: season(x.month))


# year built / trans time
data.year_built[data["year_built"]==0] = None

data["age_at_trans"] =  [x.year - y for x, y in zip(data.date, data.year_built)]


# data.apply(lambda row:


#recession dates

data["date_int"] = data["date"].apply(lambda x: int(x.strftime('%Y%m%d')))

# June 2007 to Feb 2012
def recession(date):
    """
    date in int format
    :param date:
    :return:
    """
    if date > 20070601 and date < 20120201:
        return 1
    else:
        return 0

data["recession"] = data["date_int"].apply(lambda x: recession(x))


#maping income
income = pd.read_csv("zip_income.csv")

zip_dict = {}

for index, row in income.iterrows():
    zip_dict[row["Zip_Code"]] = row["Median"]

data["med_income"] = data["prop_zip_code"].map(zip_dict)


data.to_csv("data.csv",index=False)

######### Removing Outliers ########

data2 = pd.read_csv("data.csv")

data = data2

# sns.heatmap(data.isna())

all_feats = ['fips_cd', 'apn', 'IsTraining', 'prop_house_number',
       'prop_house_number_2', 'prop_house_number_suffix',
       'prop_direction_left', 'prop_street_name', 'prop_suffix',
       'prop_direction_right', 'prop_unit_type', 'prop_unit_number',
       'prop_city', 'prop_state', 'prop_zip_code', 'prop_zip_plus_4',
       'dwelling_type', 'zoning', 'census_tract', 'mobile_home_ind',
       'timeshare_ind', 'acres', 'land_square_footage', 'irregular_lot_flg',
       'assessed_total_value', 'assessed_land_value',
       'assessed_improvement_value', 'market_total_value', 'market_land_value',
       'market_improvement_value', 'tax_amt', 'tax_year',
       'delinquent_tax_year', 'assessed_year', 'tax_cd_area',
       'building_square_feet', 'total_living_square_feet',
       'total_ground_floor_square_feet', 'total_basement_square_feet',
       'total_garage_parking_square_feet', 'year_built',
       'effective_year_built', 'bedrooms', 'total_rooms',
       'total_baths_calculated', 'air_conditioning', 'basement_cd',
       'condition', 'construction_type', 'fireplace_num', 'garage_type',
       'heating_type', 'construction_quality', 'roof_cover', 'roof_type',
       'stories_cd', 'style', 'geocode_latitude', 'geocode_longitude',
       'avm_final_value0', 'avm_std_deviation0', 'avm_final_value1',
       'avm_std_deviation1', 'avm_final_value2', 'avm_std_deviation2',
       'avm_final_value3', 'avm_std_deviation3', 'avm_final_value4',
       'avm_std_deviation4', 'first_mtg_amt', 'distressed_sale_flg',
       'sale_amt', 'transaction_date', 'boston_dist', 'philly_dist',
       'pitt_dist', 'prov_dist', 'dwel_clust', 'city_clust', 'season',
       'med_income', 'age_at_trans', 'date_int', 'recession']

my_feats = ['IsTraining',
       'prop_state',
       'land_square_footage',
       'assessed_total_value', 'assessed_land_value',
       'assessed_improvement_value', 'market_total_value', 'market_land_value',
       'market_improvement_value',
       'building_square_feet', 'total_living_square_feet',
       'total_garage_parking_square_feet',
       'bedrooms', 'total_rooms',
       'total_baths_calculated',
       'avm_final_value0', 'avm_std_deviation0', 'avm_final_value1',
       'avm_std_deviation1', 'avm_final_value2', 'avm_std_deviation2',
       'avm_final_value3', 'avm_std_deviation3', 'avm_final_value4',
       'avm_std_deviation4', 'first_mtg_amt',
       'sale_amt',  'boston_dist', 'philly_dist',
       'pitt_dist', 'prov_dist', 'dwel_clust', 'city_clust', 'season',
       'med_income', 'age_at_trans', 'date_int', 'recession']

my_feats2 = ['acres','land_square_footage','assessed_total_value','assessed_land_value','market_land_value','market_improvement_value','tax_amt','tax_year','delinquent_tax_year','assessed_year','building_square_feet','total_ground_floor_square_feet','total_basement_square_feet','total_garage_parking_square_feet','effective_year_built','bedrooms','total_rooms','total_baths_calculated','fireplace_num','philly_dist','pitt_dist','pitt_dist','dwel_clust','city_clust','date','season','med_income','recession', 'sale_amt','IsTraining']

# 'tax_year', 'geocode_latitude', 'geocode_longitude', 'acres', 'fireplace_num',        'construction_quality', 'roof_cover', 'roof_type',  'style',        'stories_cd',        'condition',
data = data[my_feats2]

for x in ['land_square_footage',
       'assessed_total_value', 'assessed_land_value',
       'assessed_improvement_value', 'market_total_value', 'market_land_value',
       'market_improvement_value',
       'building_square_feet', 'total_living_square_feet',
       'first_mtg_amt']:
    if x in data.columns:
        data[data[x]==0] = None

cats = ["prop_state","season"]

data = pd.get_dummies(data,drop_first=True)

train = data[(data.sale_amt>750) & (data.sale_amt<50000000) & (data.IsTraining == 1)]
test = data[(data.IsTraining == 0)]

cols = [col for col in train.columns if col not in ["IsTraining","sale_amt"]]
targetcol = ["sale_amt"]


import lightgbm as lgb
params = {"objective": "regression",
          "metric": "rmse",
          "num_leaves": 60,
          "max_depth": -1,
          "learning_rate": 0.01,
          "bagging_fraction": 0.9,  # subsample
          "feature_fraction": 0.9,  # colsample_bytree
          "bagging_freq": 5,  # subsample_freq
          "bagging_seed": 2018}

lgtrain, lgval = lgb.Dataset(train[cols], train[targetcol]), lgb.Dataset(test[cols], test[targetcol])
lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval], early_stopping_rounds=500, verbose_eval=200)



from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor()
gbm.fit(train[cols],train[targetcol])
pred = gbm.predict()
