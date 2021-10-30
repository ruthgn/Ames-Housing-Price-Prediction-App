import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

from pandas.api.types import CategoricalDtype
from category_encoders import MEstimateEncoder
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

st.write("""
# Ames House Price Prediction App
""")
st.caption("by [Ruth G. N.](https://www.linkedin.com/in/ruthgn/)")

st.write("""
         
This app predicts house price in Ames, Iowa—based on user-specified input.

Visit the project [notebook](https://www.kaggle.com/ruthgn/house-prices-top-8-featengineering-xgb-optuna) to learn about the model-building process or check out the complete [project repository](https://github.com/ruthgn/Ames-Housing-Price-Prediction) on GitHub. 

""")

st.info("""
The prediction model running on the app ranks in the top 8% of Kaggle's [House Price Prediction Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) leaderboard (as of 10/29/2021).
""")

st.write('---')


## Sidebar

### Header for user input section
st.sidebar.header('Specify House Variables:')

## Sidebar note
st.sidebar.markdown("""
For more information on each variable's interpretation and parameters, visit the [data dictionary](https://github.com/ruthgn/Ames-Housing-Price-Prediction/blob/main/data_description.txt).


[Example CSV input file](https://github.com/ruthgn/Ames-Housing-Price-Prediction/blob/main/sample_test.csv)
""")

### Listing features for user input

#### Setting input parameters (collecting user input)
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    input_data.index +=2920
    input_data = input_data.rename(columns={
    '1stFlrSF': 'FirstFlrSF',
    '2ndFlrSF': 'SecondFlrSF',
    '3SsnPorch': 'Threeseasonporch'
    })

else:

    test_data_param = pd.read_csv('test.csv', index_col='Id')
    X = test_data_param.rename(columns={
        '1stFlrSF': 'FirstFlrSF',
        '2ndFlrSF': 'SecondFlrSF',
        '3SsnPorch': 'Threeseasonporch'
        })

    def user_input_features():
        MSSubClass = st.sidebar.selectbox("Building class — `MSSubClass`", ('1-Story (1946 & newer, all styles)', '1-Story (1945 & older)', '1-Story (finished attic, all ages)', '1-1/2 Story (unfinished, all ages)', '1-1/2 Story (finished, all ages)', '2-Story (1946 & newer)', '2-Story (1945 & older)', '2-1/2 Story (all ages)', 'Split or Multi-level', 'Split foyer', 'Duplex (all styles/ages)', '1-Story PUD (1946 & newer)', '1-1/2 Story PUD (all ages)', '2-Story PUD (1946 & newer)', 'Multi-level PUD', '2 fam conversion (all styles/age)'))
        MSZoning = st.sidebar.selectbox("Zoning classification — `MSZoning`", ('Residential Low Density', 'Residential Medium Density', 'Residential High Density''Commercial (all)', 'Floating Village Residential'))
        LotFrontage = st.sidebar.slider("Linear feet of street connected to property", round(X.LotFrontage.min()), round(X.LotFrontage.max()), round(X.LotFrontage.median()))
        LotArea = st.sidebar.slider("Lot size in square feet", round(X.LotArea.min()), round(X.LotArea.max()), round(X.LotArea.median()))
        Street = st.sidebar.selectbox("Type of road access", ('Gravel', 'Paved'))
        Alley = st.sidebar.selectbox("Type of alley access", ('No alley access', 'Gravel', 'Paved'))
        LotShape = st.sidebar.selectbox("General shape of property", ('Regular', 'Slightly irregular', 'Moderately Irregular', 'Irregular'))
        LandContour = st.sidebar.selectbox("Flatness of the property", ('Near Flat/Level', 'Banked', 'Hillside', 'Depression')) 
        Utilities = st.sidebar.selectbox("Type of utilities available", ('All public Utilities (E,G,W,& S)', 'Electricity, Gas, and Water', 'Electricity and Gas Only', None))
        LotConfig = st.sidebar.selectbox("Lot configuration", ('Inside lot', 'Corner lot', 'Cul-de-sac', 'Frontage on 2 sides of property', 'Frontage on 3 sides of property'))
        LandSlope = st.sidebar.selectbox("Slope of property", ('Gentle slope', 'Moderate Slope', 'Severe Slope'))
        Neighborhood = st.sidebar.selectbox("Physical locations within Ames city limits", ('Bloomington Heights', 'Bluestem', 'Briardale', 'Brookside', 'Clear Creek', 'College Creek', 'Crawford', 'Edwards', 'Gilbert', 'Iowa DOT and Rail Road', 'Meadow Village', 'Mitchell', 'North Ames', 'Northridge', 'Northpark Villa', 'Northridge Heights', 'Northwest Ames', 'Old Town', 'South & West of Iowa State Univ', 'Sawyer', 'Sawyer West', 'Somerset', 'Stone Brook', 'Timberland', 'Veenker'))
        Condition1 = st.sidebar.selectbox("Proximity to main road or railroad — `Condition1`", ('Adjacent to arterial street', 'Adjacent to feeder street', 'Normal', "Within 200' of North-South Railroad", 'Adjacent to North-South Railroad', 'Near positive off-site feature', 'Adjacent to postive off-site feature', "Within 200' of East-West Railroad", 'Adjacent to East-West Railroad'))
        Condition2 = st.sidebar.selectbox("Proximity to main road or railroad (if a second is present) — `Condition2`", ('Adjacent to arterial street', 'Adjacent to feeder street', 'Normal', "Within 200' of North-South Railroad", 'Adjacent to North-South Railroad', 'Near positive off-site feature', 'Adjacent to postive off-site feature', "Within 200' of East-West Railroad", 'Adjacent to East-West Railroad'))
        BldgType = st.sidebar.selectbox("Type of dwelling", ('Single-family Detached', 'Two-family Conversion', 'Duplex', 'Townhouse End Unit', 'Townhouse Inside Unit'))
        HouseStyle = st.sidebar.selectbox("Style of dwelling", ('One story', 'One & 1/2 story: 2nd level finished', 'One & 1/2 story: 2nd level unfinished', 'Two story', 'Two & 1/2 story: 2nd level finished', 'Two & 1/2 story: 2nd level unfinished', 'Split Foyer', 'Split Level'))
        OverallQual = st.sidebar.slider("Overall material & finish quality of the house (9=Excellent, 5=Average, 1=Very Poor)", round(X.OverallQual.min()), round(X.OverallQual.max()), round(X.OverallQual.median()))
        OverallCond = st.sidebar.slider("Overall house condition rating (9=Excellent, 5=Average, 1=Very Poor)", round(X.OverallCond.min()), round(X.OverallCond.max()), round(X.OverallCond.median()))
        YearBuilt = st.sidebar.slider("Original construction date", round(X.YearBuilt.min()), round(X.YearBuilt.max()), round(X.YearBuilt.median()))
        YearRemodAdd = st.sidebar.slider("Remodel date", round(X.YearRemodAdd.min()), round(X.YearRemodAdd.max()), round(X.YearRemodAdd.median()))
        RoofStyle = st.sidebar.selectbox("Type of roof", ('Flat', 'Gable', 'Gabrel (Barn)', 'Hip', 'Mansard', 'Shed'))
        RoofMatl = st.sidebar.selectbox("Roof material", ('Clay or Tile', 'Standard (Composite) Shingle', 'Membrane', 'Metal', 'Roll', 'Gravel & Tar', 'Wood Shakes', 'Wood Shingles'))
        Exterior1st = st.sidebar.selectbox("Exterior covering on house", ('Asbestos Shingles', 'Asphalt Shingles', 'Brick Common', 'Brick Face', 'Cinder Block', 'Cement Board', 'Hard Board', 'Imitation Stucco', 'Metal Siding', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'Vinyl Siding', 'Wood Siding', 'Wood Shingles'))
        Exterior2nd = st.sidebar.selectbox("Exterior covering on house (if more than one material)", ('Asbestos Shingles', 'Asphalt Shingles', 'Brick Common', 'Brick Face', 'Cinder Block', 'Cement Board', 'Hard Board', 'Imitation Stucco', 'Metal Siding', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'Vinyl Siding', 'Wood Siding', 'Wood Shingles'))
        MasVnrType = st.sidebar.selectbox("Masonry veneer type — `MasVnrType`", ('Brick Common', 'Brick Face', 'Cinder Block', 'None', 'Stone'))
        MasVnrArea = st.sidebar.slider("Masonry veneer area in square feet — `MasVnrArea`", round(X.MasVnrArea.min()), round(X.MasVnrArea.max()), round(X.MasVnrArea.median()))
        ExterQual = st.sidebar.selectbox("Exterior material quality", ('Excellent', 'Good', 'Average/Typical', 'Fair', 'Poor'))
        ExterCond = st.sidebar.selectbox("Present condition of the material on the exterior", ('Excellent', 'Good', 'Average/Typical', 'Fair', 'Poor'))
        Foundation = st.sidebar.selectbox("Type of foundation", ('Brick & Tile', 'Cinder Block', 'Poured Contrete', 'Slab', 'Stone', 'Wood'))
        BsmtQual = st.sidebar.selectbox("Height of the basement — `BsmtQual`", ('Excellent (100+ inches)', 'Good (90-99 inches)', 'Typical (80-89 inches)', 'Fair (70-79 inches)', 'Poor (<70 inches', 'No Basement'))
        BsmtCond = st.sidebar.selectbox("General condition of the basement — `BsmtCond`", ('Excellent', 'Good', 'Typical', 'Fair', 'Poor', 'No Basement'))
        BsmtExposure = st.sidebar.selectbox("Walkout or garden level basement walls — `BsmtExposure`", ('Good Exposure', 'Average Exposure', 'Mimimum Exposure', 'No Exposure', 'No Basement'))
        BsmtFinType1 = st.sidebar.selectbox("Quality of basement finished area — `BsmtFinType1`", ('Good Living Quarters', 'Average Living Quarters', 'Below Average Living Quarters', 'Average Rec Room', 'Low Quality', 'Unfinshed', 'No Basement'))
        BsmtFinSF1 = st.sidebar.slider("Basement finished area no.1 square footage — `BsmtFinSF1`", round(X.BsmtFinSF1.min()), round(X.BsmtFinSF1.max()), round(X.BsmtFinSF1.median()))
        BsmtFinType2 = st.sidebar.selectbox("Quality of second finished area (if present) — `BsmtFinType2`", ('Good Living Quarters', 'Average Living Quarters', 'Below Average Living Quarters', 'Average Rec Room', 'Low Quality', 'Unfinshed', None))
        BsmtFinSF2 = st.sidebar.slider("Basement finished area no.2 (if present) square footage — `BsmtFinSF2`", round(X.BsmtFinSF2.min()), round(X.BsmtFinSF2.max()), round(X.BsmtFinSF2.median()))
        BsmtUnfSF = st.sidebar.slider("Unfinished square feet of basement area — `BsmtUnfSF`", round(X.BsmtUnfSF.min()), round(X.BsmtUnfSF.max()), round(X.BsmtUnfSF.median()))
        TotalBsmtSF = st.sidebar.slider("Total square feet of basement area — `TotalBsmtSF`", round(X.TotalBsmtSF.min()), round(X.TotalBsmtSF.max()), round(X.TotalBsmtSF.median()))
        Heating = st.sidebar.selectbox("Type of heating", ('Floor Furnace', 'Gas forced warm air furnace', 'Gas hot water or steam heat', 'Gravity furnace', 'Hot water or steam heat other than gas', 'Wall furnace'))
        HeatingQC = st.sidebar.selectbox("Heating quality and condition", ('Excellent', 'Good', 'Average/Typical', 'Fair', 'Poor'))
        CentralAir = st.sidebar.selectbox("Central air conditioning", ('Yes', 'No'))
        Electrical = st.sidebar.selectbox("Electrical system", ('Standard Circuit Breakers & Romex', '>60 AMP Fuse Box, Romex', '60 AMP Fuse Box, mostly Romex', '60 AMP Fuse Box, knob & tube', 'Mixed'))
        FirstFlrSF = st.sidebar.slider("First Floor square feet", round(X.FirstFlrSF.min()), round(X.FirstFlrSF.max()), round(X.FirstFlrSF.median()))
        SecondFlrSF = st.sidebar.slider("Second floor square feet", round(X.SecondFlrSF.min()), round(X.SecondFlrSF.max()), round(X.SecondFlrSF.median()))
        LowQualFinSF = st.sidebar.slider("Low quality finished square feet (all floors)", round(X.LowQualFinSF.min()), round(X.LowQualFinSF.max()), round(X.LowQualFinSF.median()))
        GrLivArea = st.sidebar.slider("Above (ground) living area square feet", round(X.GrLivArea.min()), round(X.GrLivArea.max()), round(X.GrLivArea.median()))
        BsmtFullBath = st.sidebar.slider("Basement full bathrooms", round(X.BsmtFullBath.min()), round(X.BsmtFullBath.max()), round(X.BsmtFullBath.median()))
        BsmtHalfBath = st.sidebar.slider("Basement half bathrooms", round(X.BsmtHalfBath.min()), round(X.BsmtHalfBath.max()), round(X.BsmtHalfBath.median()))
        FullBath = st.sidebar.slider("Full bathrooms above ground", round(X.FullBath.min()), round(X.FullBath.max()), round(X.FullBath.median()))
        HalfBath = st.sidebar.slider("Half baths above ground", round(X.HalfBath.min()), round(X.HalfBath.max()), round(X.HalfBath.median()))
        BedroomAbvGr = st.sidebar.slider("Number of bedrooms above basement level", round(X.BedroomAbvGr.min()), round(X.BedroomAbvGr.max()), round(X.BedroomAbvGr.median()))
        KitchenAbvGr = st.sidebar.slider("Number of kitchens", round(X.KitchenAbvGr.min()), round(X.KitchenAbvGr.max()), round(X.KitchenAbvGr.median()))
        KitchenQual = st.sidebar.selectbox("Kitchen quality", ('Excellent', 'Good', 'Typical/Average', 'Fair', 'Poor'))
        TotRmsAbvGrd = st.sidebar.slider("Total rooms above ground (does not include bathrooms)", round(X.TotRmsAbvGrd.min()), round(X.TotRmsAbvGrd.max()), round(X.TotRmsAbvGrd.median()))
        Functional = st.sidebar.selectbox("Home functionality rating", ('Typical Functionality', 'Minor Deductions 1', 'Minor Deductions 2', 'Moderate Deductions', 'Major Deductions 1', 'Major Deductions 2', 'Severely Damaged', 'Salvage only'))
        Fireplaces = st.sidebar.slider("Number of fireplaces", round(X.Fireplaces.min()), round(X.Fireplaces.max()), round(X.Fireplaces.median()))
        FireplaceQu = st.sidebar.selectbox("Fireplace quality", ('Excellent', 'Good', 'Average', 'Fair', 'Poor', 'No Fireplace'))
        GarageType = st.sidebar.selectbox("Garage location", ('More than one type of garage', 'Attached to home', 'Basement Garage', 'Built-In', 'Car Port', 'Detached from home', 'No Garage'))
        GarageYrBlt = st.sidebar.slider("Year garage was built", round(X.GarageYrBlt.min()), 2010, round(X.GarageYrBlt.median()))
        GarageFinish = st.sidebar.selectbox("Interior finish of the garage", ('Finished', 'Rough Finished', 'Unfinished', 'No Garage'))
        GarageCars = st.sidebar.slider("Size of garage in car capacity", round(X.GarageCars.min()), round(X.GarageCars.max()), round(X.GarageCars.median()))
        GarageArea = st.sidebar.slider("Size of garage in square feet", round(X.GarageArea.min()), round(X.GarageArea.max()), round(X.GarageArea.median()))
        GarageQual = st.sidebar.selectbox("Garage quality", ('Excellent', 'Good', 'Typical/Average', 'Fair', 'Poor', 'No Garage'))
        GarageCond = st.sidebar.selectbox("Garage condition", ('Excellent', 'Good', 'Typical/Average', 'Fair', 'Poor', 'No Garage'))
        PavedDrive = st.sidebar.selectbox("Paved driveway", ('Paved ', 'Partial Pavement', 'Dirt/Gravel'))
        WoodDeckSF = st.sidebar.slider("Wood deck area in square feet", round(X.WoodDeckSF.min()), round(X.WoodDeckSF.max()), round(X.WoodDeckSF.median()))
        OpenPorchSF = st.sidebar.slider("Open porch area in square feet", round(X.OpenPorchSF.min()), round(X.OpenPorchSF.max()), round(X.OpenPorchSF.median()))
        EnclosedPorch = st.sidebar.slider("Enclosed porch area in square feet", round(X.EnclosedPorch.min()), round(X.EnclosedPorch.max()), round(X.EnclosedPorch.median()))
        Threeseasonporch = st.sidebar.slider("Three season porch area in square feet", round(X.Threeseasonporch.min()), round(X.Threeseasonporch.max()), round(X.Threeseasonporch.median()))
        ScreenPorch = st.sidebar.slider("Screen porch area in square feet", round(X.ScreenPorch.min()), round(X.ScreenPorch.max()), round(X.ScreenPorch.median()))
        PoolArea = st.sidebar.slider("Pool area in square feet", round(X.PoolArea.min()), round(X.PoolArea.max()), round(X.PoolArea.median()))
        PoolQC = st.sidebar.selectbox("Pool quality", ('Excellent', 'Good', 'Average/Typical', 'Fair', 'No Pool'))
        Fence = st.sidebar.selectbox("Fence quality", ('Good Privacy', 'Minimum Privacy', 'Good Wood', 'Minimum Wood/Wire', 'No Fence'))
        MiscFeature = st.sidebar.selectbox("Miscellaneous feature not covered in other categories  — `MiscFeature`", (None, 'Elevator', '2nd Garage (if not described in garage section)', 'Other', 'Shed (over 100 SF)', 'Tennis Court'))
        MiscVal = st.sidebar.slider("Dollar value of miscellaneous feature — `MiscVal`", round(X.MiscVal.min()), round(X.MiscVal.max()), round(X.MiscVal.median()))
        MoSold = st.sidebar.selectbox("Month Sold", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        YrSold = st.sidebar.slider("Year Sold", round(X.YrSold.min()), round(X.YrSold.max()), round(X.YrSold.median()))
        SaleType = st.sidebar.selectbox("Type of sale", ('Warranty Deed - Conventional', 'Warranty Deed - Cash', 'Warranty Deed - VA Loan', 'Home just constructed and sold', 'Court Officer Deed/Estate', 'Contract -15% down pay, reg terms', 'Contract-Low down pay & int%', 'Contract-Low int%', 'Contract-Low down pay', 'Other'))
        SaleCondition = st.sidebar.selectbox("Condition of sale", ('Normal Sale', 'Abnormal Sale', 'Adjoining Land Purchase', 'Allocation', 'Sale between family members', 'Home was not completed'))
        data = {"MSSubClass": MSSubClass,
                "MSZoning": MSZoning,
                "LotFrontage": LotFrontage,
                "LotArea": LotArea,
                "Street": Street,
                "Alley": Alley,
                "LotShape": LotShape,
                "LandContour": LandContour,
                "Utilities": Utilities,
                "LotConfig": LotConfig,
                "LandSlope": LandSlope,
                "Neighborhood": Neighborhood,
                "Condition1": Condition1,
                "Condition2": Condition2,
                "BldgType": BldgType,
                "HouseStyle": HouseStyle,
                "OverallQual": OverallQual,
                "OverallCond": OverallCond,
                "YearBuilt": YearBuilt,
                "YearRemodAdd": YearRemodAdd,
                "RoofStyle": RoofStyle,
                "RoofMatl": RoofMatl,
                "Exterior1st": Exterior1st,
                "Exterior2nd": Exterior2nd,
                "MasVnrType": MasVnrType,
                "MasVnrArea": MasVnrArea,
                "ExterQual": ExterQual,
                "ExterCond": ExterCond,
                "Foundation": Foundation,
                "BsmtQual": BsmtQual,
                "BsmtCond": BsmtCond,
                "BsmtExposure": BsmtExposure,
                "BsmtFinType1": BsmtFinType1,
                "BsmtFinSF1": BsmtFinSF1,
                "BsmtFinType2": BsmtFinType2,
                "BsmtFinSF2": BsmtFinSF2,
                "BsmtUnfSF": BsmtUnfSF,
                "TotalBsmtSF": TotalBsmtSF,
                "Heating": Heating,
                "HeatingQC": HeatingQC,
                "CentralAir": CentralAir,
                "Electrical": Electrical,
                "FirstFlrSF": FirstFlrSF,
                "SecondFlrSF": SecondFlrSF,
                "LowQualFinSF": LowQualFinSF,
                "GrLivArea": GrLivArea,
                "BsmtFullBath": BsmtFullBath,
                "BsmtHalfBath": BsmtHalfBath,
                "FullBath": FullBath,
                "HalfBath": HalfBath,
                "BedroomAbvGr": BedroomAbvGr,
                "KitchenAbvGr": KitchenAbvGr,
                "KitchenQual": KitchenQual,
                "TotRmsAbvGrd": TotRmsAbvGrd,
                "Functional": Functional,
                "Fireplaces": Fireplaces,
                "FireplaceQu": FireplaceQu,
                "GarageType": GarageType,
                "GarageYrBlt": GarageYrBlt,
                "GarageFinish": GarageFinish,
                "GarageCars": GarageCars,
                "GarageArea": GarageArea,
                "GarageQual": GarageQual,
                "GarageCond": GarageCond,
                "PavedDrive": PavedDrive,
                "WoodDeckSF": WoodDeckSF,
                "OpenPorchSF": OpenPorchSF,
                "EnclosedPorch": EnclosedPorch,
                "Threeseasonporch": Threeseasonporch,
                "ScreenPorch": ScreenPorch,
                "PoolArea": PoolArea,
                "PoolQC": PoolQC,
                "Fence": Fence,
                "MiscFeature": MiscFeature,
                "MiscVal": MiscVal,
                "MoSold": MoSold,
                "YrSold": YrSold,
                "SaleType": SaleType,
                "SaleCondition": SaleCondition}
        features = pd.DataFrame(data, index=[0])
        
        # Restoring original names of levels within categorical variables for processing
        level_dictionary_data = pd.read_csv('level_dictionary.csv')
        var_list = list(level_dictionary_data['col'].unique())

        for x in var_list:
            dict_section = level_dictionary_data[level_dictionary_data['col'] == x].copy()
            orig_level_name = list(dict_section['level_label'].unique())
            new_level_name = list(dict_section['level_meaning'].unique())     
            my_dictionary = dict(zip(new_level_name, orig_level_name))
            for col in list(features.columns):
                if x==col:
                    features[col].replace(my_dictionary, inplace=True)

        return features

    input_data = user_input_features()

input_data.head()


# Loading training and test(input) data
def load_data():
    df_train = pd.read_csv('train.csv', index_col='Id')
    df_test = pd.read_csv('test.csv', index_col='Id')
    df_input = input_data
    # To match df_train and df_test and df_input before temporary merge
    df_input.columns = df_test.columns
    # Merge the splits so we can process them together
    df = pd.concat([df_train, df_test, df_input])
    # Preprocessing steps
    df = clean(df)
    df = encode(df)
    df = impute_plus(df)
    # Reform splits
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]
    df_input = df.drop(index=list(df_train.index))
    df_input = df_input.drop(index=list(df_test.index))
    return df_train, df_test, df_input

## Data Preprocessing

### Clean data
def clean(df):
    # Correct typo on Exterior2nd
    df['Exterior2nd'] = df['Exterior2nd'].replace({'Brk Cmn': 'BrkComm'})
    # Some values of GarageYrBlt are corrupt, so we'll replace them with the year house was built
    df['GarageYrBlt'] = df['GarageYrBlt'].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    # Name beginning with numbers are awkward to work with
    df.rename(columns={
        '1stFlrSF': 'FirstFlrSF',
        '2ndFlrSF': 'SecondFlrSF',
        '3SsnPorch': 'Threeseasonporch'
        }, inplace=True)
    return df

### Encode the Statistical Data Type

##### The nominative (unordered) categorical features
features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", 
                "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", 
                "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", 
                "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature", 
                "SaleType", "SaleCondition"]

##### The ordinal (ordered) categorical features 

#Pandas calls the categories "levels"
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))

ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

#Add a None level for missing values
ordered_levels = {key: ["None"] + value for key, value in
                  ordered_levels.items()}

def encode(df):
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace=True)
    # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels,
                                                    ordered=True))
    return df

### Handle missing values
def impute_plus(df):
    # Get names of columns with missing values
    cols_with_missing = [col for col in df.columns if col != 'SalePrice' and df[col].isnull().any()]
    for col in cols_with_missing:
        df[col + '_was_missing'] = df[col].isnull()
        df[col + '_was_missing'] = (df[col + '_was_missing']) * 1
    # Impute 0 for missing numeric values
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    # Impute "None" for missing categorical values
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df



#Finally load the pre-processed data


df_train, df_test, df_input = load_data() 



# Feature Engineering

## Drop low-scoring features
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]

## Label Encode
def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    return X

## Create ratio features with mathematical transforms
def mathematical_transforms(df):
    X = pd.DataFrame() # Just a dataframe to hold new features
    X['LivLotRatio'] = df.GrLivArea / df.LotArea
    X['Spaciousness'] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    return X

## Create count features
def counts(df):
    X = pd.DataFrame()
    X['PorchTypes'] = df[['WoodDeckSF',
                        'OpenPorchSF',
                        'EnclosedPorch',
                        'Threeseasonporch',
                        'ScreenPorch'
                        ]].gt(0.0).sum(axis=1)
    X['TotalHalfBath'] = df.BsmtFullBath + df.BsmtHalfBath
    X['TotalRoom'] = df.TotRmsAbvGrd + df.FullBath + df.HalfBath
    return X

## Create group transform features
def group_transforms(df):
    X = pd.DataFrame()
    X['MedNhbdArea'] = df.groupby('Neighborhood')['GrLivArea'].transform('median')
    return X

## Create PCA-inspired features
pca_features = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]

def pca_inspired(df):
    X = pd.DataFrame()
    X["GrLivAreaPlusBsmtSF"] = df.GrLivArea + df.TotalBsmtSF
    X["RecentRemodLargeBsmt"] = df.YearRemodAdd * df.TotalBsmtSF
    return X

## Target Encoding

class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded



# Create final feature set

def create_features(df_train, df_test, df_input):
    X_train = df_train.copy()
    y_train = X_train.pop('SalePrice')
    mi_scores = make_mi_scores(X_train, y_train)
    # Combine splits if test data is given
    #
    # If we're creating features for test set predictions, we should
    # use all the data we have available. After creating our features,
    # we'll recreate the splits.
    X_test = df_test.copy()
    X_test.pop("SalePrice")

    X_input = df_input.copy()
    X_input.pop("SalePrice")

    X = pd.concat([X_train, X_test, X_input])

    # Step 1: Drop features with low Mutual Information scores
    X = drop_uninformative(X, mi_scores)

    # Step 2: Add features from mathematical transforms 
    ######## (`LivLotRatio`, `Spaciousness`)
    X = X.join(mathematical_transforms(X))

    # Step 4: Add new feature from counts 
    ######## (`PorchTypes`, `TotalHalfBath`, `TotalRoom`)
    X = X.join(counts(X))

    # Step 5: Add new feature from group transform
    ######## (median `GrLivArea` by `neighborhood`)
    X = X.join(group_transforms(X))

    # Step 7: Add features from PCA
    ######## (loadings-inspired features , PCA components, & outlier indicators)
    X = X.join(pca_inspired(X))
  
    # Label encoding for the categorical features
    X = label_encode(X)

    # Reform splits
    X_train = X.loc[X_train.index, :]
    X_test = X.loc[X_test.index, :]
    X_input = X.drop(index=list(df_train.index))
    X_input = X_input.drop(index=list(df_test.index))

    # Step 8: Target Encoder
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X_train = X_train.join(encoder.fit_transform(X_train, y_train, cols=["MSSubClass"]))
    X_test = X_test.join(encoder.transform(X_test))
    X_input = X_input.join(encoder.transform(X_input))

    return X_train, X_test, X_input

# Finally create dataframes with added features
X_train, X_test, X_input = create_features(df_train, df_test, df_input)

# Reads in saved XGBRegression model
load_model = pickle.load(open('ames_house_xgb_model.pkl', 'rb'))

# Apply model to make predictions
prediction = np.exp(load_model.predict(X_input))



# Main Panel

st.header('Predicted House Price')


## Show predicted home value


if len(list(prediction)) == 1:
    single_prediction = '$ {:,.2f}'.format(float(prediction))
    st.subheader(single_prediction)
    

else:
    pd.options.display.float_format = '${:,.2f}'.format
    array_prediction = pd.DataFrame(data=prediction)
    array_prediction.index +=1
    array_prediction.columns = ["Value"]
    array_prediction["Value"] = (array_prediction["Value"]).apply(lambda x: '${:,.2f}'.format(x))
    
    st.table(array_prediction)

st.write('---')

## Explaining the model's predictions using SHAP values
explainer = shap.TreeExplainer(load_model)
shap_values = explainer.shap_values(X_input)

#(Hide Streamlit pyplot deprecation warning)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('Variable Importance')

st.caption('Which variables did the model think are most important in determining sale price?')
plt.title('Feature importance based on SHAP values')
plot1 = shap.summary_plot(shap_values, X_input, plot_type="bar")
st.pyplot(plot1, bbox_inches='tight')
st.write('---')

st.caption('How does each variable affect the house price?')
plt.title('Variable Impact on Predicted House Value')
plot2 = shap.summary_plot(shap_values, X_input)
st.pyplot(plot2, bbox_inches='tight')
st.write('---')

st.caption("""

Data obtained from the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) by Dean De Cock.

""")
