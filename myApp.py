import streamlit as st

st.title('House price prediction system')

with st.form('my_form'):
    st.subheader('**Information of your House**')

    # Input widgets
    locArea = st.text_input('How big is your house?')
    builtYear = st.text_input('What year was your house built?')
    firstF = st.text_input('What is the area of the first floor?')
    secondF = st.text_input('What is the area of the second floor?')
    bathR = st.text_input('How many bathrooms do you have in your house?')
    bedR = st.text_input('How many bedrooms do you have in your house?')
    totalR = st.text_input('What is the total number of rooms?')

    # Every form must have a submit button
    submitted = st.form_submit_button('Predict')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

data = pd.read_csv("data/train.csv")
features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
x = data[features]
y = data["SalePrice"]
x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, test_size = 0.2, random_state = 0)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(x_train, y_train)
rf_vla_pred = rf_model.predict(x_valid)
a = rf_model.predict([[locArea, builtYear, firstF,secondF,bathR,bedR,totalR]])

if submitted:
    st.markdown(f'''
        ☕ You have entered:
        - Local Area: `{locArea}`
        - Building year: `{builtYear}`
        - 1st Floor area: `{firstF}`
        - 2st Floor area: `{secondF}`
        - Num of bathroom: `{bathR}`
        - Num of bedroom: `{bedR}`
        - Total room: `{totalR}`
        - Price: `{a}`
        ''')
else:
    st.write('☝️ Place your enter!')


