## Airbnb Revenue Estimator - City of Buenos Aires

Final application: In progress

### Purpose of project

In this project I try to find out the characteristics that influence property prices and Airbnb prices. The end product is an application where users can find out how much revenue they can make renting out a property through Airbnb, and how this revenue compares to the total value of the property.

This is a personal project for self-learning purposes. The aim of this project was to carry out a data project from A to Z, using multiple data sources. I specifically wanted to learn to work with dash and plotly to deploy my models and create interactive graphs.

### Datasets

I have used the following 4 datasets:

- **Airbnb dataset**. This dataset contains information on 24.134 Airbnb listings and can be found here: http://insideairbnb.com/
- **Properati dataset**. This dataset contains information on 1.000.000 properties from the real estate website Properati: https://www.properati.com.ar
- **Metro stations dataset**: This dataset contains geodata of all the metro stations in the city of Buenos Aires: https://data.buenosaires.gob.ar/dataset/subte-estaciones
- **Neighbourhood dataset***: This file contains geodata on all neighbourhoods in the city of Buenos Aires. See the csv file 'barrios' in this repository. 

### Main findings

Tree-based models gave the best result for both predicting property prices as well as Airbnb prices.

It was quite difficult to predict the prices on Airbnb, the best result I got was an R2 of 0.554 on the test set with a CatBoost Regressor. The most important features in this model are:

- ***Cleaning fee*** (positive effect): the higher the cleaning fee, the higher the predicted price.
- ***Accommodates*** (positive effect): the more guests an airbnb listing can accomodate, the higher the predicted price.
- ***Private room*** (negative effect): if the airbnb listing is a private room, the predicted price will be lower.
- ***Reviews per month*** (negative effect): the lower the amount of reviews per month, the higher the predicted price.
- ***Latitude*** (positive effect): the bigger the latitude (the more north the property location), the higher the predicted price.

For predicting the total property prices, I choose to use a XGBoost Regressor. With this model I reached an R2 of 0.953 on the test set. The most important features are:

- **Covered surface** (positive effect): the bigger the covered surface, the higher the price.
- **Total surface** (positive effect): the bigger the total surface, the higher the price.
- **Latitude** (positive effect): the bigger the latitude (the more north the property location), the higer the price.

**An important side note**: the data from Airbnb is from June 2020 and the data from Properati is from June 2019 until July 2020. Due to Covid-19, 2020 has not been a regular year. Especially the tourism sector has been impacted a lot and this might have led to a different offer on Airbnb than other years. Therefore, my models might not be fully generalizable and the data should be updated once the situation normalizes again.

### Libraries

pandas: 1.1.3
<br>
numpy: 1.19.1
<br>
matplotlib: 3.3.1
<br>
seaborn: 0.11.0
<br>
geopandas: 0.6.1
<br>
scikit-learn: 0.23.2
<br>
plotly: 4.11.0
<br>
dash: 1.16.3
