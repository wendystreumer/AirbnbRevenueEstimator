# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:10:22 2020

@author: Wendy
"""
# https://plotly.com/python/plotly-express/

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pickle
from dash.dependencies import Input, Output, State
from math import radians, cos, sin, asin, sqrt
import dash_bootstrap_components as dbc
import unidecode

# Importing the datasets and models.
with open('airbnb_reduced.pkl', 'rb') as f_df_airbnb:
    airbnb = pickle.load(f_df_airbnb)
    
with open('properati_reduced.pkl', 'rb') as f_df_properati:
    properati = pickle.load(f_df_properati)
    
with open('airbnb_model_catboost.pkl', 'rb') as f_model_airbnb:
    airbnb_model = pickle.load(f_model_airbnb)
    
with open('properati_model_XGB.pkl', 'rb') as f_model_airbnb:
    properati_model = pickle.load(f_model_airbnb)
    
subtes = pd.read_csv('estaciones-de-subte.csv', encoding='utf8')
subte_D = subtes[subtes.linea=='D'].copy()

# Defining functions to calculate the distance between a certain location and
# the nearest metro station (one of the features in the models).
    
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def calculate_min_distance_subte(lon, lat):
    distances = []
    for index_subte, id_subte in enumerate(subtes.id):
        distances.append(haversine(lon, lat,\
                                   subtes.long.iloc[index_subte],\
                                   subtes.lat.iloc[index_subte]))
    min_distance = min(distances)
    return min_distance

def calculate_min_distance_subteD(lon, lat):
    distances = []
    for index_subte, id_subte in enumerate(subte_D.id):
        distances.append(haversine(lon, lat,\
                                   subte_D.long.iloc[index_subte],\
                                   subte_D.lat.iloc[index_subte]))
    min_distance = min(distances)
    return min_distance

# Creating empty dataframes that will be filled later with the input obtained
# in the application. These dataframes will be the input for the models.
    
columns_airbnb = ['latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms',
                    'beds', 'cleaning_fee', 'minimum_nights', 'maximum_nights',
                    'reviews_per_month', 'tv', 'wifi', 'air_conditioning', 'heating',
                    'hot_water', 'kitchen', 'pool', 'elevator', 'gym',
                    'private_entrance', 'balcony/patio', 'bbq', 'min_dist_to_subte',
                    'min_dist_to_subteD', 'Balvanera', 'Colegiales', 'Constitución',
                    'Monserrat', 'Palermo', 'Puerto Madero', 'Recoleta', 'Retiro',
                    'Saavedra', 'San Cristóbal', 'San Nicolás', 'San Telmo',
                    'Villa Urquiza', 'Private room', 'Shared room']

columns_properati = ['lat', 'lon', 'rooms', 'bedrooms', 'bathrooms', 'surface_total',
                    'surface_covered', 'pool', 'parking', 'gym', 'balcony/patio', 'garden',
                    'terrace', 'bbq', 'sum', 'min_dist_to_subte', 'min_dist_to_subteD',
                    'ALMAGRO', 'BALVANERA', 'BELGRANO', 'BOCA', 'CABALLITO', 'CHACARITA',
                    'COGHLAN', 'CONSTITUCION', 'MONSERRAT', 'NUEVA POMPEYA', 'PALERMO',
                    'PARQUE CHACABUCO', 'PATERNAL', 'PUERTO MADERO', 'RECOLETA', 'RETIRO',
                    'SAAVEDRA', 'SAN CRISTOBAL', 'SAN NICOLAS', 'VILLA DEVOTO',
                    'VILLA LUGANO']

airbnb_predict = pd.DataFrame(index=[0], columns= columns_airbnb)
properati_predict = pd.DataFrame(index=[0], columns= columns_properati)

# Creating lists with neighbourhoods, property types, room types and amenities.
    
neighbourhoods = airbnb.neighbourhood.unique().tolist()
neighbourhoods.sort()

property_types = ['Apartment', 'House']
room_types = ['Entire home/apt', 'Private room', 'Shared room']

amenities_all = ['tv', 'wifi', 'air_conditioning', 'heating', 'hot_water',
                 'kitchen', 'parking', 'pool', 'elevator', 'gym',
                 'private_entrance', 'balcony/patio', 'garden', 'terrace', 
                 'bbq', 'sum']
amenities_airbnb = ['tv', 'wifi', 'air_conditioning', 'heating',
                    'hot_water', 'kitchen', 'pool', 'elevator', 'gym',
                    'private_entrance', 'balcony/patio', 'bbq']
amenities_properati = ['pool', 'parking', 'gym', 'balcony/patio', 'garden',
                       'terrace', 'bbq', 'sum']

# Setting up the application and creating the different select and text boxes
# needed to obtain the input for the models.

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI])
server = app.server

select_neighbourhood = html.Div(
                                [html.Label('Neighbourhood:'),
                                 dbc.Select(id="Neighbourhood",
                                            options=[{
                                                'label': i,
                                                'value': i
                                                } for i in neighbourhoods],
                                            value='Almagro'
                                            )])

select_property_type = html.Div(
                                [html.Label('Property type:'),
                                 dbc.Select(id="Property_type",
                                            options=[{
                                                'label': i,
                                                'value': i
                                                } for i in property_types],
                                            value='Apartment'
                                            )])

select_room_type = html.Div(
                                [html.Label('Room type:'),
                                 dbc.Select(id="Room_type",
                                            options=[{
                                                'label': i,
                                                'value': i
                                                } for i in room_types],
                                            value='Entire home/apt'
                                            )])

select_latitude = html.Div([
                                html.Label("Latitude:"),
                                dbc.Input(id="Latitude",
                                          value=-34.606, 
                                          type="number",
                                          min=-34.725, 
                                          max=-34.525,
                                          step=0.000000000001),
                                ])

select_longitude = html.Div([
                                html.Label("Longitude:"),
                                dbc.Input(id="Longitude",
                                          value=-58.422, 
                                          type="number",
                                          min=-58.550, 
                                          max=-58.325,
                                          step=0.000000000001),
                                ])

select_covered_surface = html.Div([
                                html.Label("Covered surface in m2:"),
                                dbc.Input(id="Covered_surface",
                                          value=50, 
                                          type="number"),
                                ])

select_total_surface = html.Div([
                                html.Label("Total surface in m2:"),
                                dbc.Input(id="Total_surface",
                                          value=50, 
                                          type="number"),
                                ])

select_accommodates = html.Div([
                                html.Label("Accommodates:"),
                                dbc.Input(id="Accommodates",
                                          value=2, 
                                          type="number"),
                                ])

select_rooms = html.Div([
                                html.Label("Total rooms:"),
                                dbc.Input(id="Rooms",
                                          value=2, 
                                          type="number",
                                          ),
                                ])

select_bedrooms = html.Div([
                                html.Label("Bedrooms:"),
                                dbc.Input(id="Bedrooms",
                                          value=1, 
                                          type="number",
                                          ),
                                ])

select_bathrooms = html.Div([
                                html.Label("Bathrooms:"),
                                dbc.Input(id="Bathrooms",
                                          value=1, 
                                          type="number",
                                          ),
                                ])

select_beds = html.Div([
                                html.Label("Beds:"),
                                dbc.Input(id="Beds",
                                          value=1, 
                                          type="number",
                                          ),
                                ])

select_availability365 = html.Div([
                                html.Label("Amount of days you want to rent out\
                                           your porperty per year:"),
                                dbc.Input(id="Availability365",
                                          value=365, 
                                          type="number",
                                          min=0, 
                                          max=365, 
                                          step=1,
                                          ),
                                ])

select_amenities = html.Div([dbc.Checklist(options=[{'label': i,
                                                     'value': i
                                                     } for i in amenities_all],
                                           value=[],
                                           id="Amenities",
                                           )],
                            style={'columnCount': 2})

# Creating the header and the subheader.

header = html.H1(children='Airbnb Revenue Estimator - City of Buenos Aires', 
                 style={"text-align": "center"})

subheader = html.Div(children=[dcc.Markdown('''
                                            Fill out the details of your property and 
                                            calculate the ***estimated revenue*** by 
                                            renting out your property on Airbnb and the
                                            ***yearly return on investment*** based on the 
                                            estimated value of your property.
                                            ''')], 
                    style={"text-align": "center",
                           'font-size': 18,
                           'marginLeft': 20, 
                           'marginRight': 20})

# Structuring all the different items created above in different containers and columns.

container1 = dbc.Container([html.Br(),
                            header,
                            html.Br(),
                            subheader],                            
                            fluid=True)                               

container2 = dbc.Container(dbc.Row(
                    [
                        dbc.Col(html.Div(children=[select_neighbourhood,
                                                   select_property_type,
                                                   select_room_type,
                                                   select_availability365],
                                         style={'columnCount': 1})),
                        
                        dbc.Col(html.Div(children=[select_latitude,
                                                   select_longitude,
                                                   dcc.Markdown('''
                                                                You can search the latitude
                                                                and longitude of your property 
                                                                [HERE](https://www.maps.ie/coordinates.html).
                                                                ''',
                                                                style={'font-size': 12}),
                                                    select_covered_surface,
                                                    select_total_surface],
                                         style={'columnCount': 1})),
                        
                        dbc.Col(html.Div(children=[select_accommodates,
                                                   select_rooms,
                                                   select_bedrooms,
                                                   select_beds,
                                                   select_bathrooms],
                                         style={'columnCount': 1})), 
                        
                        dbc.Col(html.Div(children=[dbc.Label('Amenities:'),
                                                   select_amenities],
                                         )),
                    ]
                  ), fluid=True)

container3 = dbc.Container(dbc.Row([dbc.Button("Predict your estimated revenue!", 
                                               id="Predict", 
                                               color="primary", 
                                               block=True,
                                               style={'marginLeft': 10, 'marginRight': 10,
                                                      'fontWeight': 'bold'})]),
                           fluid=True)

container4 = dbc.Container([
                    dbc.Row([
                        dbc.Col([html.Div('Estimated Airbnb price:'),
                                 html.Div(id='prediction_airbnb', 
                                         style={'fontWeight': 'bold',
                                                'font-size': 18})]),
                        
                        dbc.Col([html.Div('Estimated property value:'),
                                 html.Div(id='prediction_properati', 
                                         style={'fontWeight': 'bold',
                                                'font-size': 18})]),
                        
                        dbc.Col([html.Div('Yearly revenue through Airbnb*:'),
                                 html.Div(id='yearly_revenue_airbnb', 
                                         style={'fontWeight': 'bold',
                                                'font-size': 18})]),
                        
                        dbc.Col([html.Div('Yearly return on investment*:'),
                                 html.Div(id='yearly_roi_airbnb', 
                                         style={'fontWeight': 'bold',
                                                'font-size': 18})])
                        ]),
                    html.Br(),
                    dbc.Row([dcc.Markdown('''
                                          **In case the property is rented out 
                                          every available day.*
                                          ''',
                                          style={'font-size': 14,
                                                 'marginLeft': 15})
                        ])
                    ],
                    fluid=True)

container5 = dbc.Container(dbc.Row([dbc.Col([dcc.Graph(id='graph_airbnb',
                                                       config={'displayModeBar': False})]), 
                                    dbc.Col([dcc.Graph(id='graph_properati',
                                                       config={'displayModeBar': False})])
                                    ]), 
                                    fluid=True)
# Putting together the final layout.

layout = html.Div(children=[container1, 
                            html.Br(),
                            container2,
                            html.Br(),
                            container3,
                            html.Br(),
                            container4,
                            html.Br(),
                            container5
                            ])

app.layout = layout

# Creating the callback function to update the predictions.

@app.callback(
    Output(component_id='prediction_airbnb', component_property='children'),
    Output(component_id='prediction_properati', component_property='children'),
    Output(component_id='yearly_revenue_airbnb', component_property='children'),
    Output(component_id='yearly_roi_airbnb', component_property='children'),
    [Input('Predict', 'n_clicks')],
    state=[
     State(component_id='Neighbourhood', component_property='value'),
     State(component_id='Property_type', component_property='value'),
     State(component_id='Room_type', component_property='value'),
     State(component_id='Latitude', component_property='value'),
     State(component_id='Longitude', component_property='value'),
     State(component_id='Covered_surface', component_property='value'),
     State(component_id='Total_surface', component_property='value'),
     State(component_id='Accommodates', component_property='value'),
     State(component_id='Rooms', component_property='value'),
     State(component_id='Bedrooms', component_property='value'),
     State(component_id='Bathrooms', component_property='value'),
     State(component_id='Beds', component_property='value'),
     State(component_id='Availability365', component_property='value'),
     State(component_id='Amenities', component_property='value')]
)
def update_output_div(n_clicks, neighbourhood, prop_type, room_type, lat, lon, 
                      cov_surface, total_surface, acc, rooms, bedrooms,
                      bathrooms, beds, availability, amenities):
    
    # Fill the airbnb_predict dataframe with the input data
    airbnb_predict.loc[0, 'latitude'] = lat
    airbnb_predict.loc[0, 'longitude'] = lon
    airbnb_predict.loc[0, 'accommodates'] = acc
    airbnb_predict.loc[0, 'bathrooms'] = bathrooms
    airbnb_predict.loc[0, 'bedrooms'] = bedrooms
    airbnb_predict.loc[0, 'beds'] = beds
    airbnb_predict.loc[0, 'cleaning_fee'] = 0
    airbnb_predict.loc[0, 'minimum_nights'] = 1
    airbnb_predict.loc[0, 'maximum_nights'] = availability
    airbnb_predict.loc[0, 'reviews_per_month'] = airbnb.loc[(airbnb.neighbourhood==neighbourhood)&
                                                            (airbnb.property_type_general==prop_type)&
                                                            (airbnb.room_type==room_type),
                                                            'reviews_per_month'].mean()
    for i in amenities_airbnb:
        airbnb_predict.loc[0, i] = 1 if i in amenities else 0
    airbnb_predict.loc[0, 'min_dist_to_subte'] = calculate_min_distance_subte(lon, lat)
    airbnb_predict.loc[0, 'min_dist_to_subteD'] = calculate_min_distance_subteD(lon, lat)
    airbnb_predict.loc[0, 'Balvanera'] = 1 if neighbourhood=='Balvanera' else 0
    airbnb_predict.loc[0, 'Colegiales'] = 1 if neighbourhood=='Colegiales' else 0
    airbnb_predict.loc[0, 'Constitución'] = 1 if neighbourhood=='Constitución' else 0
    airbnb_predict.loc[0, 'Monserrat'] = 1 if neighbourhood=='Monserrat' else 0
    airbnb_predict.loc[0, 'Palermo'] = 1 if neighbourhood=='Palermo' else 0
    airbnb_predict.loc[0, 'Puerto Madero'] = 1 if neighbourhood=='Puerto Madero' else 0
    airbnb_predict.loc[0, 'Recoleta'] = 1 if neighbourhood=='Recoleta' else 0
    airbnb_predict.loc[0, 'Retiro'] = 1 if neighbourhood=='Retiro' else 0
    airbnb_predict.loc[0, 'Saavedra'] = 1 if neighbourhood=='Saavedra' else 0
    airbnb_predict.loc[0, 'San Cristóbal'] = 1 if neighbourhood=='San Cristóbal' else 0
    airbnb_predict.loc[0, 'San Nicolás'] = 1 if neighbourhood=='San Nicolás' else 0
    airbnb_predict.loc[0, 'San Telmo'] = 1 if neighbourhood=='San Telmo' else 0
    airbnb_predict.loc[0, 'Villa Urquiza'] = 1 if neighbourhood=='Villa Urquiza' else 0
    airbnb_predict.loc[0, 'Private room'] = 1 if room_type=='Private room' else 0
    airbnb_predict.loc[0, 'Shared room'] = 1 if room_type=='Shared room' else 0
    airbnb_predict2 = airbnb_predict.astype(float)
    

    # Fill the properati_predict dataframe with the input data         
    properati_predict.loc[0, 'lat'] = lat
    properati_predict.loc[0, 'lon'] = lon
    properati_predict.loc[0, 'rooms'] = rooms
    properati_predict.loc[0, 'bathrooms'] = bathrooms
    properati_predict.loc[0, 'bedrooms'] = bedrooms
    properati_predict.loc[0, 'surface_total'] = total_surface
    properati_predict.loc[0, 'surface_covered'] = cov_surface
    for i in amenities_properati:
        properati_predict.loc[0, i] = 1 if i in amenities else 0
    properati_predict.loc[0, 'min_dist_to_subte'] = calculate_min_distance_subte(lon, lat)
    properati_predict.loc[0, 'min_dist_to_subteD'] = calculate_min_distance_subteD(lon, lat)
    properati_predict.loc[0, 'ALMAGRO'] = 1 if neighbourhood=='Almagro' else 0
    properati_predict.loc[0, 'BALVANERA'] = 1 if neighbourhood=='Balvanera' else 0
    properati_predict.loc[0, 'BELGRANO'] = 1 if neighbourhood=='Belgrano' else 0
    properati_predict.loc[0, 'BOCA'] = 1 if neighbourhood=='La Boca' else 0
    properati_predict.loc[0, 'CABALLITO'] = 1 if neighbourhood=='Caballito' else 0
    properati_predict.loc[0, 'CHACARITA'] = 1 if neighbourhood=='Chacarita' else 0
    properati_predict.loc[0, 'COGHLAN'] = 1 if neighbourhood=='Coghlan' else 0
    properati_predict.loc[0, 'CONSTITUCION'] = 1 if neighbourhood=='Constitución' else 0
    properati_predict.loc[0, 'MONSERRAT'] = 1 if neighbourhood=='Monserrat' else 0
    properati_predict.loc[0, 'NUEVA POMPEYA'] = 0
    properati_predict.loc[0, 'PALERMO'] = 1 if neighbourhood=='Palermo' else 0
    properati_predict.loc[0, 'PARQUE CHACABUCO'] = 1 if neighbourhood=='Parque Chacabuco' else 0
    properati_predict.loc[0, 'PATERNAL'] = 0
    properati_predict.loc[0, 'PUERTO MADERO'] = 1 if neighbourhood=='Puerto Madero' else 0
    properati_predict.loc[0, 'RECOLETA'] = 1 if neighbourhood=='Recoleta' else 0
    properati_predict.loc[0, 'RETIRO'] = 1 if neighbourhood=='Retiro' else 0
    properati_predict.loc[0, 'SAAVEDRA'] = 1 if neighbourhood=='Saavedra' else 0
    properati_predict.loc[0, 'SAN CRISTOBAL'] = 1 if neighbourhood=='San Cristóbal' else 0
    properati_predict.loc[0, 'SAN NICOLAS'] = 1 if neighbourhood=='San Nicolás' else 0
    properati_predict.loc[0, 'VILLA DEVOTO'] = 1 if neighbourhood=='Villa Devoto' else 0
    properati_predict.loc[0, 'VILLA LUGANO'] = 1 if neighbourhood=='Villa Lugano' else 0
    properati_predict2 = properati_predict.astype(float)
    
    # Predict the Airbnb price
    prediction_airbnb = np.exp(airbnb_model.predict(airbnb_predict2))
    
    # Predict the property value
    prediction_properati = np.exp(properati_model.predict(properati_predict2))
    
    # Calculate the yearly revenue that can be obtained through Airbnb
    yearly_revenue = prediction_airbnb * availability
    
    # Calculate the yearly return on investment
    yearly_roi = (yearly_revenue/prediction_properati)*100
    
    return 'USD {}/night'.format(np.round(prediction_airbnb[0], 2)),\
            'USD {0:,.2f}'.format(np.round(prediction_properati[0], 2)),\
            'USD {0:,.2f}'.format(np.round(yearly_revenue[0],2)),\
            '{}%'.format(np.round(yearly_roi[0], 2))
    
    
# Creating the callback function to update the figures.            

@app.callback(
    Output(component_id='graph_airbnb', component_property='figure'),
    Output(component_id='graph_properati', component_property='figure'),
    [Input('Predict', 'n_clicks')],
    state=[
     State(component_id='Neighbourhood', component_property='value'),
     State(component_id='Property_type', component_property='value'),
     State(component_id='Room_type', component_property='value')])
def update_figures(n_clicks, neighbourhood, prop_type, room_type):
    
    # Create a filtered dataframe with only the selected property type and neighbourhood
    filtered_df_airbnb = airbnb[(airbnb.property_type_general==prop_type)&
                                (airbnb.neighbourhood==neighbourhood)]
    
    # Create the graph for the Airbnb price distribution                                          
    fig_airbnb = go.Figure()
    fig_airbnb.add_trace(go.Histogram(x=airbnb.price_usd,
                                      xbins=dict(start=0, end=200, size=5),
                                      name='General distribution',
                                      marker_color='#B4DCEC'))
    fig_airbnb.add_trace(go.Histogram(x=filtered_df_airbnb.price_usd,
                               xbins=dict(start=0, end=200, size=5),
                               name='Selected neighbourhood and property type',
                               marker_color='#046FB5'))
    fig_airbnb.update_layout(title_text='Distribution of Airbnb prices',
                             title_x=0.5,
                             xaxis_title_text='Price per night in USD',
                             yaxis_title_text='Count',
                             yaxis_gridcolor = '#F1F1F1',
                             barmode='overlay',
                             font_color="black",
                             legend=dict(yanchor="top", xanchor="right"),
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)')
    fig_airbnb.update_traces(opacity=0.80)
    
    # Tranform the selected neighbourhood into uppercase, so it matches the BARRIO colum of
    # the properati dataset.
    neighbourhood_upper = unidecode.unidecode(str(neighbourhood).upper())
    
    # Create a filtered dataframe with only the selected property type and neighbourhood
    filtered_df_properati = properati.copy()
    filtered_df_properati.loc[filtered_df_properati.BARRIO == 'BOCA', 'BARRIO'] = 'LA BOCA'
    filtered_df_properati.loc[filtered_df_properati.BARRIO == 'NUÑEZ', 'BARRIO'] = 'NUNEZ'
    filtered_df_properati = filtered_df_properati[
            (filtered_df_properati.property_type_general==str(prop_type).lower())&
            (filtered_df_properati.BARRIO==neighbourhood_upper)]
    
    # Create the graph for the Properati price distribution 
    fig_properati = go.Figure()
    fig_properati.add_trace(go.Histogram(x=properati.price_usd,
                                         xbins=dict(start=0, end=2000000, size=30000),
                                         name='General distribution',
                                         marker_color='#B4DCEC'))
    fig_properati.add_trace(go.Histogram(x=filtered_df_properati.price_usd,
                                         xbins=dict(start=0, end=2000000, size=30000),
                                         name='Selected neighbourhood and property type',
                                         marker_color='#046FB5'))
    fig_properati.update_layout(title_text='Distribution of total property prices',
                             title_x=0.5,
                             xaxis_title_text='Total property price in USD',
                             yaxis_title_text='Count',
                             yaxis_gridcolor = '#F1F1F1',
                             barmode='overlay',
                             font_color="black",
                             legend=dict(yanchor="top", xanchor="right"),
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)')
    fig_properati.update_traces(opacity=0.80)
    
    return fig_airbnb, fig_properati
        
if __name__ == '__main__':
    app.run_server(debug=True)
