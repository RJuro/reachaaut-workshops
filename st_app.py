import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk
import numpy as np
import pandas as pd 
import seaborn as sns #seaborn til plots
from matplotlib import pyplot as plt #plot control
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
#sns.set() #plot style

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)


# LOAD DATA ONCE
@st.experimental_singleton
def load_data():
    data = pd.read_csv('cph-listings.gz')

    # also preprocess as we did in the notebook
    data = data[data.number_of_reviews > 0]
    data = data[data.room_type.isin(['Private room', 'Entire home/apt'])]
    data['price_z'] = (data['price'] - data['price'].mean())/data['price'].std(ddof=0)
    data['price_z'] = data['price_z'].abs()
    data = data[data.price_z < 3]
    data['log_price'] = np.log(data['price'])

    return data

st.set_page_config(page_title='Streamlit - DataViz',
page_icon="🚀",
layout='wide'
)

palette=['#FF4C4B','#FF9361', '#159090']


# LOAD THE DATA NOW!
data = load_data()


# STREAMLIT APP LAYOUT
data = load_data()

# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    st.title("AirBnb rentals in Copenhagen 🇩🇰")

    price_selected = st.slider("Select price range", min_value = int(data.price.min()), max_value= int(data.price.max()), value = (300,3000), step=50)

    data = data[(data.price > price_selected[0]) & (data.price < price_selected[1])]


with row1_2:
    st.markdown("""
    ##
    ## Exploring prices for AirBnb rentals in CPH using the data from
    insideairbnb.com\n

    Lorem markdownum et conlapsamque et tellus deorum cum nervo undas excussis
    amorem silent ver quidem exierat? Mei nequiquam ignesque tales, elisi a
    habitusque, Panopesque sed gladii, Cernis aut meis. Iamque glorior cecidit
    excelsa ramis, da aethera amores!
    """
    )

st.markdown('---')

row_2_1_n, row_2_2_n = st.columns((2,3))

with row_2_1_n:

    st.markdown("""

    ## Dicere quaecumque patiar hunc notamque aequa hos
Natura tu femorum regia. Primis vesci, Munychiosque dolore placat; Atracides
pecoris se.

> Citharae patet Achaidas alta sistite Lernae adicit exorabilis illum,
> dissimulator quoque dubitabilis. Enim contemptor ob infames haut, tectum levem
> cruribus, ire!

    """)

    #filter for neighborhoods
    neighbourhood_select = st.multiselect('Select neighbourhoods', data.neighbourhood.unique(), data.neighbourhood.unique())
    data = data[data.neighbourhood.isin(neighbourhood_select)]

with row_2_2_n:

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data[['name','room_type','price', "longitude", "latitude"]].dropna(),
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        radius_scale=10,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["longitude", "latitude"],
        get_radius=10*"log_price",
        get_color=[255, 140, 0],
        get_line_color=[0, 0, 0],
    )

    # Set the viewport location
    view_state = pdk.ViewState(latitude=data['latitude'].mean(), longitude=data['longitude'].mean(), zoom=12, pitch=50)

    # Renders
    r = pdk.Deck(layers=[layer], 
    initial_view_state=view_state,
    #map_style='mapbox://styles/mapbox/light-v9',
    tooltip={"text": "{name}\n{room_type}\n{price}"}
    )

    st.pydeck_chart(r)


st.markdown('---')

row_3_1_n, row_3_2_n, row_3_3_n = st.columns((2,2,1))

with row_3_1_n:

     # custom pretty color palette
    colorRange = [
    [1, 152, 189],
    [73, 227, 206],
    [216, 254, 181],
    [254, 237, 177],
    [254, 173, 84],
    [209, 55, 78]
    ]

    # Define a layer to display on a map

    layer = pdk.Layer(
        "HexagonLayer",
        data=data[["longitude", "latitude"]],
        get_position=["longitude", "latitude"],
        auto_highlight=True,
        elevation_scale=50,
        pickable=True,
        radius=50,
        opacity=0.3,
        elevation_range=[0, 200],
        color_range = colorRange,
        extruded=True,
        coverage=1,
    )

    # Set the viewport location
    view_state = pdk.ViewState(latitude=data['latitude'].mean(), longitude=data['longitude'].mean(), zoom=10, min_zoom=5, max_zoom=15, pitch=40.5, bearing=-27.36)

    # Render
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{elevationValue}"})

    st.pydeck_chart(r)

with row_3_2_n:

    st.write('''We can see that Indre By has the most amount of places. 
                Overall, private rooms are the exception. Mostly you rent out a whole room.
            ''')
    fig = plt.figure()
    sns.countplot(y="neighbourhood", 
                hue="room_type", 
                data=data, 
                palette=palette,
                order = data.neighbourhood.value_counts().index)
    st.pyplot(fig)

with row_3_3_n:

    st.markdown("""
    Meque ostendens auras est traderet, in nefas pulchra draconem oblitis praelata
corporis creverunt: sed et! Hectoreis veluti novae tecum lupos: virgo, **te**
reportat habet molliter pelle; medio non. Dominari cervos me secus palluit
utque. Is dixit dixit fessa candore passa corpus iuvat en confusa tantum
Numitorque quae praemia terga, pennas.
    """)


st.markdown('---')

row_4_1_n, row_4_2_n = st.columns((3,2))

with row_4_1_n:

    st.markdown("""
    ## Vertunt quoque

    Per unam nubilibus rursus me qua et Inarimen lapis revocamina viscera occupat
    saltumque et veras. Cedere lanigerosque antra; hic sua urbem mollia est
    contermina heros.

    ## Exspectatas iura

    Addiderat in ille, Palaemona Venerem, sumus, placeas Eurystheus populos
    pallidaque Bacchus est alto non puerilibus annos. Tum pacis locatas in vidi
    temptasse emicuit dextra invictumque opem fratrem; cum erat et ergo. Sermonibus
    quos crimine totis. [Multumque vittas](http://si-tutum.net/quod-flammis): fuit
    celeres graviore loca, indignanda formatae!

    1. Utraque ille tenet tendens stamine promptu utque
    2. Dextrasque bello publica fretum Romuleos illum gravitate
    3. Videtur non spuma mittitur
    4. Induiturque tutus
    5. Displicet Lapitheia
    6. Solvit vetus et Phrygia tuetur cinctaeque vincet

    Lemnos ita causa dependent, tulit *et quae* nasci sua ambo nervis. Fratribus
    Medusae, aut lacus quae fronde restant apertas obsidis pollentibus qui mare.
    Genitor perdere tenerum verbis praedictaque solum sacerdos madescit; neque alis
    non aberant!""")

    fig = plt.figure()
    sns.boxplot(data = data, 
                x = "price", 
                y = "neighbourhood" , 
                hue = "room_type", 
                showfliers=False,
                palette=palette,
                order = data.groupby('neighbourhood').price.mean().sort_values(ascending=False).index)
    st.pyplot(fig)

with row_4_2_n:

    est = sm.OLS(endog=data['reviews_per_month'], exog=sm.add_constant(data['price'])).fit()
    est2 = sm.OLS(endog=data['reviews_per_month'], exog=sm.add_constant(data[['price','availability_365']])).fit()


    stargazer = Stargazer([est2, est])

    st.write(stargazer)

    st.markdown('''Do cheaper places get more reviews? Looks like people gravitate towards the average priced ones. Price: $log(price)$
        ''')

    st.write('''Indre By is at the same time on average the most expensive place to stay.\n\n
            ''')

    g = sns.lmplot(x="reviews_per_month",
                y="log_price",
                hue="room_type",
                height=7,
                data=data,
                    scatter_kws={'alpha':0.5},
                palette=palette)
    g.set_xlabels('Reviews/month')
    g.set_ylabels('Price')
    st.pyplot(g)
