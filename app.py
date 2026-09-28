"""Amsterdam Airbnb Price Predictor — CatBoost Streamlit frontend.

Estimates a listing's nightly price with a trained CatBoostRegressor.
The target was square-root transformed in training, so raw predictions
are squared back to currency. Artifacts live beside this script:

- airbnb_catboost_model.joblib (CatBoostRegressor, 24 features)
- neighbourhood_encoding_lookup.pkl (neighbourhood -> encoded value)
- global_mean_price.pkl (fallback for unseen neighbourhoods)
- room_type_encoder.pkl (optional; direct one-hot fallback if absent)

Run with::

    streamlit run airbnb_catboost_app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = "airbnb_catboost_model.joblib"
ENCODER_PATH = "room_type_encoder.pkl"
LOOKUP_PATH = "neighbourhood_encoding_lookup.pkl"
GLOBAL_MEAN_PATH = "global_mean_price.pkl"

# Exact training feature order (24 columns).
FEATURE_ORDER = [
    "host_is_superhost", "host_listings_count", "host_identity_verified",
    "latitude", "longitude", "accommodates", "bathrooms", "bedrooms", "beds",
    "minimum_nights", "maximum_nights", "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm", "has_availability", "availability_365",
    "number_of_reviews", "review_scores_rating", "reviews_per_month",
    "amenities_count", "room_type_Entire home/apt", "room_type_Hotel room",
    "room_type_Private room", "room_type_Shared room",
    "neighbourhood_encoded",
]

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
ROOM_COLS = [f"room_type_{rt}" for rt in ROOM_TYPES]

AMENITIES = [
    "Wifi", "Kitchen", "Washer", "Dryer", "Air conditioning", "Heating",
    "TV", "Pool", "Free parking", "Hot tub", "Gym", "Elevator",
    "Dishwasher", "Iron", "Hair dryer", "Workspace", "BBQ grill",
    "Balcony", "Smoke alarm", "Carbon monoxide alarm",
]

MODEL_R2 = "70%"
AMSTERDAM_LAT, AMSTERDAM_LON = 52.3676, 4.9041

st.set_page_config(
    page_title="Amsterdam Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ------------------------------------------------------------------
# Custom CSS — warm coral theme
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    [data-testid="stAppViewContainer"] { background: #FAFAFA; }
    [data-testid="stSidebar"] { background: #FFFFFF; }

    .badge {
        display: inline-block; background: #1a1a1a; color: #FF5A5F;
        font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 0.3rem 0.9rem;
        border-radius: 999px; margin-bottom: 0.5rem;
    }
    .subtitle { color: #555; font-size: 1.05rem; margin-top: -0.5rem; }

    /* Primary buttons: coral with white text */
    div.stButton > button[kind="primary"] {
        background-color: #FF5A5F;
        color: #fff;
        border: none;
        font-weight: 700;
        font-size: 1.05rem;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        width: 100%;
        box-shadow: 0 4px 12px rgba(255, 90, 95, 0.35);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #E0484D;
        color: #fff;
    }

    /* KPI metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #DDDDDD;
        border-left: 5px solid #FF5A5F;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* Hero price centerpiece */
    .hero-price {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        color: #fff;
        text-align: center;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.25);
        margin-bottom: 1.25rem;
    }
    .hero-price .label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #FDA4AF;
    }
    .hero-price .value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.1rem 0;
    }
    .hero-price .range {
        color: #D1D5DB;
        font-size: 0.95rem;
    }

    /* Empty state */
    .empty-state {
        background: #FFFFFF;
        border: 1px dashed #DDDDDD;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: #717171;
    }
    .empty-state .icon { font-size: 2.4rem; }
    .empty-state h3 { color: #1a1a1a; margin: 0.4rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Artifact loading (cached; friendly errors) — unchanged
# ------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """Load CatBoost model, encoder (optional), lookup and global mean."""
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{MODEL_PATH}' not found. Place it beside this script."
        )
    try:
        encoder = joblib.load(ENCODER_PATH)
    except FileNotFoundError:
        encoder = None  # direct one-hot fallback (fixed 4 columns)
    try:
        lookup = dict(joblib.load(LOOKUP_PATH))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{LOOKUP_PATH}' not found. Place it beside this script."
        )
    try:
        global_mean = float(joblib.load(GLOBAL_MEAN_PATH))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{GLOBAL_MEAN_PATH}' not found. Place it beside this script."
        )
    return model, encoder, lookup, global_mean


try:
    model, room_encoder, hood_lookup, hood_fallback = load_assets()
except FileNotFoundError as err:
    st.error(f"Missing model file. {err}")
    st.stop()
except Exception as err:
    st.error(f"Could not load model assets: {err}")
    st.stop()

# Neighbourhood options always mirror the saved lookup file.
NEIGHBOURHOODS = sorted(hood_lookup)


# ------------------------------------------------------------------
# Feature-row builder (exact 24-column training order) — unchanged
# ------------------------------------------------------------------
def build_feature_row(room_type, neighbourhood, accommodates, bedrooms,
                      bathrooms, beds, minimum_nights, maximum_nights,
                      superhost, verified, host_listings_count,
                      availability_365, number_of_reviews,
                      review_scores_rating, reviews_per_month, latitude,
                      longitude, amenities_count):
    """Assemble a raw single-row DataFrame in FEATURE_ORDER."""
    onehots = {col: 0 for col in ROOM_COLS}
    if room_encoder is not None:
        encoded = room_encoder.transform(pd.DataFrame({"room_type": [room_type]}))
        for col, val in zip(room_encoder.get_feature_names_out(), encoded[0]):
            if col in onehots:
                onehots[col] = int(val)
    else:
        match = f"room_type_{room_type}"
        if match in onehots:
            onehots[match] = 1

    row = {
        "host_is_superhost": int(superhost),
        "host_listings_count": int(host_listings_count),
        "host_identity_verified": int(verified),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "accommodates": int(accommodates),
        "bathrooms": float(bathrooms),
        "bedrooms": int(bedrooms),
        "beds": int(beds),
        "minimum_nights": int(minimum_nights),
        "maximum_nights": int(maximum_nights),
        "minimum_nights_avg_ntm": float(minimum_nights),
        "maximum_nights_avg_ntm": float(maximum_nights),
        "has_availability": 1,
        "availability_365": int(availability_365),
        "number_of_reviews": int(number_of_reviews),
        "review_scores_rating": float(review_scores_rating),
        "reviews_per_month": float(reviews_per_month),
        "amenities_count": int(amenities_count),
        **onehots,
        "neighbourhood_encoded": float(
            hood_lookup.get(neighbourhood, hood_fallback)
        ),
    }
    return pd.DataFrame([{k: row[k] for k in FEATURE_ORDER}])


def predict_price(input_df):
    """Run CatBoost and square the sqrt-transformed output to currency."""
    raw = float(model.predict(input_df)[0])
    return max(0.0, raw ** 2)


# ------------------------------------------------------------------
# Sidebar inputs (grouped expanders)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Listing Inputs")
    st.caption("Adjust the details below, then hit **Estimate Price**.")

    with st.expander("🏠 Property Details", expanded=True):
        room_type = st.selectbox("Room type", ROOM_TYPES)
        accommodates = st.number_input("Accommodates", 1, 16, 2)
        bedrooms = st.number_input("Bedrooms", 0, 10, 1)
        bathrooms = st.number_input("Bathrooms", 0.0, 8.0, 1.0, 0.5)
        beds = st.number_input("Beds", 1, 12, 1)
        minimum_nights = st.number_input("Minimum nights", 1, 365, 2)
        maximum_nights = st.number_input("Maximum nights", 1, 2000, 30)

    with st.expander("📍 Location", expanded=True):
        neighbourhood = st.selectbox("Neighbourhood", NEIGHBOURHOODS)
        latitude = st.number_input("Latitude", value=AMSTERDAM_LAT, format="%.4f")
        longitude = st.number_input("Longitude", value=AMSTERDAM_LON, format="%.4f")

    with st.expander("✨ Amenities"):
        selected_amenities = st.multiselect("Amenities", AMENITIES,
                                            default=["Wifi", "Kitchen"])
        amenities_count = len(selected_amenities)
        st.caption(f"{amenities_count} selected (fed to the model as a count)")

    with st.expander("⭐ Host & Reviews"):
        superhost = st.checkbox("Host is a superhost", value=False)
        verified = st.checkbox("Host identity verified", value=True)
        host_listings_count = st.number_input("Host listings count", 1, 500, 1)
        availability_365 = st.slider("Availability (days/year)", 0, 365, 180)
        number_of_reviews = st.number_input("Number of reviews", 0, 5000, 25)
        review_scores_rating = st.slider("Review rating", 0.0, 5.0, 4.8, 0.05)
        reviews_per_month = st.number_input("Reviews per month", 0.0, 50.0, 1.2,
                                            0.1)

# ------------------------------------------------------------------
# Main panel
# ------------------------------------------------------------------
st.markdown('<span class="badge">CatBoost &middot; Amsterdam Listings</span>',
            unsafe_allow_html=True)
st.title("Amsterdam Airbnb Price Predictor")
st.markdown(
    '<p class="subtitle">Set your listing details in the sidebar and get '
    "an instant nightly-price estimate.</p>",
    unsafe_allow_html=True,
)
st.divider()

if st.button("Estimate Price", type="primary"):
    try:
        with st.spinner("Calculating your estimate..."):
            input_df = build_feature_row(
                room_type, neighbourhood, accommodates, bedrooms,
                bathrooms, beds, minimum_nights, maximum_nights,
                superhost, verified, host_listings_count,
                availability_365, number_of_reviews,
                review_scores_rating, reviews_per_month, latitude,
                longitude, amenities_count,
            )
            price = predict_price(input_df)
        st.session_state["last_price"] = price
        st.session_state["last_accommodates"] = int(accommodates)
    except Exception as err:
        st.error(f"Could not compute the estimate: {err}")

if "last_price" in st.session_state:
    price = st.session_state["last_price"]
    per_guest = price / max(int(st.session_state["last_accommodates"]), 1)

    st.markdown(
        f"""
        <div class="hero-price">
            <div class="label">Estimated nightly price</div>
            <p class="value">€{price:,.2f}</p>
            <div class="range">€{per_guest:,.2f} per guest &middot; per night</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Estimated Price", f"€{price:,.2f}")
    k2.metric("Price per Guest", f"€{per_guest:,.2f}")
    k3.metric("Model Accuracy (R²)", MODEL_R2)
else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">🏡</div>
            <h3>Ready when you are</h3>
            <p>Fill in your listing details in the sidebar, then press
            <b>Estimate Price</b> to see the nightly estimate here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
# pip install streamlit pandas numpy joblib scikit-learn catboost
# streamlit run airbnb_catboost_app.py
