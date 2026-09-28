"""Amsterdam Airbnb Price Predictor — CatBoost Streamlit frontend.

Light-only interface with sidebar inputs. Estimates a listing's nightly
price with a trained CatBoostRegressor. The target was square-root
transformed in training, so raw predictions are squared back to currency.
Artifacts live beside this script:

- airbnb_catboost_model.joblib (CatBoostRegressor, 24 features)
- neighbourhood_encoding_lookup.pkl (neighbourhood -> encoded value)
- global_mean_price.pkl (fallback for unseen neighbourhoods)
- room_type_encoder.pkl (optional; direct one-hot fallback if absent)

The light theme is pinned in `.streamlit/config.toml`, and every color
below is declared explicitly so browser/OS dark mode cannot alter it.

Run with::

    streamlit run airbnb_catboost_app.py
"""

import joblib
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
# Custom CSS — light-only design system.
# Every color is explicit; no dark-mode rules, no color-scheme
# detection, so the app renders identically in any browser theme.
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #F7F7F7;
        color: #222222;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    [data-testid="stAppViewContainer"] { background-color: #F7F7F7; }
    .block-container {
        background-color: #F7F7F7;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ---------- Sidebar: white, dark text, subtle border ---------- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E5E5;
    }
    [data-testid="stSidebar"] h2 { color: #222222; }
    [data-testid="stSidebar"] details summary p {
        font-weight: 700; color: #222222;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 12px;
    }

    /* ---------- Sidebar light scrollbar (track/thumb/hover) ---------- */
    [data-testid="stSidebar"] { scrollbar-width: thin;
        scrollbar-color: #C9C5BC #F1EFEA; }
    [data-testid="stSidebar"] ::-webkit-scrollbar { width: 10px; }
    [data-testid="stSidebar"] ::-webkit-scrollbar-track {
        background: #F1EFEA; border-radius: 8px;
    }
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background: #C9C5BC; border-radius: 8px;
        border: 2px solid #F1EFEA; background-clip: padding-box;
        min-height: 40px;
    }
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
        background: #A8A39A;
        border: 2px solid #F1EFEA; background-clip: padding-box;
    }

    /* ---------- Header ---------- */
    .badge {
        display: inline-block; background: #222222; color: #FF5A5F;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; padding: 0.3rem 0.9rem;
        border-radius: 999px; margin-bottom: 0.4rem;
    }
    .page-title { font-size: 2rem; font-weight: 800; color: #222222;
        margin: 0; }
    .page-sub { color: #666666; font-size: 1rem; margin-top: 0.3rem;
        max-width: 64ch; }

    /* ---------- Cards ---------- */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    .card-title { font-size: 1.15rem; font-weight: 700; color: #222222;
        margin: 0 0 0.5rem; }

    /* ---------- Inputs: white fields, dark text, gray borders ---------- */
    [data-testid="stWidgetLabel"] p { color: #222222; font-weight: 600; }
    [data-testid="stCheckbox"] label p { color: #222222; }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background-color: #FFFFFF; color: #222222;
        border-radius: 8px;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF; color: #222222;
        border-radius: 8px;
    }
    [data-testid="stSelectbox"] span,
    [data-testid="stMultiSelect"] span { color: #222222; }
    div[data-baseweb="popover"] { border-radius: 10px; overflow: hidden;
        border: 1px solid #E5E5E5; background-color: #FFFFFF; }
    ul[role="listbox"] { background-color: #FFFFFF; }
    ul[role="listbox"] li { color: #222222; }
    ul[role="listbox"] li:hover { background-color: #F2F2F2; }
    [data-testid="stCaptionContainer"] { color: #666666; }
    input:focus-visible, button:focus-visible,
    div[data-baseweb="select"]:focus-within {
        outline: 2px solid #FF5A5F; outline-offset: 2px;
    }

    /* ---------- Estimate button ---------- */
    div.stButton > button {
        background-color: #FF5A5F; color: #FFFFFF; border: none;
        font-weight: 700; font-size: 1.05rem; border-radius: 12px;
        padding: 0.75rem 1.2rem; width: 100%;
        box-shadow: 0 4px 12px rgba(255, 90, 95, 0.30);
        transition: background-color 0.12s ease, transform 0.12s ease,
            box-shadow 0.12s ease;
    }
    div.stButton > button:hover {
        background-color: #D63E43; color: #FFFFFF;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(255, 90, 95, 0.35);
    }
    div.stButton > button:active { transform: translateY(0); }
    div.stButton > button:disabled {
        opacity: 0.55; cursor: not-allowed; transform: none;
    }

    /* ---------- Result focal card (flat white, coral accent) ---------- */
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-left: 8px solid #FF5A5F;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .result-card .label {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #666666;
    }
    .result-card .value {
        font-size: clamp(2.4rem, 5vw, 3.2rem);
        font-weight: 800; color: #FF5A5F; margin: 0.1rem 0; line-height: 1.1;
    }
    .result-card .pernight { color: #666666; font-size: 1rem; }
    .result-card .note { color: #666666; font-size: 0.9rem; margin-top: 0.5rem; }

    /* ---------- KPI metrics ---------- */
    [data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1px solid #E5E5E5;
        border-left: 5px solid #FF5A5F; border-radius: 12px;
        padding: 0.9rem 1.1rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        color: #222222;
    }

    /* ---------- Compact empty state ---------- */
    .empty-state {
        background-color: #FFFFFF; border: 1px dashed #E5E5E5;
        border-radius: 12px; padding: 1.5rem 1.25rem; text-align: center;
        color: #666666;
    }
    .empty-state .icon { font-size: 1.8rem; }
    .empty-state h3 { color: #222222; margin: 0.3rem 0; font-size: 1.1rem; }
    .empty-state p { margin: 0; font-size: 0.92rem; }

    /* ---------- Alerts & dividers ---------- */
    [data-testid="stAlert"] { border-radius: 12px; }
    hr { border: none; border-top: 1px solid #E5E5E5; margin: 1rem 0; }

    /* ---------- Page scrollbar (light, visible) ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #F7F7F7; }
    ::-webkit-scrollbar-thumb {
        background: #C9C5BC; border-radius: 8px;
        border: 2px solid #F7F7F7; background-clip: padding-box;
        min-height: 40px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #A8A39A;
        border: 2px solid #F7F7F7; background-clip: padding-box; }

    .footer-note { text-align: center; color: #666666; font-size: 0.82rem;
        margin-top: 1.5rem; }

    @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .page-title { font-size: 1.6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Artifact loading — unchanged
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
# Header
# ------------------------------------------------------------------
st.markdown('<span class="badge">CatBoost &middot; Amsterdam Listings</span>',
            unsafe_allow_html=True)
st.title("Amsterdam Airbnb Price Predictor")
st.write(
    "Estimate the nightly price of an Airbnb listing based on its "
    "property details."
)

# ------------------------------------------------------------------
# Sidebar inputs (grouped sections) — same widgets and values
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Listing inputs")

    with st.expander("🏠 Property details", expanded=True):
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

    with st.expander("⭐ Host & reviews"):
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
st.divider()

if st.button("Estimate price", type="primary"):
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
        <div class="result-card">
            <div class="label">Estimated nightly price</div>
            <p class="value">€{price:,.2f}</p>
            <div class="pernight">per night</div>
            <div class="note">Based on the property details provided.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Estimated price", f"€{price:,.2f}")
    k2.metric("Price per guest", f"€{per_guest:,.2f}")
    k3.metric("Model accuracy (R²)", MODEL_R2)
else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">🏠</div>
            <h3>Ready when you are</h3>
            <p>Fill in your listing details in the sidebar, then select
            "Estimate price" to see the estimated nightly price.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(
    '<div class="footer-note">Amsterdam Airbnb Price Predictor · '
    "CatBoost · Built with Streamlit</div>",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
# pip install streamlit pandas numpy joblib scikit-learn catboost
# streamlit run airbnb_catboost_app.py
