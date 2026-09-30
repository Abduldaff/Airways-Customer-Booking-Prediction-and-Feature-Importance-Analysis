"""SkyNest — flight booking operations and conversion intelligence."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "customer_booking.csv"
METRICS_FILE = ROOT / "model_metrics.csv"
IMPORTANCE_FILE = ROOT / "feature_importance.csv"
PURPLE, BLUE, GREEN, GOLD = "#6C4CE5", "#3B82F6", "#10B981", "#F59E0B"

st.set_page_config(page_title="SkyNest | Flight Operations", page_icon="✈️", layout="wide")

st.markdown("""<style>
 .stApp {background:#f7f8fc;color:#182033} .block-container{padding-top:1.7rem;padding-bottom:2rem;max-width:1500px}
 [data-testid='stSidebar']{background:#21154c} [data-testid='stSidebar'] *{color:#f8f7ff!important}
 .metric-card{background:white;border-radius:18px;padding:18px 20px;border:1px solid #edf0f7;box-shadow:0 5px 18px rgba(35,23,74,.05);min-height:130px}
 .metric-label{font-size:.86rem;color:#667085;margin-bottom:9px}.metric-value{font-size:1.7rem;font-weight:750;color:#182033}.metric-note{font-size:.78rem;color:#10a779;margin-top:8px}
 .section-card{background:#fff;border:1px solid #e9eaf1;border-radius:18px;padding:16px 18px;margin-top:8px}
 h1,h2,h3{letter-spacing:-.03em} .caption-muted{color:#667085;font-size:.9rem}
 </style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Read data resiliently: source is latin-1 encoded."""
    data = pd.read_csv(DATA_FILE, encoding="latin1")
    data.columns = data.columns.str.strip()
    data["flight_day"] = data["flight_day"].str.strip().str.title()
    data["sales_channel"] = data["sales_channel"].str.strip().str.title()
    return data


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_FILE)


@st.cache_data(show_spinner=False)
def load_importance() -> pd.DataFrame:
    return pd.read_csv(IMPORTANCE_FILE)


@st.cache_resource(show_spinner="Training conversion model from the project dataset…")
def train_model(data: pd.DataFrame) -> Pipeline:
    features = data.drop(columns="booking_complete")
    categorical = features.select_dtypes(include="object").columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    prep = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numbers", "passthrough", numeric),
    ])
    # A compact production-friendly model; all training rows remain represented.
    model = RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=3,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(features, data.booking_complete)
    return pipe


def money(value: float) -> str:
    return f"${value:,.0f}"


def simulated_value(row: pd.Series) -> float:
    """Deterministic indicative booking value, explicitly not a source-data field."""
    return 95 + row.flight_duration * 78 + row.num_passengers * 42 + row.length_of_stay * 3


def init_state() -> None:
    if "managed_bookings" not in st.session_state:
        st.session_state.managed_bookings = []
    if "payments" not in st.session_state:
        st.session_state.payments = []
    if "activity" not in st.session_state:
        st.session_state.activity = [
            {"time": "Just now", "event": "Operations workspace opened", "type": "System"},
            {"time": "Today", "event": "Customer search dataset synchronised", "type": "Data"},
        ]
    if "flights" not in st.session_state:
        st.session_state.flights = [
            {"flight": "SN-218", "airline": "SkyNest Air", "route": "AKLDEL", "departure": datetime.combine(date.today(), datetime.strptime("09:00", "%H:%M").time()), "duration_h": 5.5, "fare": 490.0, "capacity": 180, "gate": "A12", "status": "Boarding"},
            {"flight": "SN-404", "airline": "SkyNest Air", "route": "DMKICN", "departure": datetime.combine(date.today(), datetime.strptime("13:45", "%H:%M").time()), "duration_h": 6.2, "fare": 625.0, "capacity": 220, "gate": "B06", "status": "On time"},
            {"flight": "SN-672", "airline": "SkyNest Air", "route": "PENTPE", "departure": datetime.combine(date.today() + timedelta(days=1), datetime.strptime("18:20", "%H:%M").time()), "duration_h": 4.1, "fare": 375.0, "capacity": 168, "gate": "C18", "status": "Scheduled"},
        ]


def sidebar(data: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    with st.sidebar:
        st.markdown("# ✈️ SkyNest")
        st.caption("FLIGHT OPERATIONS INTELLIGENCE")
        page = st.radio("Navigation", ["Overview", "Bookings & seats", "Flights & schedule", "Payments", "Flight tracking", "Conversion predictor", "Analytics & model", "Data quality"], label_visibility="collapsed")
        st.divider()
        st.markdown("#### Live data controls")
        channels = st.multiselect("Sales channels", sorted(data.sales_channel.unique()), default=sorted(data.sales_channel.unique()))
        trip_types = st.multiselect("Trip types", sorted(data.trip_type.unique()), default=sorted(data.trip_type.unique()))
        lead_max = st.slider("Maximum purchase lead (days)", 0, int(data.purchase_lead.max()), int(data.purchase_lead.max()))
        st.divider()
        st.caption(f"● Dataset online  ·  {len(data):,} customer searches")
        st.caption(f"Updated view: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}")
    filtered = data[(data.sales_channel.isin(channels)) & (data.trip_type.isin(trip_types)) & (data.purchase_lead <= lead_max)]
    return page, filtered


def metric_card(label: str, value: str, note: str, accent: str) -> None:
    st.markdown(f'<div class="metric-card" style="border-top:4px solid {accent}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)


def overview(data: pd.DataFrame) -> None:
    completed = data[data.booking_complete.eq(1)]
    conversion = data.booking_complete.mean()
    revenue = completed.apply(simulated_value, axis=1).sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total booking searches", f"{len(data):,}", "Live filtered dataset", PURPLE)
    with c2: metric_card("Completed bookings", f"{len(completed):,}", f"{conversion:.1%} conversion rate", GREEN)
    with c3: metric_card("Indicative booking value", money(revenue), "Derived from trip configuration", BLUE)
    with c4: metric_card("Active flight routes", f"{data.route.nunique():,}", f"Across {data.booking_origin.nunique()} origins", GOLD)

    left, right = st.columns((1.25, 1))
    with left:
        st.subheader("Conversion by departure day")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = data.groupby("flight_day").booking_complete.mean().reindex(days).fillna(0).mul(100)
        st.line_chart(daily, color=PURPLE, height=285)
        st.caption("Completed bookings ÷ searches. Values respond to the sidebar filters.")
    with right:
        st.subheader("Top routes by completed booking")
        routes = completed.route.value_counts().head(8).sort_values()
        st.bar_chart(routes, color=BLUE, horizontal=True, height=285)

    st.subheader("Operations watchlist")
    display = completed[["route", "booking_origin", "sales_channel", "trip_type", "num_passengers", "purchase_lead", "flight_hour", "flight_duration"]].head(12).copy()
    display.insert(0, "booking_id", [f"BK-{i:05d}" for i in range(1, len(display) + 1)])
    display["indicative_value"] = completed.head(len(display)).apply(simulated_value, axis=1).round(0)
    st.dataframe(display, use_container_width=True, hide_index=True, column_config={"indicative_value": st.column_config.NumberColumn("Indicative value ($)", format="$%d")})


def bookings(data: pd.DataFrame) -> None:
    st.subheader("Booking management")
    st.caption("Manage an in-session operations queue and export the filtered project records.")
    with st.expander("＋ Register a managed booking", expanded=False):
        with st.form("new_booking", clear_on_submit=True):
            a, b, c = st.columns(3)
            route = a.selectbox("Route", sorted(data.route.unique()))
            origin = b.selectbox("Booking origin", sorted(data.booking_origin.unique()))
            passengers = c.number_input("Passengers", 1, 9, 1)
            lead = a.number_input("Purchase lead (days)", 0, 867, 30)
            trip = b.selectbox("Trip type", sorted(data.trip_type.unique()))
            channel = c.selectbox("Channel", sorted(data.sales_channel.unique()))
            if st.form_submit_button("Create booking", type="primary"):
                key = hashlib.sha1(f"{route}{origin}{datetime.now()}".encode()).hexdigest()[:7].upper()
                st.session_state.managed_bookings.append({"booking_id": f"OPS-{key}", "route": route, "origin": origin, "passengers": passengers, "lead_days": lead, "trip_type": trip, "channel": channel, "status": "Confirmed", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
                st.success("Managed booking created in this session.")
    if st.session_state.managed_bookings:
        st.markdown("#### Managed booking queue")
        st.dataframe(pd.DataFrame(st.session_state.managed_bookings), hide_index=True, use_container_width=True)
    st.markdown("#### Dataset search records")
    table = data.copy()
    table.insert(0, "search_id", [f"SR-{i:06d}" for i in table.index])
    st.dataframe(table.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download filtered records (CSV)", data.to_csv(index=False).encode("utf-8"), "filtered_booking_records.csv", "text/csv")


def schedule() -> None:
    st.subheader("Flight schedule management")
    st.caption("This operational schedule is session-managed because the supplied dataset contains customer searches, not flight inventory.")
    with st.form("add_flight", clear_on_submit=True):
        a, b, c, d = st.columns(4)
        code = a.text_input("Flight number", "SN-900")
        route = b.text_input("Route code", "AKLDEL")
        departure = c.time_input("Departure time")
        capacity = d.number_input("Seat capacity", 1, 600, 180)
        if st.form_submit_button("Add flight", type="primary"):
            st.session_state.flights.append({"flight": code.upper(), "route": route.upper(), "departure": departure.strftime("%H:%M"), "status": "Scheduled", "capacity": capacity})
            st.success(f"{code.upper()} added to the live session schedule.")
    flight_df = pd.DataFrame(st.session_state.flights)
    flight_df["departure"] = pd.to_datetime(flight_df.departure, format="%H:%M").dt.time.astype(str)
    st.data_editor(flight_df, use_container_width=True, hide_index=True, disabled=["flight"], key="flight_editor")
    st.info("Edits made in the grid are reviewable in the interface; use the form to register schedule entries.")


def predictor(data: pd.DataFrame) -> None:
    st.subheader("Real-time booking conversion predictor")
    st.caption("Scores a new search with a Random Forest trained on the project’s customer_booking.csv data.")
    with st.form("predict_form"):
        a, b, c = st.columns(3)
        values = {
            "num_passengers": a.number_input("Passengers", 1, 9, 1),
            "sales_channel": b.selectbox("Sales channel", sorted(data.sales_channel.unique())),
            "trip_type": c.selectbox("Trip type", sorted(data.trip_type.unique())),
            "purchase_lead": a.number_input("Purchase lead", 0, 867, 30),
            "length_of_stay": b.number_input("Length of stay", 0, 1000, 7),
            "flight_hour": c.slider("Flight hour", 0, 23, 10),
            "flight_day": a.selectbox("Flight day", sorted(data.flight_day.unique())),
            "route": b.selectbox("Route", sorted(data.route.unique())),
            "booking_origin": c.selectbox("Booking origin", sorted(data.booking_origin.unique())),
            "wants_extra_baggage": int(a.checkbox("Extra baggage")),
            "wants_preferred_seat": int(b.checkbox("Preferred seat")),
            "wants_in_flight_meals": int(c.checkbox("In-flight meals")),
            "flight_duration": a.number_input("Flight duration (hours)", 0.1, 20.0, 5.5),
        }
        submit = st.form_submit_button("Score booking", type="primary")
    if submit:
        probability = train_model(data).predict_proba(pd.DataFrame([values]))[0, 1]
        st.metric("Predicted booking-completion probability", f"{probability:.1%}")
        if probability >= .50: st.success("High-intent search: route to sales/retention workflow.")
        elif probability >= .25: st.warning("Moderate intent: consider a tailored ancillary offer.")
        else: st.info("Lower intent: retain through remarketing rather than immediate escalation.")


def analytics(data: pd.DataFrame) -> None:
    st.subheader("Analytics & model governance")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Model validation snapshot")
        m = load_metrics(); st.dataframe(m, hide_index=True, use_container_width=True)
        st.caption("Saved evaluation results from this project’s modelling workflow.")
    with b:
        st.markdown("#### Leading conversion signals")
        imp = load_importance().head(12).sort_values("Importance")
        st.bar_chart(imp.set_index("Feature"), color=PURPLE, horizontal=True)
    st.markdown("#### Channel and ancillary performance")
    performance = data.groupby("sales_channel").agg(searches=("booking_complete", "size"), conversion=("booking_complete", "mean")).sort_values("conversion", ascending=False)
    performance["conversion"] = performance.conversion.map(lambda x: f"{x:.1%}")
    st.dataframe(performance, use_container_width=True)


def quality(data: pd.DataFrame) -> None:
    st.subheader("Data quality monitor")
    summary = pd.DataFrame({"field": data.columns, "type": data.dtypes.astype(str).values, "missing": data.isna().sum().values, "unique_values": data.nunique().values})
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows checked", f"{len(data):,}")
    c2.metric("Missing values", f"{int(data.isna().sum().sum()):,}")
    c3.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.success("Quality rule status: required dataset is readable and dashboard-ready.")


def main() -> None:
    init_state()
    try:
        data = load_data()
    except FileNotFoundError:
        st.error("customer_booking.csv was not found beside app.py.")
        st.stop()
    page, filtered = sidebar(data)
    st.title("Overview" if page == "Overview" else page)
    st.markdown("<p class='caption-muted'>Flight booking management and conversion intelligence · refreshed on each interaction</p>", unsafe_allow_html=True)
    if filtered.empty:
        st.warning("The current controls produce no records. Adjust the filters in the sidebar.")
        st.stop()
    if page == "Overview": overview(filtered)
    elif page == "Bookings": bookings(filtered)
    elif page == "Flight schedule": schedule()
    elif page == "Conversion predictor": predictor(data)
    elif page == "Analytics & model": analytics(filtered)
    else: quality(filtered)


if __name__ == "__main__":
    main()
