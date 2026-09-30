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
/* Application shell */
.stApp, [data-testid="stAppViewContainer"] {background:linear-gradient(135deg,#f7f5ff 0%,#f7f9fc 46%,#fffaf0 100%);color:#17233c}
.block-container{padding:2rem 2.5rem 3rem;max-width:1500px}
[data-testid="stHeader"]{background:rgba(247,248,252,.84);border-bottom:1px solid #eaecf3}

/* Typography: force contrast even when the browser previously used dark mode */
[data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3,[data-testid="stMain"] h4{color:#17233c!important;letter-spacing:-.035em}
[data-testid="stMain"] h1{font-size:2.45rem!important;font-weight:800!important;margin-bottom:.15rem!important}
[data-testid="stMain"] h2{font-size:1.55rem!important;margin-top:1.4rem!important}
[data-testid="stMain"] p,[data-testid="stMain"] span,[data-testid="stMain"] label,[data-testid="stMain"] [data-testid="stWidgetLabel"] p{color:#52627d!important}
.caption-muted{color:#6d7890!important;font-size:.94rem;letter-spacing:.01em}

/* Form controls */
[data-testid="stMain"] [data-baseweb="input"]>div,[data-testid="stMain"] [data-baseweb="select"]>div,[data-testid="stMain"] input,[data-testid="stMain"] textarea{background:#fff!important;border-color:#dbe1ec!important;color:#17233c!important;border-radius:10px!important}
[data-testid="stMain"] [data-baseweb="input"] input,[data-testid="stMain"] [data-baseweb="select"] span{color:#17233c!important;-webkit-text-fill-color:#17233c!important}
[data-testid="stMain"] [data-baseweb="select"] svg{fill:#55627a!important}
[data-testid="stMain"] [data-baseweb="checkbox"]+div,[data-testid="stMain"] [data-baseweb="checkbox"]+div p{color:#34425d!important}
[data-testid="stMain"] [data-testid="stNumberInput"] button{background:#f3f0ff!important;color:#5b3cc4!important}
[data-testid="stMain"] [data-testid="stSlider"] [data-testid="stThumbValue"]{color:#fff!important}
[data-testid="stMain"] .stButton>button,[data-testid="stMain"] [data-testid="stFormSubmitButton"] button{background:linear-gradient(135deg,#6c4ce5,#8a5cf6)!important;color:#fff!important;border:0!important;border-radius:10px!important;font-weight:700!important;padding:.58rem 1rem!important;box-shadow:0 7px 15px rgba(108,76,229,.20)!important}
[data-testid="stMain"] .stButton>button:hover,[data-testid="stMain"] [data-testid="stFormSubmitButton"] button:hover{background:linear-gradient(135deg,#5534c6,#7042e4)!important;color:#fff!important}

/* Cards, data and status surfaces */
.metric-card{background:linear-gradient(145deg,#fff,#fbfbff);border-radius:18px;padding:19px 20px;border:1px solid #e6e9f2;box-shadow:0 10px 25px rgba(31,35,72,.065);min-height:130px;transition:transform .2s}
.metric-card:hover{transform:translateY(-2px)}
.metric-label{font-size:.84rem;color:#667085!important;margin-bottom:10px;font-weight:600}.metric-value{font-size:1.8rem;font-weight:800;color:#17233c!important}.metric-note{font-size:.78rem;color:#079669!important;margin-top:9px;font-weight:600}
.section-card{background:#fff;border:1px solid #e9eaf1;border-radius:18px;padding:16px 18px;margin-top:8px}
[data-testid="stDataFrame"],[data-testid="stDataEditor"]{border:1px solid #e0e5ef;border-radius:14px;overflow:hidden;box-shadow:0 4px 16px rgba(40,45,80,.04)}
[data-testid="stAlert"]{border-radius:12px!important;border:0!important}
[data-testid="stMain"] [data-testid="stProgress"]>div>div>div{background:linear-gradient(90deg,#6c4ce5,#e7c866)!important}

/* Sidebar uses its own dark brand palette */
[data-testid='stSidebar']{background:linear-gradient(180deg,#251451 0%,#1b103f 100%);border-right:1px solid rgba(255,255,255,.08)}
[data-testid='stSidebar'] *{color:#f7f4ff!important}
[data-testid='stSidebar'] [data-baseweb="select"]>div,[data-testid='stSidebar'] [data-baseweb="input"]>div{background:#130b30!important;border-color:#493879!important}
[data-testid='stSidebar'] [data-baseweb="tag"]{background:#6c4ce5!important}
[data-testid='stSidebar'] [data-testid="stRadio"] label{padding:4px 7px;border-radius:8px}
[data-testid='stSidebar'] [data-testid="stRadio"] label:hover{background:rgba(255,255,255,.10)}
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
    operations = flight_table()
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total booking searches", f"{len(data):,}", "Live filtered dataset", PURPLE)
    with c2: metric_card("Completed bookings", f"{len(completed):,}", f"{conversion:.1%} conversion rate", GREEN)
    with c3: metric_card("Scheduled flights", str(len(operations)), f"{int(operations.available_seats.sum()):,} seats available", BLUE)
    with c4: metric_card("Managed booking revenue", money(sum(b["amount"] for b in st.session_state.managed_bookings)), "Confirmed in this session", GOLD)

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

    st.subheader("Demand & conversion intelligence")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### Booking funnel")
        funnel = pd.Series({"Searches": len(data), "Completed": int(data.booking_complete.sum()), "Abandoned": int((1 - data.booking_complete).sum())})
        st.bar_chart(funnel, color=PURPLE, height=240)
        st.caption("Conversion: {:.1%} of filtered searches.".format(conversion))
    with b:
        st.markdown("#### Departure-hour intent")
        hourly = data.groupby("flight_hour").booking_complete.mean().mul(100)
        st.area_chart(hourly, color=GREEN, height=240)
        st.caption("Booking completion rate by planned departure hour.")
    with c:
        st.markdown("#### Group size demand")
        groups = data.num_passengers.value_counts().sort_index()
        st.bar_chart(groups, color=GOLD, height=240)
        st.caption("Customer search volume by number of passengers.")

    st.subheader("Live flight readiness")
    readiness = operations[["flight", "available_seats", "capacity", "status"]].copy().set_index("flight")
    st.bar_chart(readiness[["available_seats", "capacity"]], color=[PURPLE, "#D9DDEF"], height=220)
    st.caption("Available capacity compared with total configured capacity for the session-managed schedule.")


def flight_table() -> pd.DataFrame:
    flights = pd.DataFrame(st.session_state.flights).copy()
    booked = pd.DataFrame(st.session_state.managed_bookings)
    seat_counts = booked.groupby("flight").size() if not booked.empty else pd.Series(dtype=int)
    flights["arrival"] = flights.apply(lambda r: r.departure + timedelta(hours=float(r.duration_h)), axis=1)
    flights["booked"] = flights.flight.map(seat_counts).fillna(0).astype(int)
    flights["available_seats"] = flights.capacity - flights.booked
    flights["load_factor"] = (flights.booked / flights.capacity * 100).round(1)
    return flights


def seat_labels(capacity: int) -> list[str]:
    rows = min((capacity + 5) // 6, 45)
    return [f"{row}{letter}" for row in range(1, rows + 1) for letter in "ABCDEF"][:capacity]


def bookings(data: pd.DataFrame) -> None:
    st.subheader("Bookings & seat assignment")
    st.caption("Create confirmed bookings with fare calculation, passenger seats, and a payment record.")
    flights = flight_table()
    choices = {f"{r.flight} · {r.route} · {r.departure:%d %b %H:%M} · {money(r.fare)}": r.flight for _, r in flights.iterrows()}
    chosen_label = st.selectbox("Select a scheduled flight", list(choices))
    chosen = flights.loc[flights.flight.eq(choices[chosen_label])].iloc[0]
    existing = pd.DataFrame(st.session_state.managed_bookings)
    occupied = set(existing.loc[existing.flight.eq(chosen.flight), "seats"].explode().dropna()) if not existing.empty else set()
    open_seats = [seat for seat in seat_labels(int(chosen.capacity)) if seat not in occupied]
    st.markdown(f"**{chosen.airline} · Gate {chosen.gate}**  \\  Departure: **{chosen.departure:%d %b %Y, %H:%M}**  \\  Arrival: **{chosen.arrival:%d %b %Y, %H:%M}**  \\  Journey: **{chosen.duration_h:.1f} hours**")
    st.progress(min(float(chosen.booked / chosen.capacity), 1.0), text=f"{len(open_seats)} of {chosen.capacity} seats available · {chosen.load_factor}% occupied")
    with st.form("new_booking", clear_on_submit=True):
        a, b, c = st.columns(3)
        name = a.text_input("Lead passenger name", "")
        email = b.text_input("Email", "")
        passengers = c.number_input("Passengers", 1, min(9, len(open_seats)), 1)
        selected_seats = st.multiselect("Choose seats", open_seats, help="Select one seat per passenger. You can select up to the number of passengers entered above.")
        st.caption("Seat selection is flexible while you choose; booking confirmation requires exactly one seat for every passenger.")
        x, y, z = st.columns(3)
        baggage = x.checkbox("Extra baggage (+$45 per passenger)")
        preferred = y.checkbox("Preferred seat (+$25 per passenger)")
        meal = z.checkbox("In-flight meal (+$18 per passenger)")
        amount = float(chosen.fare) * passengers + passengers * (45 * baggage + 25 * preferred + 18 * meal)
        st.info(f"Estimated total: {money(amount)} · Base fare: {money(float(chosen.fare))} per passenger")
        submitted = st.form_submit_button("Confirm booking & payment", type="primary")
        if submitted:
            if not name.strip() or "@" not in email:
                st.error("Enter a lead passenger name and a valid email address.")
            elif len(selected_seats) != passengers:
                st.error("Select exactly one seat for each passenger.")
            else:
                key = hashlib.sha1(f"{chosen.flight}{email}{datetime.now()}".encode()).hexdigest()[:7].upper()
                booking = {"booking_id": f"BK-{key}", "flight": chosen.flight, "route": chosen.route, "passenger": name, "email": email, "passengers": int(passengers), "seats": selected_seats, "departure": chosen.departure.strftime("%Y-%m-%d %H:%M"), "arrival": chosen.arrival.strftime("%Y-%m-%d %H:%M"), "journey_hours": chosen.duration_h, "amount": round(amount, 2), "status": "Confirmed", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
                st.session_state.managed_bookings.append(booking)
                st.session_state.payments.append({"payment_id": f"PAY-{key}", "booking_id": booking["booking_id"], "passenger": name, "amount": booking["amount"], "method": "Card", "status": "Paid", "paid_at": booking["created_at"]})
                st.session_state.activity.insert(0, {"time": "Just now", "event": f"{booking['booking_id']} confirmed on {chosen.flight}", "type": "Booking"})
                st.success(f"Booking {booking['booking_id']} confirmed. Seats: {', '.join(selected_seats)}")
    st.markdown("#### Seat map")
    visual = [f"🟥 {s}" if s in occupied else f"🟨 {s}" for s in seat_labels(int(chosen.capacity))[:48]]
    st.caption("🟥 Occupied · 🟨 Available · Use the selector above to reserve available seats.")
    st.markdown("<br>".join(" &nbsp; ".join(visual[i:i+6]) for i in range(0, len(visual), 6)), unsafe_allow_html=True)
    if st.session_state.managed_bookings:
        st.markdown("#### Confirmed booking queue")
        st.dataframe(pd.DataFrame(st.session_state.managed_bookings), hide_index=True, use_container_width=True, column_config={"amount": st.column_config.NumberColumn("Amount", format="$%.2f")})
    st.download_button("Download filtered historical records (CSV)", data.to_csv(index=False).encode("utf-8"), "filtered_booking_records.csv", "text/csv")


def schedule() -> None:
    st.subheader("Flights & schedule")
    st.caption("Build a live flight timetable with departure, arrival, journey time, capacity, fare, gate, and service status.")
    with st.form("add_flight", clear_on_submit=True):
        a, b, c, d = st.columns(4)
        code = a.text_input("Flight number", "SN-900")
        airline = b.text_input("Airline", "SkyNest Air")
        route = c.text_input("Route code (e.g. CDGJFK)", "AKLDEL")
        flight_date = d.date_input("Departure date", date.today() + timedelta(days=1))
        departure_time = a.time_input("Departure time", datetime.strptime("09:00", "%H:%M").time())
        duration = b.number_input("Journey time (hours)", 0.5, 24.0, 5.5, 0.1)
        fare = c.number_input("Base fare (USD)", 10.0, 10000.0, 450.0, 5.0)
        capacity = d.number_input("Seat capacity", 6, 270, 180, 6)
        gate = a.text_input("Gate", "A01")
        status = b.selectbox("Flight status", ["Scheduled", "On time", "Boarding", "Delayed", "Cancelled"])
        if st.form_submit_button("Add flight to schedule", type="primary"):
            departure = datetime.combine(flight_date, departure_time)
            st.session_state.flights.append({"flight": code.upper(), "airline": airline, "route": route.upper().replace("-", ""), "departure": departure, "duration_h": float(duration), "fare": float(fare), "capacity": int(capacity), "gate": gate.upper(), "status": status})
            st.session_state.activity.insert(0, {"time": "Just now", "event": f"Flight {code.upper()} added to schedule", "type": "Schedule"})
            st.success(f"{code.upper()} added. Arrival: {(departure + timedelta(hours=float(duration))):%d %b %Y, %H:%M}")
    display = flight_table()[["flight", "airline", "route", "departure", "arrival", "duration_h", "gate", "fare", "capacity", "booked", "available_seats", "load_factor", "status"]].copy()
    display = display.rename(columns={"duration_h": "journey_hours", "fare": "base_fare"})
    st.dataframe(display, use_container_width=True, hide_index=True, column_config={"departure": st.column_config.DatetimeColumn("Departure", format="DD MMM YYYY, HH:mm"), "arrival": st.column_config.DatetimeColumn("Arrival", format="DD MMM YYYY, HH:mm"), "base_fare": st.column_config.NumberColumn("Base fare", format="$%.2f"), "load_factor": st.column_config.ProgressColumn("Load factor", min_value=0, max_value=100, format="%.1f%%")})


def payments() -> None:
    st.subheader("Payments & revenue")
    records = pd.DataFrame(st.session_state.payments)
    total = float(records.amount.sum()) if not records.empty else 0.0
    a, b, c = st.columns(3)
    a.metric("Collected revenue", money(total))
    b.metric("Paid transactions", len(records))
    c.metric("Average booking value", money(total / len(records)) if len(records) else "$0")
    if records.empty:
        st.info("Payments are created automatically when a seat booking is confirmed.")
    else:
        st.dataframe(records, hide_index=True, use_container_width=True, column_config={"amount": st.column_config.NumberColumn("Amount", format="$%.2f")})


def tracking() -> None:
    st.subheader("Flight tracking & activity")
    flights = flight_table()
    for _, f in flights.iterrows():
        status_color = {"Boarding": "🟢", "On time": "🟢", "Scheduled": "🔵", "Delayed": "🟠", "Cancelled": "🔴"}.get(f.status, "⚪")
        st.markdown(f"### {status_color} {f.flight} · {f.route}  ")
        st.caption(f"{f.airline} | Gate {f.gate} | Departs {f.departure:%d %b %H:%M} | Arrives {f.arrival:%d %b %H:%M} | {f.available_seats} seats available")
    st.markdown("#### Recent operational activity")
    st.dataframe(pd.DataFrame(st.session_state.activity), hide_index=True, use_container_width=True)


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
    elif page == "Bookings & seats": bookings(filtered)
    elif page == "Flights & schedule": schedule()
    elif page == "Payments": payments()
    elif page == "Flight tracking": tracking()
    elif page == "Conversion predictor": predictor(data)
    elif page == "Analytics & model": analytics(filtered)
    else: quality(filtered)


if __name__ == "__main__":
    main()
