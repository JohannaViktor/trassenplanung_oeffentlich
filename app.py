import streamlit as st
import geopandas as gpd
import pandas as pd
import networkx as nx
import folium
import osmnx as ox
import requests

from io import BytesIO
from shapely.geometry import Point
from shapely.ops import unary_union, linemerge
from streamlit_folium import st_folium
from scipy.spatial import cKDTree


# =====================================================
# CONFIG
# =====================================================

TARGET_CRS = "EPSG:25833"
DISPLAY_CRS = "EPSG:4326"

WFS_URL = "https://gdi.berlin.de/services/wfs/detailnetz"

LAYER_KANTEN = "detailnetz:c_strassenabschnitte"
LAYER_KNOTEN = "detailnetz:a_verbindungspunkte"


# =====================================================
# SESSION STATE
# =====================================================

if "routing_result" not in st.session_state:
    st.session_state.routing_result = None


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("HS Routing MVP")

bbox_buffer = st.sidebar.slider("BBOX Buffer (m)", 500, 5000, 1500, 100)
corridor_buffer = st.sidebar.slider("Korridor Buffer (m)", 50, 1000, 200, 50)
n_routes = st.sidebar.slider("Anzahl Routen", 1, 10, 5, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Kostenlogik")

weight_ii = st.sidebar.slider("Bonus Hauptachsen (II)", 0.1, 2.0, 0.65, 0.05)
weight_iv = st.sidebar.slider("Gewicht Klasse IV", 0.1, 3.0, 0.9, 0.05)
weight_v = st.sidebar.slider("Penalty Klasse V", 0.1, 5.0, 1.5, 0.1)
weight_fuwe = st.sidebar.slider("Penalty Fußwege", 1.0, 10.0, 4.0, 0.5)
weight_park = st.sidebar.slider("Penalty Parks", 1.0, 15.0, 6.0, 0.5)


# =====================================================
# FUNCTIONS
# =====================================================

def make_bbox_from_points(p1, p2, buffer_m=1000):
    xmin = min(p1.x, p2.x) - buffer_m
    xmax = max(p1.x, p2.x) + buffer_m
    ymin = min(p1.y, p2.y) - buffer_m
    ymax = max(p1.y, p2.y) + buffer_m
    return xmin, ymin, xmax, ymax


@st.cache_data(show_spinner=False)
def load_wfs_bbox(typename, xmin, ymin, xmax, ymax):
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typenames": typename,
        "outputFormat": "application/json",
        "srsName": TARGET_CRS,
        "bbox": f"{xmin},{ymin},{xmax},{ymax},{TARGET_CRS}",
    }

    r = requests.get(WFS_URL, params=params, timeout=90)
    r.raise_for_status()

    return gpd.read_file(BytesIO(r.content))


@st.cache_data(show_spinner=False)
def load_substations():
    bbox = (13.30, 52.47, 13.50, 52.57)
    #        west   south  east   north

    tags = {"power": "substation"}

    substations = ox.features_from_bbox(bbox, tags=tags)
    substations = substations[substations.geometry.notnull()].copy()

    substations = substations[
        substations["substation"].isin(["transmission", "distribution"])
    ].copy()

    substations["geometry"] = substations.geometry.representative_point()
    substations = substations.set_crs(DISPLAY_CRS, allow_override=True)

    return substations


def hs_edge_cost(row):
    cost = row.geometry.length

    sk1 = str(row.get("strassenklasse1", "")).upper()
    sk = str(row.get("strassenklasse", "")).upper()
    sk2 = str(row.get("strassenklasse2", "")).upper()

    if sk1 == "II":
        cost *= weight_ii
    elif sk1 == "IV":
        cost *= weight_iv
    elif sk1 == "V":
        cost *= weight_v

    if sk == "F":
        cost *= 3.0
    elif sk == "P":
        cost *= 1.8

    if sk2 == "FUWE":
        cost *= weight_fuwe
    elif sk2 == "PARK":
        cost *= weight_park

    return max(cost, 1.0)


def build_graph_by_nearest_nodes(kanten_gdf, knoten_gdf, max_dist=2.0):
    G = nx.Graph()

    node_ids = knoten_gdf["dnkn__sdatenid"].astype(str).tolist()
    node_points = list(knoten_gdf.geometry)

    coords = [(p.x, p.y) for p in node_points]
    tree = cKDTree(coords)

    for node_id, p in zip(node_ids, node_points):
        G.add_node(node_id, geometry=p, x=p.x, y=p.y)

    matched = 0
    skipped = 0

    for idx, row in kanten_gdf.iterrows():
        geom = row.geometry

        if geom is None:
            skipped += 1
            continue

        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]

        for line in lines:
            if len(line.coords) < 2:
                skipped += 1
                continue

            start = Point(line.coords[0])
            end = Point(line.coords[-1])

            start_dist, start_idx = tree.query([start.x, start.y])
            end_dist, end_idx = tree.query([end.x, end.y])

            if start_dist > max_dist or end_dist > max_dist:
                skipped += 1
                continue

            u = node_ids[start_idx]
            v = node_ids[end_idx]

            if u == v:
                skipped += 1
                continue

            length = line.length
            cost = hs_edge_cost(row)

            G.add_edge(
                u,
                v,
                geometry=line,
                length=length,
                weight=cost,
                hs_cost=cost,
                strassenklasse1=row.get("strassenklasse1"),
                strassenklasse=row.get("strassenklasse"),
                strassenklasse2=row.get("strassenklasse2"),
                source_index=idx,
            )

            matched += 1

    return G, matched, skipped


def nearest_node(G, point):
    best_node = None
    best_dist = float("inf")

    for node, data in G.nodes(data=True):
        dx = data["x"] - point.x
        dy = data["y"] - point.y
        d = dx * dx + dy * dy

        if d < best_dist:
            best_dist = d
            best_node = node

    return best_node, best_dist ** 0.5


def route_to_geometry(G, path):
    geoms = []

    for u, v in zip(path[:-1], path[1:]):
        geoms.append(G[u][v]["geometry"])

    return linemerge(unary_union(geoms))


def make_display_name(row):
    for col in ["name", "operator", "ref", "substation"]:
        value = row.get(col)
        if pd.notna(value):
            return str(value)
    return "Umspannwerk"


# =====================================================
# UI
# =====================================================

st.title("Hochspannungs-Korridor MVP")

with st.expander("Kurze Einführung & Routinglogik", expanded=False):

    st.markdown("""
### Was macht die Anwendung?

Diese Anwendung berechnet erste plausible Hochspannungs-
Trassenkorridore zwischen Umspannwerken in Berlin.

Die günstigste Route wird rot dargestellt.

### Was tun?

1. Start- und Ziel-Umspannwerk wählen
2. Kostenlogik über die Slider anpassen
3. „Routing berechnen“ klicken
4. Routen vergleichen

### Wie wird aktuell geroutet?

Bevorzugt werden:
- Hauptstraßen und Magistralen
- große Infrastrukturachsen
- gut zugängliche Straßenräume

Vermieden werden:
- Fußwege
- Parks/Grünflächen
- kleinere Wohnstraßen

### Bedeutung wichtiger Slider

- **Bonus Hauptachsen (II)**  
  kleinere Werte = Hauptstraßen stärker bevorzugt

- **Penalty Fußwege / Parks**  
  größere Werte = stärkere Vermeidung

- **BBOX Buffer**  
  Größe des betrachteten Netzausschnitts

- **Korridor Buffer**  
  Breite des visualisierten Korridors

### Noch nicht berücksichtigt

- bestehende Stromkabel
- thermische Wechselwirkungen
- reale Netzkapazitäten
- Genehmigungsdetails
- Lastfluss- und Redundanzlogik
""")

substations = load_substations()

substations_display = substations.copy()
substations_display["display_name"] = substations_display.apply(make_display_name, axis=1)

station_names = sorted(substations_display["display_name"].unique().tolist())

start_station = st.selectbox("Start-Umspannwerk", station_names, index=0)
ziel_station = st.selectbox("Ziel-Umspannwerk", station_names, index=min(1, len(station_names) - 1))


# =====================================================
# ROUTING BUTTON
# =====================================================

if st.button("Routing berechnen"):

    start_row = substations_display[
        substations_display["display_name"] == start_station
    ].iloc[0]

    ziel_row = substations_display[
        substations_display["display_name"] == ziel_station
    ].iloc[0]

    start_point_wgs = start_row.geometry
    ziel_point_wgs = ziel_row.geometry

    start_point = gpd.GeoSeries([start_point_wgs], crs=DISPLAY_CRS).to_crs(TARGET_CRS).iloc[0]
    ziel_point = gpd.GeoSeries([ziel_point_wgs], crs=DISPLAY_CRS).to_crs(TARGET_CRS).iloc[0]

    xmin, ymin, xmax, ymax = make_bbox_from_points(
        start_point,
        ziel_point,
        buffer_m=bbox_buffer,
    )

    with st.spinner("Lade Detailnetz..."):
        kanten_small = load_wfs_bbox(LAYER_KANTEN, xmin, ymin, xmax, ymax)
        knoten_small = load_wfs_bbox(LAYER_KNOTEN, xmin, ymin, xmax, ymax)

    kanten_small = kanten_small.to_crs(TARGET_CRS)
    knoten_small = knoten_small.to_crs(TARGET_CRS)

    with st.spinner("Baue Graph..."):
        G, matched_edges, skipped_edges = build_graph_by_nearest_nodes(
            kanten_small,
            knoten_small,
            max_dist=2.0,
        )

    if G.number_of_edges() == 0:
        st.error("Graph enthält keine Kanten. Erhöhe den BBOX-Buffer.")
        st.stop()

    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    G_main = G.subgraph(largest_component).copy()

    start_node, start_dist = nearest_node(G_main, start_point)
    ziel_node, ziel_dist = nearest_node(G_main, ziel_point)

    with st.spinner("Berechne Routen..."):
        routes = []

        try:
            generator = nx.shortest_simple_paths(
                G_main,
                source=start_node,
                target=ziel_node,
                weight="weight",
            )

            for path in generator:
                routes.append(path)
                if len(routes) >= n_routes:
                    break

        except nx.NetworkXNoPath:
            st.error("Keine Route gefunden. Erhöhe den BBOX-Buffer oder prüfe die Start-/Zielpunkte.")
            st.stop()

    route_geoms = [route_to_geometry(G_main, r) for r in routes]

    routes_gdf = gpd.GeoDataFrame(
        {
            "route_id": range(1, len(routes) + 1),
            "cost": [
                round(nx.path_weight(G_main, r, weight="weight"), 1)
                for r in routes
            ],
            "geometry": route_geoms,
        },
        crs=TARGET_CRS,
    )

    routes_gdf["length_m"] = routes_gdf.geometry.length.round(1)
    routes_gdf["length_km"] = (routes_gdf["length_m"] / 1000).round(2)

    corridor_geom = unary_union([
        geom.buffer(corridor_buffer)
        for geom in route_geoms
    ])

    corridor_gdf = gpd.GeoDataFrame(
        {
            "name": ["HS-Korridor"],
            "buffer_m": [corridor_buffer],
            "geometry": [corridor_geom],
        },
        crs=TARGET_CRS,
    )

    # =================================================
    # FOLIUM MAP
    # =================================================

    center_lat = (start_point_wgs.y + ziel_point_wgs.y) / 2
    center_lon = (start_point_wgs.x + ziel_point_wgs.x) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
    )

    kanten_wgs = kanten_small.to_crs(DISPLAY_CRS)[["geometry"]]

    folium.GeoJson(
        kanten_wgs,
        name="Detailnetz",
        style_function=lambda x: {
            "color": "gray",
            "weight": 1,
            "opacity": 0.2,
        },
    ).add_to(m)

    corridor_wgs = corridor_gdf.to_crs(DISPLAY_CRS)[["name", "buffer_m", "geometry"]]

    folium.GeoJson(
        corridor_wgs,
        name="HS-Korridor",
        style_function=lambda x: {
            "fillColor": "orange",
            "color": "orange",
            "weight": 2,
            "fillOpacity": 0.15,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "buffer_m"],
            aliases=["Name", "Puffer m"],
        ),
    ).add_to(m)

    routes_wgs = routes_gdf.to_crs(DISPLAY_CRS)[
        ["route_id", "length_m", "length_km", "cost", "geometry"]
    ]

    route_colors = [
        "#d7191c",  # Route 1 rot
        "#fdae61",  # Route 2 orange
        "#ffff80",  # Route 3 gelb
        "#a6d96a",  # Route 4 hellgrün
        "#2b83ba",  # Route 5 blau
    ]

    for _, row in routes_wgs.sort_values("route_id", ascending=False).iterrows():
        route_id = int(row["route_id"])
        color = route_colors[(route_id - 1) % len(route_colors)]

        route_layer = gpd.GeoDataFrame(
            {
                "route_id": [route_id],
                "length_m": [row["length_m"]],
                "length_km": [row["length_km"]],
                "cost": [row["cost"]],
                "geometry": [row.geometry],
            },
            crs=DISPLAY_CRS,
        )

        folium.GeoJson(
            route_layer,
            name=f"Route {route_id}",
            style_function=lambda x, color=color, route_id=route_id: {
                "color": color,
                "weight": 8 if route_id == 1 else 4,
                "opacity": 1.0 if route_id == 1 else 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["route_id", "length_km", "cost"],
                aliases=["Route", "Länge km", "Kosten"],
            ),
        ).add_to(m)

    substations_wgs = substations_display.to_crs(DISPLAY_CRS)

    for _, row in substations_wgs.iterrows():
        geom = row.geometry
        sub_type = row.get("substation", "unknown")

        if sub_type == "transmission":
            fill_color = "red"
            radius = 9
        elif sub_type == "distribution":
            fill_color = "yellow"
            radius = 7
        else:
            fill_color = "gray"
            radius = 6

        display_name = row["display_name"]

        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=radius,
            color="black",
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=1,
            popup=(
                f"<b>{display_name}</b><br>"
                f"Typ: {sub_type}<br>"
                f"Lat: {geom.y:.6f}<br>"
                f"Lon: {geom.x:.6f}"
            ),
            tooltip=f"{display_name} ({sub_type})",
        ).add_to(m)

    folium.LayerControl().add_to(m)

    # =================================================
    # SAVE RESULTS TO SESSION STATE
    # =================================================

    st.session_state.routing_result = {
        #"map": m,
        "kanten_small": kanten_small,
        "routes_gdf": routes_gdf,
        "corridor_gdf": corridor_gdf,
        "substations_display": substations_display,
        "kanten_count": len(kanten_small),
        "knoten_count": len(knoten_small),
        "matched_edges": matched_edges,
        "skipped_edges": skipped_edges,
        "n_components": nx.number_connected_components(G),
        "main_component_nodes": G_main.number_of_nodes(),
        "main_component_edges": G_main.number_of_edges(),
        "start_station": start_station,
        "ziel_station": ziel_station,
        "start_dist": start_dist,
        "ziel_dist": ziel_dist,
    }


# =====================================================
# PERSISTENT RESULTS DISPLAY
# =====================================================

result = st.session_state.routing_result

if result is not None:

    

    st.subheader("1. Karte")

    kanten_small = result["kanten_small"]
    routes_gdf = result["routes_gdf"]
    corridor_gdf = result["corridor_gdf"]
    substations_display = result["substations_display"]

    routes_wgs = routes_gdf.to_crs(DISPLAY_CRS)
    corridor_wgs = corridor_gdf.to_crs(DISPLAY_CRS)[["name", "buffer_m", "geometry"]]
    kanten_wgs = kanten_small.to_crs(DISPLAY_CRS)[["geometry"]]
    substations_wgs = substations_display.to_crs(DISPLAY_CRS)

    center = routes_wgs.geometry.iloc[0].centroid

    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    folium.GeoJson(
        kanten_wgs,
        name="Detailnetz",
        style_function=lambda x: {
            "color": "gray",
            "weight": 1,
            "opacity": 0.2,
        },
    ).add_to(m)

    folium.GeoJson(
        corridor_wgs,
        name="HS-Korridor",
        style_function=lambda x: {
            "fillColor": "orange",
            "color": "orange",
            "weight": 2,
            "fillOpacity": 0.15,
        },
    ).add_to(m)

    route_colors = [
        "#d7191c",
        "#2b83ba",
        "#2b83ba",
        "#2b83ba",
        "#2b83ba",
    ]

    for _, row in routes_wgs.sort_values("route_id", ascending=False).iterrows():
        route_id = int(row["route_id"])
        color = route_colors[(route_id - 1) % len(route_colors)]

        route_layer = gpd.GeoDataFrame(
            {
                "route_id": [route_id],
                "length_km": [row["length_km"]],
                "cost": [row["cost"]],
                "geometry": [row.geometry],
            },
            crs=DISPLAY_CRS,
        )

        folium.GeoJson(
            route_layer,
            name=f"Route {route_id}",
            style_function=lambda x, color=color, route_id=route_id: {
                "color": color,
                "weight": 8 if route_id == 1 else 4,
                "opacity": 1.0 if route_id == 1 else 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["route_id", "length_km", "cost"],
                aliases=["Route", "Länge km", "Kosten"],
            ),
        ).add_to(m)

    for _, row in substations_wgs.iterrows():
        geom = row.geometry
        sub_type = row.get("substation", "unknown")

        fill_color = "red" if sub_type == "transmission" else "yellow"

        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=9 if sub_type == "transmission" else 7,
            color="black",
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=1,
            popup=f"{row['display_name']}<br>{sub_type}",
        ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(
        m,
        width=1400,
        height=850,
        returned_objects=[]
    )

    st.subheader("2. Routing-Status")

    st.caption(
        f"Route von {result['start_station']} → {result['ziel_station']}"
    )

    status_df = pd.DataFrame(
        {
            "Geladene Kanten": [result["kanten_count"]],
            "Geladene Knoten": [result["knoten_count"]],
            "Komponenten": [result["n_components"]],
            "Gematchte Kanten": [result["matched_edges"]],
            "Übersprungene Kanten": [result["skipped_edges"]],
            "HK Kanten": [result["main_component_edges"]],
            "Start Dist (m)": [round(result["start_dist"], 1)],
            "Ziel Dist (m)": [round(result["ziel_dist"], 1)],
        }
    )

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("3. Routenvergleich")

    st.dataframe(
        result["routes_gdf"][
            ["route_id", "length_m", "length_km", "cost"]
        ],
        use_container_width=True,
    )

else:
    st.info("Wähle Start/Ziel und klicke auf „Routing berechnen“.")