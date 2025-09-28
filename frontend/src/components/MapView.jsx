import React from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
  Tooltip,
} from "react-leaflet";

function MapView() {
  const stations = [
    { id: 1, name: "Jind Junction", position: [29.3193, 76.3138] },
    { id: 2, name: "Sangrur", position: [30.2541, 75.8383] },
    { id: 3, name: "Bathinda Junction", position: [30.211, 74.9455] },
  ];

  const routePositions = stations.map((s) => s.position);

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-[400px]">
      <MapContainer
        center={[30.0, 75.5]}
        zoom={8}
        className="h-full rounded-lg"
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        />
        <TileLayer
          url="https://a.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openrailwaymap.org/">OpenRailwayMap</a>'
        />

        {stations.map((station) => (
          <Marker key={station.id} position={station.position}>
            {/* --- CHANGE #1: ADDING PERMANENT LABELS --- */}
            <Tooltip
              permanent={true}
              direction="right"
              offset={[10, 0]}
              className="leaflet-tooltip-label" // Custom class for styling
            >
              {station.name}
            </Tooltip>
          </Marker>
        ))}

        {/* --- CHANGE #2: ADDING DOTTED LINE STYLE --- */}
        <Polyline
          pathOptions={{ color: "#f59e0b", weight: 5, dashArray: "1, 10" }}
          positions={routePositions}
        />
      </MapContainer>
    </div>
  );
}

export default MapView;
