import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx"; // Make sure this points to .jsx
import "./index.css";
import "leaflet/dist/leaflet.css"; // Add this line for the map styles

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
