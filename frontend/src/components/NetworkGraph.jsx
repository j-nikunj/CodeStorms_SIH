import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
} from "recharts";
import { useAppStore } from "../context/AppContext";

function NetworkGraph() {
  const { baseGraphData, simulatedGraphData } = useAppStore();

  // Combine base data with simulated data if it exists
  const displayData = baseGraphData.map((base, index) => ({
    ...base,
    ...(simulatedGraphData
      ? {
          "Train A (Simulated)":
            simulatedGraphData[index]["Train A (Simulated)"],
          "Train B (Simulated)":
            simulatedGraphData[index]["Train B (Simulated)"],
        }
      : {}),
  }));

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-[400px]">
      <h2 className="text-lg font-semibold mb-2">
        Network Congestion Analysis
      </h2>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={displayData}>
          <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} />
          <XAxis
            dataKey="time"
            stroke="#9ca3af"
            angle={-45}
            textAnchor="end"
            height={50}
            label={{
              value: "Time",
              position: "insideBottom",
              offset: -5,
              fill: "#9ca3af",
            }}
          />
          <YAxis
            stroke="#9ca3af"
            label={{
              value: "Delay (mins)",
              angle: -90,
              position: "insideLeft",
              fill: "#9ca3af",
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
            }}
          />
          <Legend />

          {/* --- BASE SCENARIO (The Problem) --- */}
          <Line
            name="Train A (Current)"
            type="monotone"
            dataKey="Train A Delay"
            stroke="#8884d8"
            strokeWidth={2}
          />
          <Line
            name="Train B (Current)"
            type="monotone"
            dataKey="Train B Delay"
            stroke="#82ca9d"
            strokeWidth={2}
          />

          {/* --- SIMULATED SCENARIO (The Solution) --- */}
          {simulatedGraphData && (
            <Line
              name="Train A (Simulated)"
              type="monotone"
              dataKey="Train A (Simulated)"
              stroke="#c184f5"
              strokeWidth={2}
              strokeDasharray="5 5"
            />
          )}
          {simulatedGraphData && (
            <Line
              name="Train B (Simulated)"
              type="monotone"
              dataKey="Train B (Simulated)"
              stroke="#a1e6c3"
              strokeWidth={2}
              strokeDasharray="5 5"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default NetworkGraph;
