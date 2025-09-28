import React from "react";
import {
  ClockIcon,
  CloudIcon,
  ArrowsRightLeftIcon,
} from "@heroicons/react/24/outline";
import Header from "./components/Header";
import KPIBox from "./components/KPIBox";
import MapView from "./components/MapView";
import NetworkGraph from "./components/NetworkGraph";
import ModelRecommendation from "./components/ModelRecommendation";
import DisruptionsCard from "./components/DisruptionsCard";
import SidePanel from "./components/SidePanel";
import { useAppStore } from "./context/AppContext";

function App() {
  const { sectionThroughput, weather, getAverageDelay } = useAppStore();
  const networkDelay = getAverageDelay();

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      <SidePanel />
      <div className="flex-1 p-4 overflow-auto">
        <Header />
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* --- CHANGES ARE HERE: Using `colorName` prop --- */}
          <KPIBox
            title="Average Network Delay"
            value={`${networkDelay} mins`}
            icon={ClockIcon}
            change="+2.1%"
            colorName="yellow"
          />
          <KPIBox
            title="Live Weather"
            value={`${weather}°C`}
            icon={CloudIcon}
            change="Partly Cloudy"
            colorName="blue"
          />
          <KPIBox
            title="Section Throughput"
            value={`${sectionThroughput} Trains/Hour`}
            icon={ArrowsRightLeftIcon}
            change="-0.5%"
            colorName="purple"
          />
        </div>
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <MapView />
          <NetworkGraph />
          <ModelRecommendation />
          <DisruptionsCard />
        </div>
      </div>
    </div>
  );
}

export default App;
