import React, { useState } from "react";
import { useAppStore } from "../context/AppContext";

function SidePanel() {
  const { runSimulation, clearDisruptions } = useAppStore();
  const [delay, setDelay] = useState(14); // Local state for the input field

  const handleRunSimulation = () => {
    runSimulation(parseInt(delay, 10));
  };

  return (
    <div className="w-64 bg-gray-800 p-4 flex flex-col text-white">
      <div>
        <h1 className="text-white text-xl font-bold">Administrator</h1>
        <p className="text-gray-400 text-sm">ID No.- SK******</p>
        <p className="text-gray-400 text-sm mb-6">Station Master</p>
      </div>

      <div className="bg-gray-900 p-4 rounded-lg shadow-lg">
        <h2 className="text-md font-semibold text-white mb-4">
          What-If Simulation
        </h2>

        <label className="text-xs text-gray-400">Select Train</label>
        <select className="bg-gray-700 text-white w-full p-2 rounded-md mb-3 text-sm">
          <option>Train 7867F</option>
          <option>Train 1285B</option>
        </select>

        <label className="text-xs text-gray-400">Inject Delay (mins)</label>
        <input
          type="number"
          value={delay}
          onChange={(e) => setDelay(e.target.value)}
          className="bg-gray-700 text-white w-full p-2 rounded-md mb-4 text-sm"
        />
        <div className="flex flex-col gap-2">
          <button
            onClick={handleRunSimulation}
            className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2 px-2 rounded-lg transition duration-300 text-sm"
          >
            Run Simulation
          </button>
          <button
            onClick={clearDisruptions}
            className="w-full bg-gray-600 hover:bg-gray-500 text-white font-bold py-2 px-2 rounded-lg transition duration-300 text-sm"
          >
            Clear Disruptions
          </button>
        </div>
      </div>

      <div className="flex-grow"></div>

      <div>
        <button className="w-full text-left text-gray-400 hover:bg-gray-700 p-2 rounded-md mb-2">
          Settings
        </button>
        <button className="w-full text-left text-gray-400 hover:bg-gray-700 p-2 rounded-md">
          Logout
        </button>
      </div>
    </div>
  );
}

export default SidePanel;
