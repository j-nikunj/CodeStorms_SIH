import React, { useState } from "react";
import { CheckCircleIcon, StarIcon, EyeIcon } from "@heroicons/react/24/solid";
import { useAppStore } from "../context/AppContext";

function ModelRecommendation() {
  const { recommendations, executeRecommendation, previewRecommendation } =
    useAppStore();
  const [activePreview, setActivePreview] = useState(null);

  const handlePreview = (recId) => {
    previewRecommendation(recId);
    setActivePreview(recId);
  };

  return (
    // --- CHANGE IS HERE: Gray background with a green accent border ---
    <div className="bg-gray-800 border-l-4 border-green-500 rounded-lg p-4 flex flex-col justify-between h-[300px] shadow-md">
      <div>
        <h2 className="text-lg font-semibold mb-4 text-gray-200">
          Model Recommendations
        </h2>
        {recommendations.length > 0 ? (
          recommendations.map((rec) => (
            <div
              key={rec.id}
              onClick={() => handlePreview(rec.id)}
              className={`p-3 rounded-md mb-2 flex items-start gap-3 cursor-pointer transition duration-200 ${
                rec.isRecommended
                  ? "bg-blue-600 hover:bg-blue-500" // Original highlight color
                  : "bg-gray-700 hover:bg-gray-600"
              } ${activePreview === rec.id ? "ring-2 ring-white" : ""}`}
            >
              {rec.isRecommended && (
                <StarIcon className="h-5 w-5 text-yellow-400 mt-1 flex-shrink-0" />
              )}
              {!rec.isRecommended && (
                <EyeIcon className="h-5 w-5 text-gray-400 mt-1 flex-shrink-0" />
              )}
              <p
                className={
                  rec.isRecommended ? "font-bold text-white" : "text-gray-300"
                }
              >
                {rec.text}
              </p>
            </div>
          ))
        ) : (
          <p className="text-gray-400 mt-4">
            Run a simulation to generate recommendations.
          </p>
        )}
      </div>
      <button
        onClick={executeRecommendation}
        className="bg-green-600 hover:bg-green-500 text-white font-bold py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition duration-300 disabled:bg-gray-600"
        disabled={!activePreview}
      >
        <CheckCircleIcon className="h-5 w-5" />
        Execute Selected Plan
      </button>
    </div>
  );
}

export default ModelRecommendation;
