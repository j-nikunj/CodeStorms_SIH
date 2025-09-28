import React from "react";
import {
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArchiveBoxXMarkIcon,
} from "@heroicons/react/24/solid";
import { useAppStore } from "../context/AppContext";

function DisruptionsCard() {
  const { disruptions } = useAppStore();

  const logIcons = {
    alert: (
      <ExclamationTriangleIcon className="h-6 w-6 text-red-400 flex-shrink-0" />
    ),
    info: (
      <InformationCircleIcon className="h-6 w-6 text-blue-400 flex-shrink-0" />
    ),
  };

  return (
    <div className="bg-gray-800 border-l-4 border-red-500 rounded-lg p-4 h-[300px] flex flex-col shadow-md">
      <h2 className="text-lg font-semibold mb-4 text-gray-200">
        Disruption Log
      </h2>

      <div className="flex-grow overflow-auto pr-2">
        {disruptions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <ArchiveBoxXMarkIcon className="h-12 w-12 text-gray-500 mb-2" />
            <p>Log is empty.</p>
          </div>
        ) : (
          disruptions.map((log) => (
            <div
              key={log.id}
              className="bg-gray-900 p-3 rounded-md mb-3 flex items-start gap-3"
            >
              {logIcons[log.type] || logIcons.info}
              <div className="flex-grow">
                <p className="text-sm text-white">{log.text}</p>
                <p className="text-xs text-gray-400 font-mono">
                  {log.timestamp}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default DisruptionsCard;
