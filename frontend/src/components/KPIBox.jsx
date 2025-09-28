import React from "react";

function KPIBox({ title, value, icon, change, colorName }) {
  // --- THIS IS THE FIX ---
  // We map simple names to the full Tailwind classes
  const colorMap = {
    yellow: "border-yellow-500",
    blue: "border-blue-500",
    purple: "border-purple-500",
  };

  const changeColorClass =
    change && change.startsWith("+") ? "text-green-400" : "text-red-400";

  return (
    // And here we look up the full class name from our map
    <div
      className={`bg-gray-800 rounded-lg p-4 flex items-center gap-4 border-l-4 ${
        colorMap[colorName] || "border-gray-700"
      } shadow-md`}
    >
      <div className="flex-shrink-0">
        {React.createElement(icon, { className: "h-8 w-8 text-gray-400" })}
      </div>
      <div className="flex-grow">
        <h2 className="text-sm text-gray-400">{title}</h2>
        <div className="flex items-baseline gap-2">
          <p className="text-2xl font-semibold">{value}</p>
          {change && (
            <p className={`text-sm font-semibold ${changeColorClass}`}>
              {change}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default KPIBox;
