import React, { useState, useEffect } from "react";
import dayjs from "dayjs";

function Header() {
  const [time, setTime] = useState(dayjs());

  useEffect(() => {
    // Update the time every second
    const interval = setInterval(() => {
      setTime(dayjs());
    }, 1000);

    // Cleanup interval on component unmount
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex justify-between items-center border-b border-gray-700 pb-4">
      <h1 className="text-3xl font-bold">Model-Based Railway Command Center</h1>
      {/* --- LIVE CLOCK IS HERE --- */}
      <div className="text-xl font-mono text-gray-300">
        {time.format("HH:mm:ss")}
      </div>
    </div>
  );
}

export default Header;
