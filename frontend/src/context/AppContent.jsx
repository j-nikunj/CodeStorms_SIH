import { create } from "zustand";
import dayjs from "dayjs";

const baseData = [
  { time: "15:30", "Train A Delay": 4, "Train B Delay": 2 },
  { time: "15:40", "Train A Delay": 5, "Train B Delay": 3 },
  { time: "15:50", "Train A Delay": 6, "Train B Delay": 4 },
  { time: "16:00", "Train A Delay": 5, "Train B Delay": 22 }, // <-- CONGESTION SPIKE
  { time: "16:10", "Train A Delay": 4, "Train B Delay": 8 },
  { time: "16:20", "Train A Delay": 5, "Train B Delay": 6 },
];

const calculateAverageDelay = (data, key) => {
  if (!data || data.length === 0) return 0;
  const total = data.reduce((sum, item) => sum + item[key], 0);
  return (total / data.length).toFixed(1);
};

export const useAppStore = create((set, get) => ({
  weather: 28,
  sectionThroughput: 12,
  baseGraphData: baseData,
  simulatedGraphData: null,
  recommendations: [],
  // --- CHANGE #1: Disruptions is now an array for logging ---
  disruptions: [
    {
      id: 1,
      type: "info",
      text: "System Initialized",
      timestamp: dayjs().format("HH:mm:ss"),
    },
  ],
  trains: [
    {
      id: "T1",
      pathIndex: 0,
      path: [
        [29.31, 76.31],
        [29.6, 76.1],
        [30.0, 75.8],
        [30.25, 75.83],
      ],
    },
    {
      id: "T2",
      pathIndex: 0,
      path: [
        [30.21, 74.94],
        [30.1, 75.2],
        [30.0, 75.5],
        [29.8, 75.9],
      ],
    },
  ],
  getAverageDelay: () =>
    calculateAverageDelay(get().baseGraphData, "Train A Delay"),

  runSimulation: (delay) =>
    set((state) => {
      const newRecommendations = [
        {
          id: 1,
          text: `Reroute Freight 7867F via alternate path.`,
          isRecommended: false,
        },
        {
          id: 2,
          text: `Hold Train A at Jind Jn for 3 minutes to resolve network congestion.`,
          isRecommended: true,
        },
      ];

      // --- CHANGE #2: Add a new log entry to the START of the array ---
      const newLogEntry = {
        id: Date.now(),
        type: "alert",
        text: `Simulation run: ${delay} min delay injected.`,
        timestamp: dayjs().format("HH:mm:ss"),
      };

      return {
        recommendations: newRecommendations,
        disruptions: [newLogEntry, ...state.disruptions], // Prepend new log
      };
    }),

  previewRecommendation: (recId) =>
    set((state) => {
      const isRecommended = recId === 2;
      const simulatedData = state.baseGraphData.map((d) => {
        let newTrainADelay;
        let newTrainBDelay;
        if (isRecommended) {
          newTrainADelay = d["Train A Delay"] + 3;
          newTrainBDelay =
            d["Train B Delay"] > 20
              ? Math.floor(newTrainADelay / 2)
              : d["Train B Delay"];
        } else {
          newTrainADelay = d["Train A Delay"];
          newTrainBDelay = d["Train B Delay"] + 15;
        }
        return {
          time: d.time,
          "Train A (Simulated)": newTrainADelay,
          "Train B (Simulated)": newTrainBDelay,
        };
      });
      return { simulatedGraphData: simulatedData };
    }),

  executeRecommendation: () =>
    set((state) => {
      if (!state.simulatedGraphData) return {};
      const newBaseData = state.simulatedGraphData.map((d) => ({
        time: d.time,
        "Train A Delay": d["Train A (Simulated)"],
        "Train B Delay": d["Train B (Simulated)"],
      }));
      return {
        baseGraphData: newBaseData,
        simulatedGraphData: null,
        recommendations: [],
      };
    }),

  // --- CHANGE #3: This now resets the log to an empty array ---
  clearDisruptions: () =>
    set({
      disruptions: [],
      simulatedGraphData: null,
      recommendations: [],
    }),

  moveTrains: () =>
    set((state) => ({
      trains: state.trains.map((train) => {
        const nextIndex = (train.pathIndex + 1) % train.path.length;
        return { ...train, pathIndex: nextIndex };
      }),
    })),
}));
