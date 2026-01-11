"use client";

import { Settings, Globe, Thermometer } from "lucide-react";

export default function RightSidebar({
  temperature,
  setTemperature,
  webSearch,
  setWebSearch,
}: {
  temperature: number;
  setTemperature: (v: number) => void;
  webSearch: boolean;
  setWebSearch: (v: boolean) => void;
}) {
  return (
    <div className="w-72 bg-black border-l border-gray-800 text-gray-300 flex flex-col h-screen p-6 hidden lg:flex">
      
      <div className="flex items-center gap-2 mb-8 text-indigo-400">
        <Settings className="w-5 h-5" />
        <h2 className="font-semibold text-lg">Model Config</h2>
      </div>

      <div className="mb-8">
        <div className="flex justify-between items-center mb-3">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Thermometer className="w-4 h-4 text-gray-500" />
            Temperature
          </div>
          <span className="text-xs bg-gray-800 px-2 py-1 rounded text-indigo-400 border border-gray-700">
            {temperature.toFixed(1)}
          </span>
        </div>

        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-indigo-600 hover:accent-indigo-500"
        />

        <div className="flex justify-between text-[10px] text-gray-600 mt-2 font-mono">
          <span>Precise (0)</span>
          <span>Creative (1)</span>
        </div>
      </div>

      <div className="mb-8 p-4 bg-gray-900 rounded-xl border border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Globe
              className={`w-5 h-5 ${
                webSearch ? "text-blue-400" : "text-gray-600"
              }`}
            />
            <div className="flex flex-col">
              <span className="text-sm font-medium text-white">
                Web Search
              </span>
              <span className="text-[10px] text-gray-500">
                Access live internet data
              </span>
            </div>
          </div>

          <button
            onClick={() => setWebSearch(!webSearch)}
            className={`w-11 h-6 flex items-center rounded-full transition-colors duration-300 ${
              webSearch ? "bg-indigo-600" : "bg-gray-700"
            }`}
          >
            <div
              className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform duration-300 ${
                webSearch ? "translate-x-6" : "translate-x-1"
              }`}
            />
          </button>
        </div>
      </div>
    </div>
  );
}
