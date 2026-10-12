"use client";

import React, { useState } from "react";
import { SensorMetadata } from "@/lib/types";

interface SensorInspectorProps {
  selectedSensor: SensorMetadata;
  onSelectSensor: (sensor: SensorMetadata) => void;
  allSensors: SensorMetadata[];
  stepData: any;
  history: any[];
}

export const SensorInspector: React.FC<SensorInspectorProps> = ({
  selectedSensor,
  onSelectSensor,
  allSensors,
  stepData,
  history,
}) => {
  const [filterCategory, setFilterCategory] = useState<string>("all");

  const currentVal = stepData?.sensor_readings?.[selectedSensor.name];
  const [minNom, maxNom] = selectedSensor.nominal_range;

  // Calculate normalized z-score approximately for demonstration
  const normalizedVal =
    currentVal !== undefined
      ? ((currentVal - (minNom + maxNom) / 2) / ((maxNom - minNom) / 4)).toFixed(2)
      : "0.00";

  // Build sparkline points from history
  const historyVals = history
    .map((h) => h?.sensor_readings?.[selectedSensor.name])
    .filter((v) => v !== undefined);

  const sparkPoints = historyVals.length > 0 ? historyVals : [currentVal || minNom];
  const sparkMin = Math.min(...sparkPoints, minNom);
  const sparkMax = Math.max(...sparkPoints, maxNom);
  const sparkRange = sparkMax - sparkMin || 1;

  const sparkPolyline = sparkPoints
    .map((v, i) => {
      const x = Math.round((i / Math.max(sparkPoints.length - 1, 1)) * 260 + 10);
      const y = Math.round(50 - ((v - sparkMin) / sparkRange) * 40);
      return `${x},${y}`;
    })
    .join(" ");

  const filteredList =
    filterCategory === "all"
      ? allSensors
      : allSensors.filter((s) => s.category === filterCategory);

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-lg p-4 shadow-sm select-none">
      <div className="flex items-center justify-between border-b border-[#F3F4F6] pb-2.5 mb-3">
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-[#171717] font-mono uppercase">
            Multivariate Telemetry Inspector
          </h3>
          <p className="text-xs text-[#6B7280]">
            Channel #{selectedSensor.index} • Causal Window Sync (W=4, S=4)
          </p>
        </div>
        <div className="flex gap-1">
          {["all", "temperature", "pressure", "speed"].map((cat) => (
            <button
              key={cat}
              onClick={() => setFilterCategory(cat)}
              className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded border transition-colors ${
                filterCategory === cat
                  ? "bg-[#171717] text-white border-[#171717]"
                  : "bg-[#F9FAFB] text-[#6B7280] border-[#E5E7EB] hover:bg-[#F3F4F6]"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Main Single Sensor Detail Card */}
      <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3 mb-3">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-base font-bold text-[#171717]">
                {selectedSensor.name}
              </span>
              <span className="text-xs font-mono text-[#3563FF] bg-[#EFF6FF] px-2 py-0.5 rounded border border-[#BFDBFE]">
                {selectedSensor.label}
              </span>
            </div>
            <div className="text-xs text-[#4B5563] mt-1 font-mono">
              Sensor Location: <span className="text-[#171717]">{selectedSensor.location}</span>
            </div>
          </div>
          <div className="text-right">
            <div className="font-mono text-2xl font-bold text-[#171717]">
              {currentVal !== undefined ? currentVal.toFixed(1) : "—"}
              <span className="text-xs text-[#6B7280] font-normal ml-1">
                {selectedSensor.unit}
              </span>
            </div>
            <div className="text-[10px] font-mono text-[#6B7280]">
              z-score: <span className="font-semibold text-[#171717]">{normalizedVal}σ</span>
            </div>
          </div>
        </div>

        {/* Telemetry Sparkline SVG */}
        <div className="mt-3 bg-white border border-[#E5E7EB] rounded p-2">
          <div className="flex items-center justify-between text-[10px] font-mono text-[#6B7280] mb-1">
            <span>Recent Window Trend ({sparkPoints.length} points)</span>
            <span>Nominal: [{minNom}, {maxNom}] {selectedSensor.unit}</span>
          </div>
          <svg viewBox="0 0 280 60" className="w-full h-12 overflow-visible">
            {/* Nominal range guideline */}
            <line x1="10" y1="10" x2="270" y2="10" stroke="#E5E7EB" strokeDasharray="3 3" />
            <line x1="10" y1="50" x2="270" y2="50" stroke="#E5E7EB" strokeDasharray="3 3" />
            {/* Sparkline curve */}
            <polyline
              fill="none"
              stroke="#3563FF"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={sparkPolyline}
            />
            {/* Current point dot */}
            {historyVals.length > 0 && (
              <circle
                cx={270}
                cy={Math.round(50 - (((historyVals[historyVals.length - 1] || minNom) - sparkMin) / sparkRange) * 40)}
                r="3.5"
                fill="#3563FF"
                stroke="#FFFFFF"
                strokeWidth="1.5"
              />
            )}
          </svg>
        </div>

        {/* Model Downstream Lineage */}
        <div className="mt-2.5 text-[11px] font-mono text-[#4B5563] bg-white border border-[#E5E7EB] rounded p-2 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#10B981]" />
            <span>Downstream Target:</span>
          </div>
          <span className="text-[#171717] font-semibold">
            Stage 1 Window (Ch #{selectedSensor.index}) → Causal Conv Encoder → Ridge Raw Bypass
          </span>
        </div>
      </div>

      {/* 18 Channels Grid List */}
      <div className="text-[11px] font-mono text-[#6B7280] uppercase mb-1.5 flex justify-between">
        <span>Channel Bank ({filteredList.length} of {allSensors.length})</span>
        <span>Click to Inspect</span>
      </div>
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-1.5 max-h-[140px] overflow-y-auto pr-1">
        {filteredList.map((sensor) => {
          const val = stepData?.sensor_readings?.[sensor.name];
          const isSelected = selectedSensor.name === sensor.name;

          return (
            <button
              key={sensor.name}
              onClick={() => onSelectSensor(sensor)}
              className={`p-1.5 rounded border text-left font-mono transition-all ${
                isSelected
                  ? "bg-[#EFF6FF] border-[#3563FF] text-[#1D4ED8] ring-1 ring-[#3563FF]"
                  : "bg-[#F9FAFB] border-[#E5E7EB] text-[#374151] hover:bg-[#F3F4F6]"
              }`}
            >
              <div className="text-[10px] font-bold truncate">{sensor.name}</div>
              <div className="text-[11px] font-semibold text-[#171717] mt-0.5">
                {val !== undefined ? val.toFixed(0) : "—"}
              </div>
              <div className="text-[9px] text-[#9CA3AF] truncate">{sensor.unit}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
