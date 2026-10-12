"use client";

import React from "react";
import { SensorMetadata } from "@/lib/types";

interface TurbofanRigProps {
  selectedSensor: SensorMetadata;
  onSelectSensor: (sensor: SensorMetadata) => void;
  allSensors: SensorMetadata[];
  isProcessing: boolean;
  stepData: any;
}

interface Hotspot {
  sensorName: string;
  cx: number;
  cy: number;
  label: string;
  sublabel: string;
}

const HOTSPOTS: Hotspot[] = [
  { sensorName: "T2", cx: 120, cy: 190, label: "T2 / P2", sublabel: "Fan Inlet" },
  { sensorName: "T24", cx: 215, cy: 175, label: "T24", sublabel: "LPC Outlet" },
  { sensorName: "T30", cx: 330, cy: 185, label: "T30 / P30", sublabel: "HPC Combustor" },
  { sensorName: "Wf", cx: 380, cy: 235, label: "Wf", sublabel: "Fuel Manifold" },
  { sensorName: "Nc", cx: 430, cy: 160, label: "Nc / Nf", sublabel: "Spool Speeds" },
  { sensorName: "T50", cx: 540, cy: 190, label: "T50", sublabel: "Turbine Exhaust" },
];

export const TurbofanRig: React.FC<TurbofanRigProps> = ({
  selectedSensor,
  onSelectSensor,
  allSensors,
  isProcessing,
  stepData,
}) => {
  const currentVal = stepData?.sensor_readings?.[selectedSensor.name];

  return (
    <div className="relative flex flex-col bg-white border border-[#E5E7EB] rounded-lg p-4 shadow-sm select-none">
      {/* Header & Academic Surrogate Badge */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#F3F4F6] pb-3 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-[#3563FF] animate-pulse" />
            <h2 className="text-sm font-semibold tracking-wide text-[#171717] uppercase font-mono">
              Industrial Test Cell Rig — N-CMAPSS DS01 Turbofan
            </h2>
          </div>
          <p className="text-xs text-[#6B7280] mt-0.5">
            18-Channel Multivariate Telemetry Acquisition Stand
          </p>
        </div>
        <div className="text-right">
          <span className="inline-block text-[10px] font-mono px-2 py-0.5 rounded bg-[#F3F4F6] text-[#4B5563] border border-[#E5E7EB]">
            Surrogate Methodology Testbed (Not Physical Switchgear)
          </span>
        </div>
      </div>

      {/* Interactive Vector Scene */}
      <div className="relative w-full h-[360px] bg-[#F9FAFB] border border-[#E5E7EB] rounded overflow-hidden flex items-center justify-center">
        {/* Subtle Engineering Floor Grid */}
        <svg
          viewBox="0 0 680 340"
          className="w-full h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <pattern
              id="engineeringGrid"
              width="20"
              height="20"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 20 0 L 0 0 0 20"
                fill="none"
                stroke="#E5E7EB"
                strokeWidth="0.75"
              />
            </pattern>
            {/* Blue pulse filter */}
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background Grid */}
          <rect width="680" height="340" fill="url(#engineeringGrid)" />

          {/* Test Cell Mounting Infrastructure */}
          {/* Ceiling Mount Stanchion */}
          <path
            d="M 280 10 L 280 80 M 420 10 L 420 80 M 260 80 L 440 80"
            stroke="#9CA3AF"
            strokeWidth="3"
            strokeLinecap="round"
          />
          <line x1="280" y1="80" x2="350" y2="120" stroke="#9CA3AF" strokeWidth="2.5" />
          <line x1="420" y1="80" x2="350" y2="120" stroke="#9CA3AF" strokeWidth="2.5" />

          {/* Ground Support Pedestals */}
          <rect x="180" y="270" width="40" height="60" fill="#E5E7EB" stroke="#9CA3AF" strokeWidth="1.5" />
          <rect x="460" y="270" width="40" height="60" fill="#E5E7EB" stroke="#9CA3AF" strokeWidth="1.5" />
          <line x1="160" y1="330" x2="520" y2="330" stroke="#171717" strokeWidth="2" />
          {/* Ground Calibration Hatchings */}
          <path d="M 170 335 L 180 330 M 210 335 L 220 330 M 250 335 L 260 330 M 450 335 L 460 330 M 490 335 L 500 330" stroke="#9CA3AF" strokeWidth="1" />

          {/* Engine Nacelle & Core Geometry (Subtle Isometric Wireframe) */}
          {/* Outer Nacelle Outline */}
          <path
            d="M 90 120 C 120 100, 200 95, 300 95 C 400 95, 480 100, 580 140 L 570 240 C 480 280, 400 285, 300 285 C 200 285, 120 280, 90 260 Z"
            fill="#FFFFFF"
            stroke="#4B5563"
            strokeWidth="1.8"
          />
          {/* Intake Cowl Lip */}
          <ellipse cx="95" cy="190" rx="20" ry="70" fill="#F3F4F6" stroke="#171717" strokeWidth="2" />

          {/* Fan Spinner Cone */}
          <path d="M 85 190 L 150 170 L 150 210 Z" fill="#D1D5DB" stroke="#171717" strokeWidth="1.5" />

          {/* Fan Blades Disk */}
          <line x1="150" y1="125" x2="150" y2="255" stroke="#374151" strokeWidth="3" />
          <line x1="156" y1="130" x2="156" y2="250" stroke="#9CA3AF" strokeWidth="1.5" strokeDasharray="3 3" />

          {/* Core Engine Casing (Inner spool) */}
          <path
            d="M 170 150 L 520 160 L 520 220 L 170 230 Z"
            fill="#F3F4F6"
            stroke="#4B5563"
            strokeWidth="1.5"
          />

          {/* Low-Pressure Compressor (LPC) Section */}
          <rect x="180" y="155" width="60" height="70" fill="#E5E7EB" stroke="#6B7280" strokeWidth="1" />
          <text x="210" y="195" fontSize="8" fill="#6B7280" textAnchor="middle" fontFamily="monospace">LPC</text>

          {/* High-Pressure Compressor (HPC) Section */}
          <rect x="250" y="158" width="90" height="64" fill="#E5E7EB" stroke="#6B7280" strokeWidth="1" />
          <text x="295" y="195" fontSize="8" fill="#6B7280" textAnchor="middle" fontFamily="monospace">HPC</text>

          {/* Combustion Chamber */}
          <path
            d="M 350 162 L 400 162 L 400 218 L 350 218 Z"
            fill="#FEE2E2"
            stroke="#EF4444"
            strokeWidth="1.2"
            strokeDasharray="4 2"
          />
          <text x="375" y="193" fontSize="8" fill="#B91C1C" textAnchor="middle" fontFamily="monospace" fontWeight="bold">BURNER</text>

          {/* High-Pressure Turbine (HPT) */}
          <rect x="410" y="160" width="40" height="60" fill="#E5E7EB" stroke="#6B7280" strokeWidth="1" />
          <text x="430" y="195" fontSize="8" fill="#6B7280" textAnchor="middle" fontFamily="monospace">HPT</text>

          {/* Low-Pressure Turbine (LPT) */}
          <rect x="460" y="163" width="50" height="54" fill="#E5E7EB" stroke="#6B7280" strokeWidth="1" />
          <text x="485" y="195" fontSize="8" fill="#6B7280" textAnchor="middle" fontFamily="monospace">LPT</text>

          {/* Core Exhaust Nozzle */}
          <path d="M 520 170 L 590 180 L 590 200 L 520 210 Z" fill="#D1D5DB" stroke="#171717" strokeWidth="1.5" />

          {/* Exhaust Plume Guide */}
          <path
            d="M 590 185 C 620 180, 640 175, 660 170 M 590 195 C 620 200, 640 205, 660 210"
            stroke="#9CA3AF"
            strokeWidth="1.2"
            strokeDasharray="3 3"
          />

          {/* Signal Conduits (Thin Wires from Hotspots to Pipeline Bus) */}
          <path
            d="M 120 190 L 120 70 L 640 70"
            fill="none"
            stroke="#3563FF"
            strokeWidth={selectedSensor.name === "T2" ? 2 : 0.8}
            strokeDasharray={isProcessing ? "4 2" : "none"}
            opacity={selectedSensor.name === "T2" ? 0.9 : 0.3}
          />
          <path
            d="M 330 185 L 330 60 L 640 60"
            fill="none"
            stroke="#3563FF"
            strokeWidth={selectedSensor.name === "T30" ? 2 : 0.8}
            strokeDasharray={isProcessing ? "4 2" : "none"}
            opacity={selectedSensor.name === "T30" ? 0.9 : 0.3}
          />
          <path
            d="M 540 190 L 540 50 L 640 50"
            fill="none"
            stroke="#3563FF"
            strokeWidth={selectedSensor.name === "T50" ? 2 : 0.8}
            strokeDasharray={isProcessing ? "4 2" : "none"}
            opacity={selectedSensor.name === "T50" ? 0.9 : 0.3}
          />

          {/* Telemetry Junction Box at top right */}
          <rect x="625" y="40" width="30" height="40" fill="#171717" rx="2" />
          <circle cx="640" cy="50" r="3" fill="#3563FF" />
          <circle cx="640" cy="60" r="3" fill="#10B981" />
          <circle cx="640" cy="70" r="3" fill="#F59E0B" />
          <text x="640" y="32" fontSize="7" fill="#171717" textAnchor="middle" fontFamily="monospace">EGPM BUS</text>

          {/* Hotspots: Interactive Sensor Markers */}
          {HOTSPOTS.map((spot) => {
            const isSelected = selectedSensor.name === spot.sensorName;
            const targetMeta = allSensors.find((s) => s.name === spot.sensorName) || selectedSensor;

            return (
              <g
                key={spot.sensorName}
                className="cursor-pointer transition-transform hover:scale-110"
                onClick={() => onSelectSensor(targetMeta)}
              >
                {/* Ping animation when selected */}
                {isSelected && (
                  <circle
                    cx={spot.cx}
                    cy={spot.cy}
                    r="12"
                    fill="none"
                    stroke="#3563FF"
                    strokeWidth="1.5"
                    className="animate-ping"
                    opacity="0.75"
                  />
                )}
                {/* Marker Outer Circle */}
                <circle
                  cx={spot.cx}
                  cy={spot.cy}
                  r="7"
                  fill={isSelected ? "#3563FF" : "#FFFFFF"}
                  stroke={isSelected ? "#1D4ED8" : "#4B5563"}
                  strokeWidth="2"
                />
                {/* Marker Inner Dot */}
                <circle
                  cx={spot.cx}
                  cy={spot.cy}
                  r="2.5"
                  fill={isSelected ? "#FFFFFF" : "#3563FF"}
                />

                {/* Hotspot Text Label */}
                <rect
                  x={spot.cx - 24}
                  y={spot.cy + 11}
                  width="48"
                  height="16"
                  rx="3"
                  fill={isSelected ? "#171717" : "#FFFFFF"}
                  stroke={isSelected ? "#3563FF" : "#D1D5DB"}
                  strokeWidth="1"
                />
                <text
                  x={spot.cx}
                  y={spot.cy + 22}
                  fontSize="8"
                  fontWeight="600"
                  fontFamily="monospace"
                  fill={isSelected ? "#FFFFFF" : "#171717"}
                  textAnchor="middle"
                >
                  {spot.label}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Selected Sensor Floating Info Overlay */}
        <div className="absolute top-3 left-3 bg-white/95 backdrop-blur-sm border border-[#E5E7EB] rounded p-2.5 text-xs shadow-sm max-w-[210px] pointer-events-none">
          <div className="flex items-center justify-between gap-1 text-[11px] font-mono font-bold text-[#171717]">
            <span>{selectedSensor.name}</span>
            <span className="text-[#3563FF] bg-[#EFF6FF] px-1.5 py-0.2 rounded border border-[#BFDBFE]">
              CH #{selectedSensor.index}
            </span>
          </div>
          <div className="text-[11px] text-[#4B5563] truncate mt-0.5 font-medium">
            {selectedSensor.label}
          </div>
          <div className="mt-1.5 flex items-baseline justify-between border-t border-[#F3F4F6] pt-1.5">
            <span className="text-[10px] text-[#6B7280]">Current Reading:</span>
            <span className="font-mono font-bold text-[#171717]">
              {currentVal !== undefined ? currentVal.toFixed(1) : "—"} {selectedSensor.unit}
            </span>
          </div>
          <div className="text-[9px] text-[#9CA3AF] mt-0.5">
            Nominal: {selectedSensor.nominal_range[0]} – {selectedSensor.nominal_range[1]} {selectedSensor.unit}
          </div>
        </div>
      </div>

      {/* Sensor Quick-Selector Badges */}
      <div className="mt-3">
        <div className="text-[11px] font-mono text-[#6B7280] uppercase mb-1.5 flex items-center justify-between">
          <span>Inspect Active Sensor Point</span>
          <span className="text-[10px] text-[#9CA3AF]">Click to probe signal channel</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {HOTSPOTS.map((spot) => {
            const meta = allSensors.find((s) => s.name === spot.sensorName);
            if (!meta) return null;
            const isSel = selectedSensor.name === meta.name;
            return (
              <button
                key={meta.name}
                onClick={() => onSelectSensor(meta)}
                className={`text-xs px-2.5 py-1 rounded font-mono transition-all border ${
                  isSel
                    ? "bg-[#3563FF] text-white border-[#3563FF] shadow-xs"
                    : "bg-[#F9FAFB] text-[#374151] border-[#E5E7EB] hover:bg-[#F3F4F6]"
                }`}
              >
                {meta.name} ({spot.sublabel})
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
