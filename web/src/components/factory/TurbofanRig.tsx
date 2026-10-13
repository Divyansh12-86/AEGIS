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

interface SensorStation {
  sensorName: string;
  cx: number;
  cy: number;
  label: string;
  sublabel: string;
  stationId: string;
  badgeX: number;
  badgeY: number;
  wirePath: string;
}

const SENSOR_STATIONS: SensorStation[] = [
  {
    sensorName: "T2",
    cx: 120,
    cy: 190,
    label: "T2 / P2",
    sublabel: "Fan Inlet",
    stationId: "STA 2.0",
    badgeX: 96,
    badgeY: 215,
    wirePath: "M 120 190 L 120 52 L 605 52",
  },
  {
    sensorName: "T24",
    cx: 215,
    cy: 175,
    label: "T24",
    sublabel: "LPC Outlet",
    stationId: "STA 2.4",
    badgeX: 191,
    badgeY: 200,
    wirePath: "M 215 175 L 215 52 L 605 52",
  },
  {
    sensorName: "T30",
    cx: 325,
    cy: 185,
    label: "T30 / P30",
    sublabel: "HPC Combustor",
    stationId: "STA 3.0",
    badgeX: 301,
    badgeY: 210,
    wirePath: "M 325 185 L 325 52 L 605 52",
  },
  {
    sensorName: "Wf",
    cx: 380,
    cy: 240,
    label: "Wf",
    sublabel: "Fuel Manifold",
    stationId: "STA 3.5",
    badgeX: 356,
    badgeY: 272,
    wirePath: "M 380 240 L 380 264 L 355 264 L 355 52 L 605 52",
  },
  {
    sensorName: "Nc",
    cx: 440,
    cy: 150,
    label: "Nc / Nf",
    sublabel: "Spool Speeds",
    stationId: "STA 4.0",
    badgeX: 416,
    badgeY: 120,
    wirePath: "M 440 150 L 440 52 L 605 52",
  },
  {
    sensorName: "T50",
    cx: 540,
    cy: 190,
    label: "T50",
    sublabel: "Turbine Exhaust",
    stationId: "STA 5.0",
    badgeX: 516,
    badgeY: 215,
    wirePath: "M 540 190 L 540 52 L 605 52",
  },
];

export const TurbofanRig: React.FC<TurbofanRigProps> = ({
  selectedSensor,
  onSelectSensor,
  allSensors,
  isProcessing,
  stepData,
}) => {
  const currentVal = stepData?.sensor_readings?.[selectedSensor.name];
  const isAnomalous = stepData?.is_anomaly || false;
  const isUnreliable = stepData?.is_unreliable || false;

  return (
    <div className="relative flex flex-col bg-white border border-[#E5E7EB] rounded-lg p-4 shadow-sm select-none">
      {/* Header & Academic Surrogate Badge */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#F3F4F6] pb-3 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-2.5 h-2.5 rounded-full ${
                isProcessing ? "bg-[#3563FF] animate-ping" : "bg-[#10B981]"
              }`}
            />
            <h2 className="text-sm font-semibold tracking-wide text-[#171717] uppercase font-mono">
              Industrial Test Cell Rig — N-CMAPSS DS01 Turbofan
            </h2>
          </div>
          <p className="text-xs text-[#6B7280] mt-0.5">
            18-Channel Multivariate Telemetry Acquisition Stand • Synchronous EGPM Bus
          </p>
        </div>
        <div className="text-right flex items-center gap-2">
          <span className="inline-block text-[10px] font-mono px-2 py-0.5 rounded bg-[#F3F4F6] text-[#4B5563] border border-[#E5E7EB]">
            Surrogate Methodology Testbed (Not Physical Switchgear)
          </span>
        </div>
      </div>

      {/* Interactive Vector Scene */}
      <div className="relative w-full h-[360px] bg-[#F9FAFB] border border-[#E5E7EB] rounded overflow-hidden flex items-center justify-center">
        <svg
          viewBox="0 0 680 340"
          className="w-full h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <style>
              {`
                @keyframes telemetryFlow {
                  from { stroke-dashoffset: 24; }
                  to { stroke-dashoffset: 0; }
                }
                .telemetry-active-wire {
                  stroke-dasharray: 6 4;
                  animation: telemetryFlow 0.8s linear infinite;
                }
                .telemetry-idle-wire {
                  stroke-dasharray: 2 6;
                  animation: telemetryFlow 2.4s linear infinite;
                }
              `}
            </style>
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
            {/* Blue glow filter for active telemetry wiring */}
            <filter id="blueGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            {/* Sensor tap marker filter */}
            <filter id="markerShadow" x="-10%" y="-10%" width="120%" height="120%">
              <feDropShadow dx="0" dy="1" stdDeviation="1.5" floodColor="#000000" floodOpacity="0.15" />
            </filter>
          </defs>

          {/* Background Engineering Grid */}
          <rect width="680" height="340" fill="url(#engineeringGrid)" />

          {/* Test Cell Mounting Infrastructure */}
          {/* Ceiling Structural I-Beam Gantry */}
          <line x1="60" y1="16" x2="620" y2="16" stroke="#4B5563" strokeWidth="3" />
          <line x1="60" y1="19" x2="620" y2="19" stroke="#9CA3AF" strokeWidth="2" />
          <line x1="60" y1="22" x2="620" y2="22" stroke="#4B5563" strokeWidth="3" />

          {/* Overhead Suspension Pylons (cleanly separated from wiring) */}
          {/* Forward Suspension Pylon (attaches to Fan/LPC casing) */}
          <polygon
            points="195,22 215,22 208,110 202,110"
            fill="#E5E7EB"
            stroke="#6B7280"
            strokeWidth="1.2"
          />
          <circle cx="205" cy="110" r="3.5" fill="#171717" stroke="#9CA3AF" strokeWidth="1" />

          {/* Aft Suspension Pylon (attaches to Turbine casing) */}
          <polygon
            points="465,22 485,22 478,115 472,115"
            fill="#E5E7EB"
            stroke="#6B7280"
            strokeWidth="1.2"
          />
          <circle cx="475" cy="115" r="3.5" fill="#171717" stroke="#9CA3AF" strokeWidth="1" />

          {/* Ground Support Pedestals & Load Cells */}
          <rect x="180" y="270" width="40" height="60" fill="#E5E7EB" stroke="#9CA3AF" strokeWidth="1.5" />
          <rect x="460" y="270" width="40" height="60" fill="#E5E7EB" stroke="#9CA3AF" strokeWidth="1.5" />
          <circle cx="200" cy="270" r="3.5" fill="#171717" />
          <circle cx="480" cy="270" r="3.5" fill="#171717" />
          <line x1="150" y1="330" x2="530" y2="330" stroke="#171717" strokeWidth="2" />
          {/* Ground Hatchings */}
          <path
            d="M 160 335 L 170 330 M 190 335 L 200 330 M 220 335 L 230 330 M 440 335 L 450 330 M 470 335 L 480 330 M 500 335 L 510 330"
            stroke="#9CA3AF"
            strokeWidth="1"
          />

          {/* OVERHEAD CABLE LADDER (Wireway Trunk at y=48..56) */}
          {/* Cable Ladder Hanger Brackets from Ceiling Beam */}
          <line x1="160" y1="22" x2="160" y2="48" stroke="#9CA3AF" strokeWidth="1" />
          <line x1="280" y1="22" x2="280" y2="48" stroke="#9CA3AF" strokeWidth="1" />
          <line x1="400" y1="22" x2="400" y2="48" stroke="#9CA3AF" strokeWidth="1" />
          <line x1="520" y1="22" x2="520" y2="48" stroke="#9CA3AF" strokeWidth="1" />

          {/* Cable Ladder Tray Rails */}
          <line x1="90" y1="48" x2="605" y2="48" stroke="#9CA3AF" strokeWidth="1.5" />
          <line x1="90" y1="56" x2="605" y2="56" stroke="#9CA3AF" strokeWidth="1.5" />
          {/* Cable Ladder Rungs */}
          {Array.from({ length: 26 }).map((_, i) => (
            <line
              key={i}
              x1={105 + i * 19}
              y1="48"
              x2={105 + i * 19}
              y2="56"
              stroke="#D1D5DB"
              strokeWidth="1"
            />
          ))}

          {/* Engine Nacelle & Core Geometry (Precise Vector Schematic) */}
          {/* Outer Nacelle Cowling Profile */}
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

          {/* Fan Blades Disk with Guide Stators */}
          <line x1="150" y1="125" x2="150" y2="255" stroke="#374151" strokeWidth="3" />
          <line x1="156" y1="130" x2="156" y2="250" stroke="#9CA3AF" strokeWidth="1.5" strokeDasharray="3 3" />

          {/* Core Engine Casing (Inner Spool Nacelle) */}
          <path
            d="M 170 150 L 520 160 L 520 220 L 170 230 Z"
            fill="#F3F4F6"
            stroke="#4B5563"
            strokeWidth="1.5"
          />

          {/* Station 2.0 - 2.4: Low-Pressure Compressor (LPC) Section */}
          <rect x="180" y="155" width="60" height="70" fill="#E5E7EB" stroke="#6B7280" strokeWidth="1" />
          <text x="210" y="195" fontSize="8" fill="#6B7280" textAnchor="middle" fontFamily="monospace">LPC</text>

          {/* Station 2.4 - 3.0: High-Pressure Compressor (HPC) Section */}
          <rect x="250" y="158" width="90" height="64" fill="#E5E7EB" stroke="#6B7280" strokeWidth="1" />
          <text x="295" y="195" fontSize="8" fill="#6B7280" textAnchor="middle" fontFamily="monospace">HPC</text>

          {/* Station 3.0 - 3.5: Combustion Chamber (Burner) */}
          <path
            d="M 350 162 L 400 162 L 400 218 L 350 218 Z"
            fill="#FEE2E2"
            stroke="#EF4444"
            strokeWidth="1.2"
            strokeDasharray="4 2"
          />
          <text x="375" y="193" fontSize="8" fill="#B91C1C" textAnchor="middle" fontFamily="monospace" fontWeight="bold">BURNER</text>

          {/* Station 4.0: High-Pressure Turbine (HPT) */}
          <rect x="410" y="160" width="40" height="60" fill="#E5E7EB" stroke="#6B7280" strokeWidth="1" />
          <text x="430" y="195" fontSize="8" fill="#6B7280" textAnchor="middle" fontFamily="monospace">HPT</text>

          {/* Station 5.0: Low-Pressure Turbine (LPT) */}
          <rect x="460" y="163" width="50" height="54" fill="#E5E7EB" stroke="#6B7280" strokeWidth="1" />
          <text x="485" y="195" fontSize="8" fill="#6B7280" textAnchor="middle" fontFamily="monospace">LPT</text>

          {/* Core Exhaust Nozzle & Centerbody */}
          <path d="M 520 170 L 590 180 L 590 200 L 520 210 Z" fill="#D1D5DB" stroke="#171717" strokeWidth="1.5" />

          {/* Exhaust Plume Guide Vectors */}
          <path
            d="M 590 185 C 620 180, 640 175, 660 170 M 590 195 C 620 200, 640 205, 660 210"
            stroke="#9CA3AF"
            strokeWidth="1.2"
            strokeDasharray="3 3"
          />

          {/* Mechanical Station Datum Markers along Bottom */}
          <g fontSize="7" fill="#9CA3AF" fontFamily="monospace" textAnchor="middle">
            <text x="120" y="302">STA 2.0</text>
            <text x="215" y="302">STA 2.4</text>
            <text x="325" y="302">STA 3.0</text>
            <text x="380" y="302">STA 3.5</text>
            <text x="440" y="302">STA 4.0</text>
            <text x="540" y="302">STA 5.0</text>
          </g>

          {/* ========================================================================= */}
          {/* INDUSTRIAL SENSOR WIRING HARNESS — ALL 6 STATIONS CONNECTED TO EGPM BUS  */}
          {/* ========================================================================= */}

          {/* Base Unselected Shielded Conduits (for all stations) */}
          {SENSOR_STATIONS.map((station) => {
            const isSelected = selectedSensor.name === station.sensorName;
            return (
              <g key={`conduit-base-${station.sensorName}`}>
                {/* Physical Conduit Outer Sheath */}
                <path
                  d={station.wirePath}
                  fill="none"
                  stroke={isSelected ? "#3563FF" : "#D1D5DB"}
                  strokeWidth={isSelected ? 2.5 : 1.2}
                  opacity={isSelected ? 1.0 : 0.6}
                />
                {/* Conduit Junction Clamps */}
                <rect
                  x={station.cx - 2}
                  y={50}
                  width="4"
                  height="4"
                  fill={isSelected ? "#3563FF" : "#9CA3AF"}
                  rx="1"
                />
              </g>
            );
          })}

          {/* Active Data Stream Overlays (dashed telemetry flows) */}
          {SENSOR_STATIONS.map((station) => {
            const isSelected = selectedSensor.name === station.sensorName;

            if (isSelected) {
              return (
                <g key={`conduit-active-${station.sensorName}`}>
                  {/* Glowing Highlight Sheath */}
                  <path
                    d={station.wirePath}
                    fill="none"
                    stroke="#3563FF"
                    strokeWidth="3.5"
                    opacity="0.3"
                    filter="url(#blueGlow)"
                  />
                  {/* High-frequency animated telemetry dash stream */}
                  <path
                    d={station.wirePath}
                    fill="none"
                    stroke="#FFFFFF"
                    strokeWidth="1.8"
                    className="telemetry-active-wire"
                  />
                  {/* Animated telemetry packet traveling from sensor probe into DAQ */}
                  <circle r="3.5" fill="#3563FF" filter="url(#blueGlow)">
                    <animateMotion
                      path={station.wirePath}
                      dur="1.3s"
                      repeatCount="indefinite"
                    />
                  </circle>
                  <circle r="1.8" fill="#FFFFFF">
                    <animateMotion
                      path={station.wirePath}
                      dur="1.3s"
                      repeatCount="indefinite"
                    />
                  </circle>
                </g>
              );
            }

            // If processing, subtle background flow on all channels
            if (isProcessing) {
              return (
                <path
                  key={`conduit-proc-${station.sensorName}`}
                  d={station.wirePath}
                  fill="none"
                  stroke="#3563FF"
                  strokeWidth="0.9"
                  opacity="0.5"
                  className="telemetry-idle-wire"
                />
              );
            }

            return null;
          })}

          {/* ========================================================================= */}
          {/* DAQ / EGPM EDGE CONCENTRATOR ENCLOSURE (Top Right, x=605..668, y=28..84)  */}
          {/* ========================================================================= */}
          <g>
            {/* Cable Gland Interface Plate */}
            <rect x="601" y="44" width="4" height="16" fill="#4B5563" rx="1" />
            <circle cx="603" cy="52" r="2.5" fill="#171717" />

            {/* Enclosure Outer Chassis */}
            <rect
              x="605"
              y="28"
              width="63"
              height="56"
              rx="3"
              fill="#171717"
              stroke="#374151"
              strokeWidth="1.5"
            />
            {/* Metallic Header Plate */}
            <rect x="605" y="28" width="63" height="12" fill="#262626" rx="2" />
            <text
              x="636"
              y="37"
              fontSize="6.5"
              fill="#E5E7EB"
              textAnchor="middle"
              fontFamily="monospace"
              fontWeight="bold"
            >
              EGPM EDGE DAQ
            </text>

            {/* Status LEDs & Diagnostic Labels */}
            {/* LED 1: PWR */}
            <circle cx="616" cy="48" r="2.5" fill="#10B981" />
            <text x="623" y="50" fontSize="5.5" fill="#9CA3AF" fontFamily="monospace">PWR</text>

            {/* LED 2: BUS (Sync/Active) */}
            <circle
              cx="644"
              cy="48"
              r="2.5"
              fill={isProcessing ? "#3563FF" : "#60A5FA"}
              className={isProcessing ? "animate-pulse" : ""}
            />
            <text x="651" y="50" fontSize="5.5" fill="#9CA3AF" fontFamily="monospace">BUS</text>

            {/* LED 3: VQ (Quantized Event Sync) */}
            <circle cx="616" cy="60" r="2.5" fill="#8B5CF6" />
            <text x="623" y="62" fontSize="5.5" fill="#9CA3AF" fontFamily="monospace">VQ</text>

            {/* LED 4: HAZ (Fault/Hazard Alert) */}
            <circle
              cx="644"
              cy="60"
              r="2.5"
              fill={isUnreliable ? "#EF4444" : isAnomalous ? "#F59E0B" : "#10B981"}
            />
            <text x="651" y="62" fontSize="5.5" fill="#9CA3AF" fontFamily="monospace">HAZ</text>

            {/* Real-Time Bitstream Sampling Rate Indicator */}
            <text
              x="636"
              y="76"
              fontSize="5.5"
              fill="#60A5FA"
              textAnchor="middle"
              fontFamily="monospace"
              fontWeight="bold"
            >
              18-CH • 4.2 kHz
            </text>
          </g>

          {/* ========================================================================= */}
          {/* INTERACTIVE SENSOR PROBE HOTSPOTS & CALIBRATED BADGES                     */}
          {/* ========================================================================= */}
          {SENSOR_STATIONS.map((spot) => {
            const isSelected = selectedSensor.name === spot.sensorName;
            const targetMeta = allSensors.find((s) => s.name === spot.sensorName) || selectedSensor;

            return (
              <g
                key={spot.sensorName}
                className="cursor-pointer transition-transform hover:scale-105"
                onClick={() => onSelectSensor(targetMeta)}
                filter="url(#markerShadow)"
              >
                {/* Ping pulse ring when selected */}
                {isSelected && (
                  <circle
                    cx={spot.cx}
                    cy={spot.cy}
                    r="13"
                    fill="none"
                    stroke="#3563FF"
                    strokeWidth="1.5"
                    className="animate-ping"
                    opacity="0.8"
                  />
                )}

                {/* Probe Reticle Crosshairs */}
                <line
                  x1={spot.cx - 9}
                  y1={spot.cy}
                  x2={spot.cx + 9}
                  y2={spot.cy}
                  stroke={isSelected ? "#3563FF" : "#9CA3AF"}
                  strokeWidth="0.8"
                />
                <line
                  x1={spot.cx}
                  y1={spot.cy - 9}
                  x2={spot.cx}
                  y2={spot.cy + 9}
                  stroke={isSelected ? "#3563FF" : "#9CA3AF"}
                  strokeWidth="0.8"
                />

                {/* Probe Head Outer Circle */}
                <circle
                  cx={spot.cx}
                  cy={spot.cy}
                  r="7"
                  fill={isSelected ? "#3563FF" : "#FFFFFF"}
                  stroke={isSelected ? "#1D4ED8" : "#4B5563"}
                  strokeWidth="2"
                />
                {/* Probe Center Contact Pin */}
                <circle
                  cx={spot.cx}
                  cy={spot.cy}
                  r="2.5"
                  fill={isSelected ? "#FFFFFF" : "#3563FF"}
                />

                {/* Probe Station Engineering Callout Badge */}
                <rect
                  x={spot.badgeX}
                  y={spot.badgeY}
                  width="48"
                  height="16"
                  rx="3"
                  fill={isSelected ? "#171717" : "#FFFFFF"}
                  stroke={isSelected ? "#3563FF" : "#D1D5DB"}
                  strokeWidth={isSelected ? 1.5 : 1}
                />
                <text
                  x={spot.badgeX + 24}
                  y={spot.badgeY + 11}
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
          {SENSOR_STATIONS.map((spot) => {
            const meta = allSensors.find((s) => s.name === spot.sensorName);
            if (!meta) return null;
            const isSel = selectedSensor.name === meta.name;
            return (
              <button
                key={meta.name}
                onClick={() => onSelectSensor(meta)}
                className={`text-xs px-2.5 py-1 rounded font-mono transition-all border ${
                  isSel
                    ? "bg-[#3563FF] text-white border-[#3563FF] shadow-xs font-semibold"
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
