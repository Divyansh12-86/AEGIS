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
  nominalMax: number;
}

const SENSOR_STATIONS: SensorStation[] = [
  {
    sensorName: "T2",
    cx: 120,
    cy: 190,
    label: "T2 / P2",
    sublabel: "Fan Inlet",
    stationId: "STA 2.0",
    badgeX: 92,
    badgeY: 215,
    wirePath: "M 120 190 L 120 52 L 605 52",
    nominalMax: 560,
  },
  {
    sensorName: "T24",
    cx: 215,
    cy: 175,
    label: "T24",
    sublabel: "LPC Outlet",
    stationId: "STA 2.4",
    badgeX: 187,
    badgeY: 200,
    wirePath: "M 215 175 L 215 52 L 605 52",
    nominalMax: 680,
  },
  {
    sensorName: "T30",
    cx: 325,
    cy: 185,
    label: "T30 / P30",
    sublabel: "HPC Combustor",
    stationId: "STA 3.0",
    badgeX: 297,
    badgeY: 210,
    wirePath: "M 325 185 L 325 52 L 605 52",
    nominalMax: 1680,
  },
  {
    sensorName: "Wf",
    cx: 380,
    cy: 240,
    label: "Wf",
    sublabel: "Fuel Manifold",
    stationId: "STA 3.5",
    badgeX: 352,
    badgeY: 272,
    wirePath: "M 380 240 L 380 264 L 355 264 L 355 52 L 605 52",
    nominalMax: 5.5,
  },
  {
    sensorName: "Nc",
    cx: 440,
    cy: 150,
    label: "Nc / Nf",
    sublabel: "Spool Speeds",
    stationId: "STA 4.0",
    badgeX: 412,
    badgeY: 118,
    wirePath: "M 440 150 L 440 52 L 605 52",
    nominalMax: 9400,
  },
  {
    sensorName: "T50",
    cx: 540,
    cy: 190,
    label: "T50",
    sublabel: "Turbine Exhaust",
    stationId: "STA 5.0",
    badgeX: 512,
    badgeY: 215,
    wirePath: "M 540 190 L 540 52 L 605 52",
    nominalMax: 1380,
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
  const isAnomalous = stepData?.is_anomaly || stepData?.anomaly?.is_anomalous || false;
  const isUnreliable = stepData?.is_unreliable || false;
  const stateId = stepData?.state_id;

  // Extract key aerothermal telemetry
  const t2 = stepData?.sensor_readings?.T2 ?? 512.4;
  const t24 = stepData?.sensor_readings?.T24 ?? 618.5;
  const t30 = stepData?.sensor_readings?.T30 ?? 1490.2;
  const t50 = stepData?.sensor_readings?.T50 ?? 1285.4;
  const wf = stepData?.sensor_readings?.Wf ?? 3.65;
  const nc = stepData?.sensor_readings?.Nc ?? 8850.0;
  const nf = stepData?.sensor_readings?.Nf ?? 2150.0;
  const alt = stepData?.sensor_readings?.alt ?? 25000.0;
  const mach = stepData?.sensor_readings?.Mach ?? 0.65;
  const tra = stepData?.sensor_readings?.TRA ?? 65.0;

  // Heat & situational metrics
  const isOverheat = t50 > 1360 || t30 > 1650 || isAnomalous;
  const isCriticalHeat = t50 > 1400 || isUnreliable;
  const coreHeatNormalized = Math.min(1, Math.max(0, (t30 - 1420) / 320));
  const exhaustHeatRatio = Math.min(1, Math.max(0, (t50 - 1240) / 240));

  // Dynamic fan rotation period based on spool speed Nc (0.2s to 0.9s)
  const fanSpinSec = Math.max(0.2, Math.min(1.0, (8850 / (nc || 8850)) * 0.45));

  // Determine situational banner text & tone
  let regimeLabel = "NOMINAL CRUISE";
  let regimeColor = "#10B981";
  let regimeBg = "#ECFDF5";
  let regimeBorder = "#A7F3D0";

  if (isUnreliable) {
    regimeLabel = "SAFETY GUARD: UNRELIABLE CODEBOOK";
    regimeColor = "#DC2626";
    regimeBg = "#FEF2F2";
    regimeBorder = "#FCA5A5";
  } else if (isAnomalous || stateId === 4 || stateId === 5) {
    regimeLabel = "ALERT: ACCELERATING THERMAL RUNAWAY";
    regimeColor = "#DC2626";
    regimeBg = "#FEF2F2";
    regimeBorder = "#FCA5A5";
  } else if (tra > 75 || stateId === 2 || stateId === 3) {
    regimeLabel = "CLIMB / HIGH THROTTLE TRANSITION";
    regimeColor = "#D97706";
    regimeBg = "#FFFBEB";
    regimeBorder = "#FDE68A";
  }

  // Get live formatted string for any sensor
  const getSensorDisplay = (name: string) => {
    const val = stepData?.sensor_readings?.[name];
    if (val === undefined) return "—";
    if (name === "Nc" || name === "Nf") return `${val.toFixed(0)} rpm`;
    if (name === "Wf") return `${val.toFixed(2)} pps`;
    return `${val.toFixed(1)} °R`;
  };

  return (
    <div className="relative flex flex-col bg-white border border-[#E5E7EB] rounded-lg p-4 shadow-sm select-none">
      {/* Header & Academic Surrogate Badge */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#F3F4F6] pb-3 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-2.5 h-2.5 rounded-full ${
                isCriticalHeat
                  ? "bg-[#EF4444] animate-ping"
                  : isOverheat
                  ? "bg-[#F59E0B] animate-pulse"
                  : isProcessing
                  ? "bg-[#3563FF] animate-ping"
                  : "bg-[#10B981]"
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
      <div className="relative w-full h-[370px] bg-[#F9FAFB] border border-[#E5E7EB] rounded overflow-hidden flex items-center justify-center">
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
                @keyframes fanBladeSpin {
                  from { transform: rotate(0deg); }
                  to { transform: rotate(360deg); }
                }
                @keyframes heatShimmer {
                  0% { opacity: 0.6; transform: scaleY(1); }
                  50% { opacity: 0.9; transform: scaleY(1.08); }
                  100% { opacity: 0.6; transform: scaleY(1); }
                }
                .telemetry-active-wire {
                  stroke-dasharray: 6 4;
                  animation: telemetryFlow 0.8s linear infinite;
                }
                .telemetry-idle-wire {
                  stroke-dasharray: 2 6;
                  animation: telemetryFlow 2.4s linear infinite;
                }
                .fan-rotor {
                  transform-origin: 150px 190px;
                  animation: fanBladeSpin ${fanSpinSec}s linear infinite;
                }
                .heat-shimmer {
                  animation: heatShimmer 1.8s ease-in-out infinite;
                  transform-origin: 375px 190px;
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

            {/* Dynamic Core Thermal Gradient */}
            <linearGradient id="coreThermalGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#E2E8F0" />
              <stop offset="25%" stopColor={isOverheat ? "#FED7AA" : "#CBD5E1"} />
              <stop
                offset="60%"
                stopColor={
                  isCriticalHeat
                    ? "#FCA5A5"
                    : isOverheat
                    ? "#FDBA74"
                    : "#F1F5F9"
                }
              />
              <stop
                offset="85%"
                stopColor={
                  isCriticalHeat
                    ? "#EF4444"
                    : isOverheat
                    ? "#FB923C"
                    : "#E2E8F0"
                }
              />
              <stop
                offset="100%"
                stopColor={
                  isCriticalHeat
                    ? "#DC2626"
                    : isOverheat
                    ? "#F97316"
                    : "#D1D5DB"
                }
              />
            </linearGradient>

            {/* Dynamic Exhaust Plume Gradient */}
            <linearGradient id="plumeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop
                offset="0%"
                stopColor={
                  isCriticalHeat
                    ? "#EF4444"
                    : isOverheat
                    ? "#F59E0B"
                    : "#9CA3AF"
                }
                stopOpacity={0.8}
              />
              <stop
                offset="100%"
                stopColor={
                  isCriticalHeat
                    ? "#DC2626"
                    : isOverheat
                    ? "#D97706"
                    : "#E5E7EB"
                }
                stopOpacity={0.0}
              />
            </linearGradient>

            {/* Blue glow filter for active telemetry wiring */}
            <filter id="blueGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            {/* Heat glow filter for combustion/exhaust */}
            <filter id="heatGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            {/* Marker shadow filter */}
            <filter id="markerShadow" x="-10%" y="-10%" width="120%" height="120%">
              <feDropShadow dx="0" dy="1" stdDeviation="1.5" floodColor="#000000" floodOpacity="0.15" />
            </filter>
          </defs>

          {/* Background Engineering Grid */}
          <rect width="680" height="340" fill="url(#engineeringGrid)" />

          {/* ========================================================================= */}
          {/* TACTICAL OPERATING REGIME HUD TAPE (y=14..32)                             */}
          {/* ========================================================================= */}
          <g>
            <rect
              x="60"
              y="14"
              width="535"
              height="20"
              rx="3"
              fill={regimeBg}
              stroke={regimeBorder}
              strokeWidth="1"
            />
            {/* Left Status Pill */}
            <circle cx="72" cy="24" r="3.5" fill={regimeColor} className={isOverheat ? "animate-ping" : ""} />
            <circle cx="72" cy="24" r="2.5" fill={regimeColor} />
            <text
              x="82"
              y="27"
              fontSize="7.5"
              fontFamily="monospace"
              fontWeight="bold"
              fill={regimeColor}
            >
              {regimeLabel}
            </text>

            {/* Flight Envelope Parameters */}
            <text
              x="250"
              y="27"
              fontSize="7"
              fontFamily="monospace"
              fill="#4B5563"
            >
              ALT: {alt.toFixed(0)} ft • M: {mach.toFixed(2)} • TRA: {tra.toFixed(1)}%
            </text>

            {/* Core Thermal Load Bar */}
            <text
              x="425"
              y="27"
              fontSize="7"
              fontFamily="monospace"
              fill="#4B5563"
            >
              T_CORE: {t30.toFixed(0)}°R
            </text>
            <rect x="490" y="20" width="40" height="7" rx="1.5" fill="#E5E7EB" />
            <rect
              x="490"
              y="20"
              width={Math.round(coreHeatNormalized * 40)}
              height="7"
              rx="1.5"
              fill={isCriticalHeat ? "#EF4444" : isOverheat ? "#F59E0B" : "#3563FF"}
            />
            <text
              x="538"
              y="26"
              fontSize="6"
              fontFamily="monospace"
              fill={isOverheat ? "#DC2626" : "#6B7280"}
              fontWeight="bold"
            >
              {((coreHeatNormalized) * 100).toFixed(0)}%
            </text>
          </g>

          {/* Test Cell Mounting Infrastructure */}
          {/* Ceiling Gantry Rail */}
          <line x1="60" y1="36" x2="600" y2="36" stroke="#4B5563" strokeWidth="2.5" />

          {/* Suspension Pylons (attaches to Fan/LPC and Turbine casing) */}
          <polygon
            points="195,36 215,36 208,110 202,110"
            fill="#E5E7EB"
            stroke="#6B7280"
            strokeWidth="1.2"
          />
          <circle cx="205" cy="110" r="3.5" fill="#171717" stroke="#9CA3AF" strokeWidth="1" />

          <polygon
            points="465,36 485,36 478,115 472,115"
            fill="#E5E7EB"
            stroke="#6B7280"
            strokeWidth="1.2"
          />
          <circle cx="475" cy="115" r="3.5" fill="#171717" stroke="#9CA3AF" strokeWidth="1" />

          {/* Ground Support Pedestals & Load Cells */}
          <rect x="180" y="270" width="40" height="60" fill="#E5E7EB" stroke="#9CA3AF" strokeWidth="1.5" />
          <rect x="460" y="270" width="40" height="60" fill="#E5E7EB" stroke="#9CA3AF" strokeWidth="1.5" />
          <circle
            cx="200"
            cy="270"
            r="3.5"
            fill={isOverheat ? "#EF4444" : "#171717"}
            className={isOverheat ? "animate-pulse" : ""}
          />
          <circle
            cx="480"
            cy="270"
            r="3.5"
            fill={isOverheat ? "#EF4444" : "#171717"}
            className={isOverheat ? "animate-pulse" : ""}
          />
          <line x1="150" y1="330" x2="530" y2="330" stroke="#171717" strokeWidth="2" />
          {/* Ground Hatchings */}
          <path
            d="M 160 335 L 170 330 M 190 335 L 200 330 M 220 335 L 230 330 M 440 335 L 450 330 M 470 335 L 480 330 M 500 335 L 510 330"
            stroke="#9CA3AF"
            strokeWidth="1"
          />

          {/* OVERHEAD CABLE LADDER (Wireway Trunk at y=48..56) */}
          <line x1="90" y1="48" x2="605" y2="48" stroke="#9CA3AF" strokeWidth="1.5" />
          <line x1="90" y1="56" x2="605" y2="56" stroke="#9CA3AF" strokeWidth="1.5" />
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

          {/* ========================================================================= */}
          {/* ENGINE NACELLE & CORE GEOMETRY (DYNAMIC THERMAL SHADING)                  */}
          {/* ========================================================================= */}

          {/* Outer Nacelle Cowling Profile */}
          <path
            d="M 90 120 C 120 100, 200 95, 300 95 C 400 95, 480 100, 580 140 L 570 240 C 480 280, 400 285, 300 285 C 200 285, 120 280, 90 260 Z"
            fill="#FFFFFF"
            stroke="#4B5563"
            strokeWidth="1.8"
          />
          {/* Intake Cowl Lip */}
          <ellipse cx="95" cy="190" rx="20" ry="70" fill="#F3F4F6" stroke="#171717" strokeWidth="2" />

          {/* Dynamic Rotating Fan Blades Assembly */}
          <g className="fan-rotor">
            {/* Fan Blades Disk (8 blades radiating from hub) */}
            {Array.from({ length: 8 }).map((_, bIdx) => {
              const angle = (bIdx * 45 * Math.PI) / 180;
              const bx = 150 + Math.cos(angle) * 58;
              const by = 190 + Math.sin(angle) * 58;
              return (
                <line
                  key={bIdx}
                  x1="150"
                  y1="190"
                  x2={bx}
                  y2={by}
                  stroke="#374151"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  opacity="0.85"
                />
              );
            })}
          </g>

          {/* Stationary Guide Stator Disk */}
          <line x1="158" y1="130" x2="158" y2="250" stroke="#9CA3AF" strokeWidth="1.5" strokeDasharray="3 3" />

          {/* Fan Spinner Center Cone */}
          <path d="M 85 190 L 150 170 L 150 210 Z" fill="#D1D5DB" stroke="#171717" strokeWidth="1.5" />
          <circle cx="85" cy="190" r="3" fill="#3563FF" />

          {/* Core Engine Casing (DYNAMICTHERMAL GRADIENT) */}
          <path
            d="M 170 150 L 520 160 L 520 220 L 170 230 Z"
            fill="url(#coreThermalGrad)"
            stroke={isCriticalHeat ? "#EF4444" : isOverheat ? "#F59E0B" : "#4B5563"}
            strokeWidth="1.5"
          />

          {/* Station 2.0 - 2.4: Low-Pressure Compressor (LPC) Section */}
          <rect x="180" y="155" width="60" height="70" fill="#E5E7EB" stroke="#6B7280" strokeWidth="1" opacity="0.8" />
          <text x="210" y="195" fontSize="8" fill="#4B5563" textAnchor="middle" fontFamily="monospace" fontWeight="bold">LPC</text>

          {/* Station 2.4 - 3.0: High-Pressure Compressor (HPC) Section */}
          <rect
            x="250"
            y="158"
            width="90"
            height="64"
            fill={isOverheat ? "#FEF3C7" : "#E5E7EB"}
            stroke={isOverheat ? "#F59E0B" : "#6B7280"}
            strokeWidth="1"
            opacity="0.85"
          />
          <text x="295" y="195" fontSize="8" fill="#4B5563" textAnchor="middle" fontFamily="monospace" fontWeight="bold">HPC</text>

          {/* Station 3.0 - 3.5: Combustion Chamber (Burner) — Dynamic Flames */}
          <path
            d="M 350 162 L 400 162 L 400 218 L 350 218 Z"
            fill={
              isCriticalHeat
                ? "#FEE2E2"
                : isOverheat
                ? "#FEF3C7"
                : "#FEE2E2"
            }
            stroke={isCriticalHeat ? "#DC2626" : isOverheat ? "#D97706" : "#EF4444"}
            strokeWidth={isOverheat ? 2 : 1.2}
            strokeDasharray={isOverheat ? "none" : "4 2"}
            className={isOverheat ? "heat-shimmer" : ""}
          />
          <text
            x="375"
            y="193"
            fontSize="8"
            fill={isCriticalHeat ? "#B91C1C" : isOverheat ? "#B45309" : "#B91C1C"}
            textAnchor="middle"
            fontFamily="monospace"
            fontWeight="bold"
          >
            BURNER
          </text>

          {/* Station 4.0: High-Pressure Turbine (HPT) */}
          <rect
            x="410"
            y="160"
            width="40"
            height="60"
            fill={isCriticalHeat ? "#FEE2E2" : isOverheat ? "#FED7AA" : "#E5E7EB"}
            stroke={isCriticalHeat ? "#EF4444" : "#6B7280"}
            strokeWidth="1"
          />
          <text x="430" y="195" fontSize="8" fill="#4B5563" textAnchor="middle" fontFamily="monospace" fontWeight="bold">HPT</text>

          {/* Station 5.0: Low-Pressure Turbine (LPT) */}
          <rect
            x="460"
            y="163"
            width="50"
            height="54"
            fill={isCriticalHeat ? "#FEE2E2" : isOverheat ? "#FED7AA" : "#E5E7EB"}
            stroke={isCriticalHeat ? "#EF4444" : "#6B7280"}
            strokeWidth="1"
          />
          <text x="485" y="195" fontSize="8" fill="#4B5563" textAnchor="middle" fontFamily="monospace" fontWeight="bold">LPT</text>

          {/* Core Exhaust Nozzle & Centerbody */}
          <path
            d="M 520 170 L 590 180 L 590 200 L 520 210 Z"
            fill={isCriticalHeat ? "#FCA5A5" : "#D1D5DB"}
            stroke={isCriticalHeat ? "#DC2626" : "#171717"}
            strokeWidth="1.5"
          />

          {/* DYNAMIC HIGH-VELOCITY JET EXHAUST PLUME */}
          <g>
            {/* Outer Exhaust Plume Core */}
            <path
              d="M 590 180 C 625 170, 650 160, 675 155 L 675 225 C 650 220, 625 210, 590 200 Z"
              fill="url(#plumeGrad)"
              opacity={isOverheat ? 0.75 : 0.35}
            />
            {/* Streamline Vectors */}
            <path
              d="M 590 185 C 620 180, 645 175, 670 170 M 590 195 C 620 200, 645 205, 670 210"
              stroke={isCriticalHeat ? "#EF4444" : isOverheat ? "#F59E0B" : "#9CA3AF"}
              strokeWidth={isOverheat ? 1.8 : 1.2}
              strokeDasharray="4 3"
              className="telemetry-active-wire"
            />
            {/* Centerline Mach Streak */}
            <line
              x1="590"
              y1="190"
              x2="675"
              y2="190"
              stroke={isCriticalHeat ? "#FFFFFF" : isOverheat ? "#FDE68A" : "#D1D5DB"}
              strokeWidth={isOverheat ? 2 : 1}
              strokeDasharray="6 4"
              className="telemetry-active-wire"
            />
          </g>

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
                <path
                  d={station.wirePath}
                  fill="none"
                  stroke={isSelected ? "#3563FF" : "#D1D5DB"}
                  strokeWidth={isSelected ? 2.5 : 1.2}
                  opacity={isSelected ? 1.0 : 0.6}
                />
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
                  <path
                    d={station.wirePath}
                    fill="none"
                    stroke="#3563FF"
                    strokeWidth="3.5"
                    opacity="0.3"
                    filter="url(#blueGlow)"
                  />
                  <path
                    d={station.wirePath}
                    fill="none"
                    stroke="#FFFFFF"
                    strokeWidth="1.8"
                    className="telemetry-active-wire"
                  />
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
            <rect x="601" y="44" width="4" height="16" fill="#4B5563" rx="1" />
            <circle cx="603" cy="52" r="2.5" fill="#171717" />

            <rect
              x="605"
              y="28"
              width="63"
              height="56"
              rx="3"
              fill="#171717"
              stroke={isCriticalHeat ? "#EF4444" : "#374151"}
              strokeWidth={isCriticalHeat ? 2 : 1.5}
            />
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
            <circle cx="616" cy="48" r="2.5" fill="#10B981" />
            <text x="623" y="50" fontSize="5.5" fill="#9CA3AF" fontFamily="monospace">PWR</text>

            <circle
              cx="644"
              cy="48"
              r="2.5"
              fill={isProcessing ? "#3563FF" : "#60A5FA"}
              className={isProcessing ? "animate-pulse" : ""}
            />
            <text x="651" y="50" fontSize="5.5" fill="#9CA3AF" fontFamily="monospace">BUS</text>

            <circle cx="616" cy="60" r="2.5" fill="#8B5CF6" />
            <text x="623" y="62" fontSize="5.5" fill="#9CA3AF" fontFamily="monospace">VQ</text>

            <circle
              cx="644"
              cy="60"
              r="2.5"
              fill={isCriticalHeat ? "#EF4444" : isOverheat ? "#F59E0B" : "#10B981"}
              className={isOverheat ? "animate-ping" : ""}
            />
            <text x="651" y="62" fontSize="5.5" fill="#9CA3AF" fontFamily="monospace">HAZ</text>

            <text
              x="636"
              y="76"
              fontSize="5.5"
              fill={isOverheat ? "#EF4444" : "#60A5FA"}
              textAnchor="middle"
              fontFamily="monospace"
              fontWeight="bold"
            >
              18-CH • 4.2 kHz
            </text>
          </g>

          {/* ========================================================================= */}
          {/* INTERACTIVE SENSOR PROBE HOTSPOTS & LIVE VALUE CALLOUTS                   */}
          {/* ========================================================================= */}
          {SENSOR_STATIONS.map((spot) => {
            const isSelected = selectedSensor.name === spot.sensorName;
            const targetMeta = allSensors.find((s) => s.name === spot.sensorName) || selectedSensor;
            const liveDisplay = getSensorDisplay(spot.sensorName);
            const valNum = stepData?.sensor_readings?.[spot.sensorName] ?? 0;
            const isStationOverheat = valNum > spot.nominalMax;

            return (
              <g
                key={spot.sensorName}
                className="cursor-pointer transition-transform hover:scale-105"
                onClick={() => onSelectSensor(targetMeta)}
                filter="url(#markerShadow)"
              >
                {/* Ping pulse ring when selected or overheating */}
                {(isSelected || isStationOverheat) && (
                  <circle
                    cx={spot.cx}
                    cy={spot.cy}
                    r="13"
                    fill="none"
                    stroke={isStationOverheat ? "#EF4444" : "#3563FF"}
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
                  stroke={isStationOverheat ? "#EF4444" : isSelected ? "#3563FF" : "#9CA3AF"}
                  strokeWidth="0.8"
                />
                <line
                  x1={spot.cx}
                  y1={spot.cy - 9}
                  x2={spot.cx}
                  y2={spot.cy + 9}
                  stroke={isStationOverheat ? "#EF4444" : isSelected ? "#3563FF" : "#9CA3AF"}
                  strokeWidth="0.8"
                />

                {/* Probe Head Outer Circle */}
                <circle
                  cx={spot.cx}
                  cy={spot.cy}
                  r="7"
                  fill={
                    isStationOverheat
                      ? "#FEF2F2"
                      : isSelected
                      ? "#3563FF"
                      : "#FFFFFF"
                  }
                  stroke={
                    isStationOverheat
                      ? "#DC2626"
                      : isSelected
                      ? "#1D4ED8"
                      : "#4B5563"
                  }
                  strokeWidth="2"
                />
                {/* Probe Center Contact Pin */}
                <circle
                  cx={spot.cx}
                  cy={spot.cy}
                  r="2.5"
                  fill={
                    isStationOverheat
                      ? "#DC2626"
                      : isSelected
                      ? "#FFFFFF"
                      : "#3563FF"
                  }
                />

                {/* Probe Station Engineering Callout Badge (NOW WITH LIVE READING) */}
                <rect
                  x={spot.badgeX}
                  y={spot.badgeY}
                  width="56"
                  height="22"
                  rx="3"
                  fill={
                    isStationOverheat
                      ? "#FEF2F2"
                      : isSelected
                      ? "#171717"
                      : "#FFFFFF"
                  }
                  stroke={
                    isStationOverheat
                      ? "#EF4444"
                      : isSelected
                      ? "#3563FF"
                      : "#D1D5DB"
                  }
                  strokeWidth={isSelected || isStationOverheat ? 1.5 : 1}
                />
                {/* Station Label */}
                <text
                  x={spot.badgeX + 28}
                  y={spot.badgeY + 9}
                  fontSize="7.5"
                  fontWeight="bold"
                  fontFamily="monospace"
                  fill={
                    isStationOverheat
                      ? "#B91C1C"
                      : isSelected
                      ? "#FFFFFF"
                      : "#171717"
                  }
                  textAnchor="middle"
                >
                  {spot.label}
                </text>
                {/* Live Telemetry Value */}
                <text
                  x={spot.badgeX + 28}
                  y={spot.badgeY + 18}
                  fontSize="7"
                  fontWeight="600"
                  fontFamily="monospace"
                  fill={
                    isStationOverheat
                      ? "#DC2626"
                      : isSelected
                      ? "#60A5FA"
                      : "#4B5563"
                  }
                  textAnchor="middle"
                >
                  {liveDisplay}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Selected Sensor Floating Info Overlay */}
        <div className="absolute bottom-3 left-3 bg-white/95 backdrop-blur-sm border border-[#E5E7EB] rounded p-2.5 text-xs shadow-sm max-w-[210px] pointer-events-none">
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
            const valNum = stepData?.sensor_readings?.[spot.sensorName] ?? 0;
            const isStationOverheat = valNum > spot.nominalMax;

            return (
              <button
                key={meta.name}
                onClick={() => onSelectSensor(meta)}
                className={`text-xs px-2.5 py-1 rounded font-mono transition-all border flex items-center gap-1.5 ${
                  isSel
                    ? "bg-[#3563FF] text-white border-[#3563FF] shadow-xs font-semibold"
                    : isStationOverheat
                    ? "bg-[#FEF2F2] text-[#B91C1C] border-[#FCA5A5] hover:bg-[#FEE2E2]"
                    : "bg-[#F9FAFB] text-[#374151] border-[#E5E7EB] hover:bg-[#F3F4F6]"
                }`}
              >
                <span>{meta.name}</span>
                <span className={`text-[10px] opacity-80`}>
                  ({getSensorDisplay(meta.name)})
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
