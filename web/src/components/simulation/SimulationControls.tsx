"use client";

import React from "react";
import { Play, Pause, SkipForward, RotateCcw, FastForward, Server, Laptop } from "lucide-react";
import { SCENARIOS } from "@/lib/scenarios";

interface SimulationControlsProps {
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStepForward: () => void;
  onReset: () => void;
  speed: number;
  onChangeSpeed: (speed: number) => void;
  currentScenarioId: string;
  onSelectScenario: (scenarioId: string) => void;
  isLiveBackend: boolean;
  currentStepIndex: number;
  totalSteps: number;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  isPlaying,
  onTogglePlay,
  onStepForward,
  onReset,
  speed,
  onChangeSpeed,
  currentScenarioId,
  onSelectScenario,
  isLiveBackend,
  currentStepIndex,
  totalSteps,
}) => {
  return (
    <div className="bg-white border border-[#E5E7EB] rounded-lg p-3 shadow-sm select-none flex flex-wrap items-center justify-between gap-3">
      {/* Scenario Selector & Status Indicator */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 font-mono text-xs font-semibold text-[#171717]">
          <span>Scenario:</span>
          <select
            value={currentScenarioId}
            onChange={(e) => onSelectScenario(e.target.value)}
            className="bg-[#F9FAFB] border border-[#E5E7EB] rounded px-2.5 py-1 text-xs font-mono text-[#171717] focus:outline-none focus:border-[#3563FF]"
          >
            {Object.values(SCENARIOS).map((s) => (
              <option key={s.id} value={s.id}>
                {s.title}
              </option>
            ))}
          </select>
        </div>

        {/* Backend / Replay Mode Indicator Badge */}
        <div
          className={`flex items-center gap-1.5 text-[11px] font-mono px-2.5 py-1 rounded border ${
            isLiveBackend
              ? "bg-[#ECFDF5] border-[#A7F3D0] text-[#065F46]"
              : "bg-[#EFF6FF] border-[#BFDBFE] text-[#1E3A8A]"
          }`}
        >
          {isLiveBackend ? (
            <>
              <Server className="w-3.5 h-3.5 text-[#059669]" />
              <span>● LIVE FASTAPI BACKEND (:8000)</span>
            </>
          ) : (
            <>
              <Laptop className="w-3.5 h-3.5 text-[#3563FF]" />
              <span>● DETERMINISTIC RECORDED REPLAY</span>
            </>
          )}
        </div>
      </div>

      {/* Main Simulation Playback Controller */}
      <div className="flex items-center gap-2">
        {/* Play / Pause Button */}
        <button
          onClick={onTogglePlay}
          className={`px-3 py-1.5 rounded font-mono text-xs font-semibold flex items-center gap-1.5 transition-all shadow-xs ${
            isPlaying
              ? "bg-[#D97706] text-white hover:bg-[#B45309]"
              : "bg-[#3563FF] text-white hover:bg-[#1D4ED8]"
          }`}
        >
          {isPlaying ? (
            <>
              <Pause className="w-3.5 h-3.5" />
              <span>Pause</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5" />
              <span>Start Simulation</span>
            </>
          )}
        </button>

        {/* Step Button */}
        <button
          onClick={onStepForward}
          disabled={isPlaying}
          className="px-2.5 py-1.5 rounded bg-[#F9FAFB] border border-[#E5E7EB] hover:bg-[#F3F4F6] text-[#374151] font-mono text-xs font-semibold flex items-center gap-1 disabled:opacity-50 transition-colors"
          title="Step One Window (+1 Cycle)"
        >
          <SkipForward className="w-3.5 h-3.5" />
          <span>Step Window</span>
        </button>

        {/* Reset Button */}
        <button
          onClick={onReset}
          className="p-1.5 rounded bg-[#F9FAFB] border border-[#E5E7EB] hover:bg-[#F3F4F6] text-[#6B7280] hover:text-[#171717] transition-colors"
          title="Reset Scenario"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        {/* Playback Speed Selector */}
        <div className="flex items-center border border-[#E5E7EB] rounded overflow-hidden text-[10px] font-mono">
          {[0.5, 1.0, 2.0].map((s) => (
            <button
              key={s}
              onClick={() => onChangeSpeed(s)}
              className={`px-2 py-1 transition-colors ${
                speed === s
                  ? "bg-[#171717] text-white font-bold"
                  : "bg-white text-[#6B7280] hover:bg-[#F9FAFB]"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Step Counter Indicator */}
        <div className="text-[11px] font-mono text-[#6B7280] px-2 py-1 bg-[#F9FAFB] rounded border border-[#E5E7EB]">
          Step {currentStepIndex + 1} / {totalSteps}
        </div>
      </div>
    </div>
  );
};
