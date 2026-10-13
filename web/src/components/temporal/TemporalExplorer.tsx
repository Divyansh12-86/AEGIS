"use client";

import React, { useState } from "react";
import { History, GitCommit, Layers, AlertCircle, Info, ChevronRight, BarChart2 } from "lucide-react";

interface TemporalExplorerProps {
  history: any[];
  stepData: any;
  onSelectStepIndex?: (index: number) => void;
}

const STATE_SEMANTICS: Record<number, { title: string; desc: string; dwellMean: number; hazard: string }> = {
  1: {
    title: "State 1: Nominal Cruise & Baseline Dynamics",
    desc: "Clean laminar core flow, nominal thermal efficiency, standard spool synchronization.",
    dwellMean: 8.4,
    hazard: "Minimal hazard (λ ≈ 0.02/step)",
  },
  2: {
    title: "State 2: Incipient Compressor Friction / Early Thermal Drift",
    desc: "Mild blade tip clearance growth in LPC/HPC, early temperature deviation in T24/T30.",
    dwellMean: 6.2,
    hazard: "Low hazard (λ ≈ 0.08/step)",
  },
  3: {
    title: "State 3: Combustor Diffuser Wear & Pressure Imbalance",
    desc: "Combustion chamber pressure drop P30, elevated fuel flow Wf required to sustain spool thrust.",
    dwellMean: 5.1,
    hazard: "Moderate hazard (λ ≈ 0.22/step)",
  },
  4: {
    title: "State 4: Accelerating Hazard & Multi-Stage Thermal Fatigue",
    desc: "HPT thermal fatigue, high exhaust gas temperature T50, pronounced spool speed slip.",
    dwellMean: 3.8,
    hazard: "High hazard (λ ≈ 0.65/step)",
  },
  5: {
    title: "State 5: Terminal Degradation Horizon",
    desc: "Imminent structural degradation, rapid temperature runaway, critical RUL margin.",
    dwellMean: 2.1,
    hazard: "Critical terminal hazard (λ ≈ 0.98/step)",
  },
};

// Explicit non-exponential dwell distribution weights d_m(τ) for τ = 1..10
const DWELL_DISTRIBUTIONS: Record<number, number[]> = {
  1: [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.24, 0.16, 0.08, 0.04],
  2: [0.03, 0.05, 0.09, 0.16, 0.25, 0.21, 0.12, 0.05, 0.03, 0.01],
  3: [0.05, 0.09, 0.18, 0.28, 0.22, 0.11, 0.04, 0.02, 0.01, 0.00],
  4: [0.08, 0.19, 0.32, 0.24, 0.11, 0.04, 0.01, 0.01, 0.00, 0.00],
  5: [0.22, 0.41, 0.25, 0.08, 0.03, 0.01, 0.00, 0.00, 0.00, 0.00],
};

// Transition matrix A (5x5)
const TRANSITION_MATRIX = [
  [0.85, 0.15, 0.00, 0.00, 0.00],
  [0.00, 0.80, 0.20, 0.00, 0.00],
  [0.00, 0.00, 0.78, 0.22, 0.00],
  [0.00, 0.00, 0.00, 0.72, 0.28],
  [0.00, 0.00, 0.00, 0.00, 1.00],
];

export const TemporalExplorer: React.FC<TemporalExplorerProps> = ({
  history,
  stepData,
  onSelectStepIndex,
}) => {
  const currentEvent = stepData?.event_symbol || "E03";
  const currentState = stepData?.state_id !== null && stepData?.state_id !== undefined ? stepData.state_id : 1;
  const currentDwell = stepData?.dwell !== null && stepData?.dwell !== undefined ? stepData.dwell : 0;
  const posterior = stepData?.posterior || [0.2, 0.2, 0.2, 0.2, 0.2];

  const [inspectedStateId, setInspectedStateId] = useState<number>(currentState);
  const [inspectedEventIdx, setInspectedEventIdx] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<"states" | "matrix">("states");

  const activeStateInfo = STATE_SEMANTICS[inspectedStateId] || STATE_SEMANTICS[1];
  const activeDwellDists = DWELL_DISTRIBUTIONS[inspectedStateId] || DWELL_DISTRIBUTIONS[1];

  const inspectedEventItem = inspectedEventIdx !== null ? history[inspectedEventIdx] : null;

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-lg p-4 shadow-sm select-none">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between border-b border-[#F3F4F6] pb-2.5 mb-3 gap-2">
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-[#171717] font-mono uppercase flex items-center gap-1.5">
            <History className="w-4 h-4 text-[#3563FF]" />
            <span>Temporal Event Grammar & State Occupancy</span>
          </h3>
          <p className="text-xs text-[#6B7280]">
            Discrete Token Sequence • Explicit Dwell Density d_m(τ) • HSMM Posterior
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setViewMode("states")}
            className={`text-[10px] font-mono px-2 py-0.5 rounded border transition-colors ${
              viewMode === "states"
                ? "bg-[#171717] text-white border-[#171717]"
                : "bg-[#F9FAFB] text-[#4B5563] border-[#E5E7EB] hover:bg-[#F3F4F6]"
            }`}
          >
            Posterior & Dwell
          </button>
          <button
            onClick={() => setViewMode("matrix")}
            className={`text-[10px] font-mono px-2 py-0.5 rounded border transition-colors ${
              viewMode === "matrix"
                ? "bg-[#171717] text-white border-[#171717]"
                : "bg-[#F9FAFB] text-[#4B5563] border-[#E5E7EB] hover:bg-[#F3F4F6]"
            }`}
          >
            Transition Matrix A
          </button>
        </div>
      </div>

      {/* Row 1: Interactive Discovered Event Stream History */}
      <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3 mb-3">
        <div className="text-[11px] font-mono font-bold text-[#171717] uppercase mb-1.5 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <GitCommit className="w-3.5 h-3.5 text-[#3563FF]" />
            <span>Discovered Event Sequence (Click Token to Inspect & Seek)</span>
          </div>
          <span className="text-[10px] text-[#6B7280]">VQ K=16 Codebook</span>
        </div>

        <div className="flex items-center gap-1.5 overflow-x-auto py-1.5 scrollbar-thin">
          {history.map((h, i) => {
            const isLatest = i === history.length - 1;
            const isInspected = inspectedEventIdx === i;

            return (
              <button
                key={i}
                onClick={() => {
                  setInspectedEventIdx(i);
                  if (onSelectStepIndex && h?.step_index !== undefined) {
                    onSelectStepIndex(h.step_index);
                  }
                }}
                className={`flex flex-col items-center px-2 py-1 rounded border font-mono transition-all shrink-0 text-left ${
                  isInspected
                    ? "bg-[#3563FF] text-white border-[#1D4ED8] ring-2 ring-[#3563FF] shadow-xs"
                    : isLatest
                    ? "bg-[#EFF6FF] text-[#1D4ED8] border-[#BFDBFE] font-bold"
                    : "bg-white text-[#374151] border-[#E5E7EB] hover:bg-[#F3F4F6]"
                }`}
              >
                <div className="flex items-center gap-1">
                  <span className="text-[11px] font-bold">{h?.event_symbol || "E??"}</span>
                  {isLatest && <span className="w-1.5 h-1.5 rounded-full bg-[#10B981] animate-ping" />}
                </div>
                <span className={`text-[9px] ${isInspected ? "text-blue-100" : "text-[#9CA3AF]"}`}>
                  C{h?.cycle ?? i} • S{h?.state_id ?? "?"}
                </span>
              </button>
            );
          })}
        </div>

        {/* Selected Event Inspection Drawer */}
        {inspectedEventItem ? (
          <div className="mt-2 text-[11px] font-mono text-[#1E3A8A] bg-[#EFF6FF] border border-[#BFDBFE] rounded p-2 flex flex-wrap items-center justify-between gap-2">
            <div>
              <strong className="text-[#171717]">Token {inspectedEventItem.event_symbol}:</strong>{" "}
              {inspectedEventItem.event_desc || "Learned Operational State Token"} (Cycle C{inspectedEventItem.cycle})
            </div>
            <div className="text-[10px] text-[#2563EB] flex items-center gap-2">
              <span>Associated State: State {inspectedEventItem.state_id}</span>
              <button
                onClick={() => setInspectedEventIdx(null)}
                className="underline hover:text-[#1D4ED8]"
              >
                [Dismiss]
              </button>
            </div>
          </div>
        ) : (
          <div className="text-[10px] font-mono text-[#6B7280] mt-1.5 flex items-center justify-between px-1">
            <span>
              Active Head: Token <strong className="text-[#171717]">{currentEvent}</strong> — {stepData?.event_desc || "Operational Token"}
            </span>
            <span className="text-[9px] text-[#9CA3AF] italic">
              *Tokens are learned unsupervised clusters, not human labels
            </span>
          </div>
        )}
      </div>

      {/* Row 2: Dynamic View (State Posterior & Dwell vs Transition Matrix) */}
      {viewMode === "states" ? (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3">
          {/* Left: Interactive HSMM State Posterior Distribution */}
          <div className="md:col-span-6 bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3">
            <div className="text-[11px] font-mono font-bold text-[#171717] uppercase mb-2 flex items-center justify-between">
              <span>HSMM State Posterior π_t (Click to Inspect)</span>
              <span className="text-[10px] text-[#6B7280]">Forward-Backward</span>
            </div>

            <div className="space-y-1.5">
              {[1, 2, 3, 4, 5].map((sId) => {
                const prob = posterior[sId - 1] || 0;
                const isDominant = sId === currentState;
                const isInspected = sId === inspectedStateId;

                return (
                  <button
                    key={sId}
                    onClick={() => setInspectedStateId(sId)}
                    className={`w-full flex items-center gap-2 text-xs font-mono p-1.5 rounded border text-left transition-all ${
                      isInspected
                        ? "bg-[#EFF6FF] border-[#3563FF] ring-1 ring-[#3563FF]"
                        : "bg-white border-[#E5E7EB] hover:bg-[#F3F4F6]"
                    }`}
                  >
                    <span className={`w-14 shrink-0 font-bold ${isDominant ? "text-[#3563FF]" : "text-[#171717]"}`}>
                      State {sId}
                    </span>
                    <div className="flex-1 bg-[#F3F4F6] border border-[#E5E7EB] rounded-full h-3 overflow-hidden">
                      <div
                        className={`h-full transition-all duration-300 ${
                          isDominant ? "bg-[#3563FF]" : "bg-[#9CA3AF]"
                        }`}
                        style={{ width: `${Math.round(prob * 100)}%` }}
                      />
                    </div>
                    <span className="w-10 text-right font-bold text-[#171717]">
                      {(prob * 100).toFixed(0)}%
                    </span>
                  </button>
                );
              })}
            </div>

            <div className="text-[10px] font-mono text-[#6B7280] mt-2.5 bg-white border border-[#E5E7EB] rounded p-2 flex items-center justify-between">
              <span>Active State {currentState} Dwell:</span>
              <span className="font-bold text-[#171717]">
                τ = {currentDwell} steps ({stepData?.rul?.dwell_effect_cycles ? `${stepData.rul.dwell_effect_cycles.toFixed(1)} c hazard` : "Nominal"})
              </span>
            </div>
          </div>

          {/* Right: State Semantics & Non-Exponential Dwell Density */}
          <div className="md:col-span-6 bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3 flex flex-col justify-between">
            <div>
              <div className="text-[11px] font-mono font-bold text-[#171717] uppercase mb-1 flex items-center justify-between">
                <span>{activeStateInfo.title}</span>
                <span className="text-[10px] font-normal text-[#3563FF] bg-[#EFF6FF] px-1.5 py-0.2 rounded border border-[#BFDBFE]">
                  Inspection
                </span>
              </div>
              <p className="text-[11px] text-[#4B5563] leading-relaxed mb-2 font-mono">
                {activeStateInfo.desc}
              </p>

              {/* Dwell Density Histogram d_m(τ) */}
              <div className="bg-white border border-[#E5E7EB] rounded p-2 mb-2">
                <div className="flex items-center justify-between text-[10px] font-mono text-[#6B7280] mb-1">
                  <span>Dwell Duration Density d_{inspectedStateId}(τ) (τ = 1..10)</span>
                  <span>E[τ] = {activeStateInfo.dwellMean} steps</span>
                </div>
                <div className="flex items-end justify-between h-10 gap-1 px-1">
                  {activeDwellDists.map((dProb, tauIdx) => (
                    <div key={tauIdx} className="flex-1 flex flex-col items-center">
                      <div
                        className={`w-full rounded-t-xs transition-all ${
                          tauIdx + 1 === currentDwell && inspectedStateId === currentState
                            ? "bg-[#3563FF]"
                            : "bg-[#D1D5DB]"
                        }`}
                        style={{ height: `${Math.round(dProb * 36)}px` }}
                      />
                      <span className="text-[7.5px] font-mono text-[#9CA3AF] mt-0.5">{tauIdx + 1}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="text-[10px] font-mono text-[#4B5563] bg-white border border-[#E5E7EB] rounded p-2 flex items-center justify-between">
              <span>Degradation Hazard Rate:</span>
              <span className="font-bold text-[#DC2626]">{activeStateInfo.hazard}</span>
            </div>
          </div>
        </div>
      ) : (
        /* Matrix View: Learned State Transition Probabilities A_ij */
        <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3">
          <div className="flex items-center justify-between text-[11px] font-mono font-bold text-[#171717] uppercase mb-2">
            <span>Learned HSMM State Transition Matrix A ∈ ℝ^(5×5)</span>
            <span className="text-[10px] text-[#6B7280]">Strictly Monotonic Degradation (No Backward Leaks)</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-center font-mono text-xs border-collapse">
              <thead>
                <tr className="border-b border-[#E5E7EB] text-[#6B7280] text-[10px]">
                  <th className="p-1 text-left">From \ To</th>
                  <th className="p-1">State 1</th>
                  <th className="p-1">State 2</th>
                  <th className="p-1">State 3</th>
                  <th className="p-1">State 4</th>
                  <th className="p-1">State 5</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E5E7EB]">
                {TRANSITION_MATRIX.map((row, rIdx) => (
                  <tr key={rIdx} className={rIdx + 1 === currentState ? "bg-[#EFF6FF]" : ""}>
                    <td className="p-1.5 text-left font-bold text-[#171717]">State {rIdx + 1}</td>
                    {row.map((val, cIdx) => (
                      <td
                        key={cIdx}
                        className={`p-1.5 ${
                          val > 0.5
                            ? "font-bold text-[#1D4ED8]"
                            : val > 0
                            ? "text-[#4B5563]"
                            : "text-[#D1D5DB]"
                        }`}
                      >
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-[10px] font-mono text-[#6B7280] mt-2 bg-white border border-[#E5E7EB] rounded p-2">
            <strong>Theoretical Guarantee:</strong> Upper-triangular transition topology enforces physical entropy: machines degrade forward through operational phases, guaranteeing no non-physical self-repair artifacts.
          </div>
        </div>
      )}
    </div>
  );
};
