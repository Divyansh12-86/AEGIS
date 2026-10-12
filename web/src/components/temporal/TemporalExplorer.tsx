"use client";

import React from "react";
import { Clock, History, GitCommit, Activity } from "lucide-react";

interface TemporalExplorerProps {
  history: any[];
  stepData: any;
  onSelectStepIndex?: (index: number) => void;
}

export const TemporalExplorer: React.FC<TemporalExplorerProps> = ({
  history,
  stepData,
  onSelectStepIndex,
}) => {
  const currentEvent = stepData?.event_symbol || "E03";
  const currentState = stepData?.state_id !== null && stepData?.state_id !== undefined ? stepData.state_id : "?";
  const currentDwell = stepData?.dwell !== null && stepData?.dwell !== undefined ? stepData.dwell : 0;
  const posterior = stepData?.posterior || [0.2, 0.2, 0.2, 0.2, 0.2];

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-lg p-4 shadow-sm select-none">
      <div className="flex items-center justify-between border-b border-[#F3F4F6] pb-2.5 mb-3">
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-[#171717] font-mono uppercase flex items-center gap-1.5">
            <History className="w-4 h-4 text-[#3563FF]" />
            <span>Temporal Event Grammar & State Occupancy</span>
          </h3>
          <p className="text-xs text-[#6B7280]">
            Discrete Token Sequence • HSMM Posterior • Non-Exponential Dwell τ
          </p>
        </div>
        <div className="text-xs font-mono bg-[#F9FAFB] border border-[#E5E7EB] px-2.5 py-1 rounded text-[#4B5563]">
          Active Dwell: <strong className="text-[#171717]">{currentDwell} steps</strong> in State {currentState}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Left: Event Stream Timeline */}
        <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3">
          <div className="text-[11px] font-mono font-bold text-[#171717] uppercase mb-2 flex items-center justify-between">
            <span>Discovered Event Sequence History</span>
            <span className="text-[10px] text-[#6B7280]">VQ K=16 Tokens</span>
          </div>

          <div className="flex items-center gap-1.5 overflow-x-auto py-2">
            {history.map((h, i) => {
              const isCurrent = i === history.length - 1;
              return (
                <div
                  key={i}
                  className={`flex flex-col items-center px-2 py-1 rounded border font-mono transition-all shrink-0 ${
                    isCurrent
                      ? "bg-[#3563FF] text-white border-[#3563FF] shadow-xs"
                      : "bg-white text-[#374151] border-[#E5E7EB]"
                  }`}
                >
                  <span className="text-[11px] font-bold">{h?.event_symbol || "E??"}</span>
                  <span className={`text-[9px] ${isCurrent ? "text-blue-100" : "text-[#9CA3AF]"}`}>
                    C{h?.cycle ?? i}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="text-[10px] font-mono text-[#6B7280] mt-2 bg-white border border-[#E5E7EB] rounded p-2">
            Current Token: <strong className="text-[#171717]">{currentEvent}</strong> — {stepData?.event_desc || "Operational Token"}
          </div>
        </div>

        {/* Right: State Occupancy & Posterior Probabilities */}
        <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3">
          <div className="text-[11px] font-mono font-bold text-[#171717] uppercase mb-2 flex items-center justify-between">
            <span>HSMM State Posterior Distribution</span>
            <span className="text-[10px] text-[#6B7280]">Forward-Backward FB</span>
          </div>

          <div className="space-y-1.5">
            {[1, 2, 3, 4, 5].map((sId) => {
              const prob = posterior[sId - 1] || 0;
              const isDominant = sId === currentState;

              return (
                <div key={sId} className="flex items-center gap-2 text-xs font-mono">
                  <span className={`w-14 truncate ${isDominant ? "font-bold text-[#171717]" : "text-[#6B7280]"}`}>
                    State {sId}
                  </span>
                  <div className="flex-1 bg-white border border-[#E5E7EB] rounded-full h-3 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${
                        isDominant ? "bg-[#3563FF]" : "bg-[#D1D5DB]"
                      }`}
                      style={{ width: `${Math.round(prob * 100)}%` }}
                    />
                  </div>
                  <span className="w-10 text-right font-bold text-[#171717]">
                    {(prob * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>

          <div className="text-[10px] font-mono text-[#6B7280] mt-2 bg-white border border-[#E5E7EB] rounded p-2 flex items-center justify-between">
            <span>Dwell Hazard Multiplier:</span>
            <span className="font-bold text-[#171717]">
              {stepData?.rul?.dwell_effect_cycles ? `${stepData.rul.dwell_effect_cycles.toFixed(1)} cycles` : "Nominal"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
