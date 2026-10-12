"use client";

import React, { useState } from "react";
import { ArrowRight, ShieldCheck, AlertTriangle, Layers, Activity, Cpu, Sparkles, HelpCircle } from "lucide-react";

interface EGPMPipelineProps {
  stepData: any;
  isProcessing: boolean;
  activeStageIndex: number;
  onOpenEvaluation: () => void;
}

const HEATMAP_OPACITIES = [0.2, 0.4, 0.7, 0.3, 0.5, 0.8, 0.3, 0.6, 0.2, 0.5, 0.9, 0.4, 0.6, 0.3, 0.7, 0.5];
const LATENT_BAR_HEIGHTS = [28, 14, 22, 30, 16, 26, 12, 24];
const LATENT_BAR_OPACITIES = [0.4, 0.7, 1.0, 0.4, 0.7, 1.0, 0.4, 0.7];

export const EGPMPipeline: React.FC<EGPMPipelineProps> = ({
  stepData,
  isProcessing,
  activeStageIndex,
  onOpenEvaluation,
}) => {
  const [inspectedStage, setInspectedStage] = useState<number | null>(null);

  const eventSymbol = stepData?.event_symbol || "E03";
  const stateId = stepData?.state_id !== null && stepData?.state_id !== undefined ? stepData.state_id : "?";
  const dwell = stepData?.dwell !== null && stepData?.dwell !== undefined ? stepData.dwell : "—";
  const posterior = stepData?.posterior || [0.2, 0.2, 0.2, 0.2, 0.2];
  const rulExpected = stepData?.rul?.expected;
  const rulCi = stepData?.rul?.ci90 || [null, null];
  const isUnreliable = stepData?.is_unreliable || false;

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-lg p-4 shadow-sm select-none">
      {/* Header with Visual Flow Title */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#F3F4F6] pb-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold tracking-wide text-[#171717] uppercase font-mono">
              EGPM Neuro-Symbolic Processing Flowchart
            </h2>
            <span className="text-[10px] bg-[#EFF6FF] text-[#1D4ED8] font-mono px-2 py-0.5 rounded border border-[#BFDBFE]">
              Causal • Discrete • Dwell-Aware
            </span>
          </div>
          <p className="text-xs text-[#6B7280] mt-0.5">
            Synchronous End-to-End Inference: Sensor Window → Causal Encoder → VQ Tokenizer → HSMM → Multi-Task Heads
          </p>
        </div>
        <button
          onClick={onOpenEvaluation}
          className="text-xs font-mono font-semibold text-[#3563FF] bg-[#EFF6FF] hover:bg-[#DBEAFE] px-2.5 py-1 rounded border border-[#BFDBFE] transition-colors flex items-center gap-1.5"
        >
          <Activity className="w-3.5 h-3.5" />
          <span>View Empirical Benchmark Evidence (DS01)</span>
        </button>
      </div>

      {/* Five-Stage Horizontal Flowchart */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3 relative">
        {/* Animated Connector Line in Desktop */}
        <div className="hidden md:block absolute top-[72px] left-[10%] right-[10%] h-[2px] bg-[#E5E7EB] z-0">
          <div
            className={`h-full bg-[#3563FF] transition-all duration-500 ${
              isProcessing ? "opacity-100" : "opacity-40"
            }`}
            style={{ width: `${activeStageIndex * 25}%` }}
          />
        </div>

        {/* ---------------- STAGE 1: SENSOR WINDOW ---------------- */}
        <div
          onClick={() => setInspectedStage(1)}
          className={`relative z-10 bg-[#F9FAFB] border rounded p-3 cursor-pointer transition-all hover:border-[#3563FF] ${
            activeStageIndex === 0
              ? "ring-2 ring-[#3563FF] border-[#3563FF] bg-white"
              : "border-[#E5E7EB]"
          }`}
        >
          <div className="flex items-center justify-between text-[11px] font-mono text-[#6B7280] mb-1.5">
            <span>STAGE 1</span>
            <span className="w-2 h-2 rounded-full bg-[#10B981]" />
          </div>
          <div className="font-mono text-xs font-bold text-[#171717]">Sensor Window</div>
          <div className="text-[10px] text-[#6B7280] mt-0.5 mb-2 leading-tight">
            18 channels × W=4 time steps (causal sliding window)
          </div>

          {/* Mini Heatmap Visualization (18x4 grid preview) */}
          <div className="bg-white border border-[#E5E7EB] rounded p-1.5 mb-2">
            <div className="grid grid-cols-4 gap-0.5 h-10">
              {Array.from({ length: 16 }).map((_, i) => (
                <div
                  key={i}
                  className="rounded-[1px] transition-colors"
                  style={{
                    backgroundColor: `rgba(53, 99, 255, ${HEATMAP_OPACITIES[i]})`,
                  }}
                />
              ))}
            </div>
            <div className="text-[9px] font-mono text-[#9CA3AF] text-center mt-1">
              [W=4, d=18] Matrix
            </div>
          </div>

          <div className="text-[10px] font-mono text-[#4B5563] bg-white border border-[#E5E7EB] rounded px-1.5 py-1 text-center">
            Zero Future Leakage
          </div>
        </div>

        {/* ---------------- STAGE 2: CAUSAL ENCODER ---------------- */}
        <div
          onClick={() => setInspectedStage(2)}
          className={`relative z-10 bg-[#F9FAFB] border rounded p-3 cursor-pointer transition-all hover:border-[#3563FF] ${
            activeStageIndex === 1
              ? "ring-2 ring-[#3563FF] border-[#3563FF] bg-white"
              : "border-[#E5E7EB]"
          }`}
        >
          <div className="flex items-center justify-between text-[11px] font-mono text-[#6B7280] mb-1.5">
            <span>STAGE 2</span>
            <span className="w-2 h-2 rounded-full bg-[#3563FF]" />
          </div>
          <div className="font-mono text-xs font-bold text-[#171717]">Causal Temporal Encoder</div>
          <div className="text-[10px] text-[#6B7280] mt-0.5 mb-2 leading-tight">
            3 dilated causal conv blocks (d=1,2,4) → 16-dim latent vector
          </div>

          {/* Latent Vector Representation Visualizer */}
          <div className="bg-white border border-[#E5E7EB] rounded p-1.5 mb-2">
            <div className="flex items-center gap-0.5 h-10 justify-between px-1">
              {Array.from({ length: 8 }).map((_, i) => (
                <div
                  key={i}
                  className="w-2 bg-[#3563FF] rounded-xs transition-all"
                  style={{
                    height: `${LATENT_BAR_HEIGHTS[i]}px`,
                    opacity: LATENT_BAR_OPACITIES[i],
                  }}
                />
              ))}
            </div>
            <div className="text-[9px] font-mono text-[#9CA3AF] text-center mt-1">
              z_t ∈ ℝ^16 (Latent Space)
            </div>
          </div>

          <div className="text-[10px] font-mono text-[#4B5563] bg-white border border-[#E5E7EB] rounded px-1.5 py-1 text-center">
            Strict Temporal Causality
          </div>
        </div>

        {/* ---------------- STAGE 3: VQ TOKENIZER ---------------- */}
        <div
          onClick={() => setInspectedStage(3)}
          className={`relative z-10 bg-[#F9FAFB] border rounded p-3 cursor-pointer transition-all hover:border-[#3563FF] ${
            activeStageIndex === 2
              ? "ring-2 ring-[#3563FF] border-[#3563FF] bg-white"
              : "border-[#E5E7EB]"
          }`}
        >
          <div className="flex items-center justify-between text-[11px] font-mono text-[#6B7280] mb-1.5">
            <span>STAGE 3</span>
            <span className="w-2 h-2 rounded-full bg-[#8B5CF6]" />
          </div>
          <div className="font-mono text-xs font-bold text-[#171717]">VQ Event Tokenizer</div>
          <div className="text-[10px] text-[#6B7280] mt-0.5 mb-2 leading-tight">
            Codebook K=16 • STE Straight-Through Vector Quantization
          </div>

          {/* Selected Event Badge */}
          <div className="bg-white border border-[#E5E7EB] rounded p-1.5 mb-2 text-center h-16 flex flex-col justify-center">
            <div className="font-mono text-xl font-bold text-[#171717] flex items-center justify-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#3563FF]" />
              <span>{eventSymbol}</span>
            </div>
            <div className="text-[9px] text-[#6B7280] truncate font-mono mt-0.5 px-1">
              {stepData?.event_desc || "Learned Discrete Token"}
            </div>
          </div>

          <div className="text-[9px] font-mono text-[#9CA3AF] bg-white border border-[#E5E7EB] rounded px-1 py-1 text-center truncate">
            *Symbolic token, not physical label
          </div>
        </div>

        {/* ---------------- STAGE 4: EXPLICIT-DURATION HSMM ---------------- */}
        <div
          onClick={() => setInspectedStage(4)}
          className={`relative z-10 bg-[#F9FAFB] border rounded p-3 cursor-pointer transition-all hover:border-[#3563FF] ${
            activeStageIndex === 3
              ? "ring-2 ring-[#3563FF] border-[#3563FF] bg-white"
              : "border-[#E5E7EB]"
          }`}
        >
          <div className="flex items-center justify-between text-[11px] font-mono text-[#6B7280] mb-1.5">
            <span>STAGE 4</span>
            <span className="w-2 h-2 rounded-full bg-[#F59E0B]" />
          </div>
          <div className="font-mono text-xs font-bold text-[#171717]">Explicit-Duration HSMM</div>
          <div className="text-[10px] text-[#6B7280] mt-0.5 mb-2 leading-tight">
            5 temporal states • Explicit non-exponential dwell d_m(τ)
          </div>

          {/* Posterior Distribution Bars */}
          <div className="bg-white border border-[#E5E7EB] rounded p-1.5 mb-2 h-16 flex flex-col justify-center">
            <div className="flex items-end justify-between h-9 px-1 gap-1">
              {posterior.map((p: number, i: number) => (
                <div key={i} className="flex-1 flex flex-col items-center">
                  <div
                    className={`w-full rounded-t-xs transition-all ${
                      i + 1 === stateId ? "bg-[#3563FF]" : "bg-[#D1D5DB]"
                    }`}
                    style={{ height: `${Math.round(Math.max(4, p * 32))}px` }}
                  />
                  <span className="text-[8px] font-mono text-[#6B7280] mt-0.5">S{i + 1}</span>
                </div>
              ))}
            </div>
            <div className="text-[9px] font-mono text-[#171717] text-center font-bold">
              Active: State {stateId} (τ = {dwell} steps)
            </div>
          </div>

          <div className="text-[10px] font-mono text-[#4B5563] bg-white border border-[#E5E7EB] rounded px-1.5 py-1 text-center">
            Non-Markovian Dwell Hazard
          </div>
        </div>

        {/* ---------------- STAGE 5: MULTI-TASK PREDICTIVE HEADS ---------------- */}
        <div
          onClick={() => setInspectedStage(5)}
          className={`relative z-10 bg-[#F9FAFB] border rounded p-3 cursor-pointer transition-all hover:border-[#3563FF] ${
            activeStageIndex === 4
              ? "ring-2 ring-[#3563FF] border-[#3563FF] bg-white"
              : "border-[#E5E7EB]"
          }`}
        >
          <div className="flex items-center justify-between text-[11px] font-mono text-[#6B7280] mb-1.5">
            <span>STAGE 5</span>
            <span className="w-2 h-2 rounded-full bg-[#EF4444]" />
          </div>
          <div className="font-mono text-xs font-bold text-[#171717]">Predictive Task Heads</div>
          <div className="text-[10px] text-[#6B7280] mt-0.5 mb-2 leading-tight">
            Dual-path RUL (Raw + Struct) + Anomaly + Fault + Health
          </div>

          {/* RUL Output Badge */}
          <div className="bg-white border border-[#E5E7EB] rounded p-1.5 mb-2 h-16 flex flex-col justify-center text-center">
            {isUnreliable ? (
              <div className="text-[11px] font-mono font-bold text-[#DC2626] leading-tight">
                STRUCTURED ESTIMATE SUPPRESSED
              </div>
            ) : (
              <>
                <div className="font-mono text-lg font-bold text-[#171717]">
                  {rulExpected !== null && rulExpected !== undefined ? rulExpected.toFixed(1) : "—"}
                  <span className="text-xs text-[#6B7280] font-normal ml-1">cycles</span>
                </div>
                <div className="text-[9px] font-mono text-[#6B7280]">
                  90% Interval: [{rulCi[0]?.toFixed(1) ?? "—"}, {rulCi[1]?.toFixed(1) ?? "—"}]
                </div>
              </>
            )}
          </div>

          <div className="text-[9px] font-mono text-[#10B981] bg-[#ECFDF5] border border-[#A7F3D0] rounded px-1 py-1 text-center font-semibold">
            Raw + HSMM Synergy
          </div>
        </div>
      </div>

      {/* CRITICAL ARCHITECTURAL VISUALIZATION: DUAL-PATH RUL CONVERGENCE */}
      <div className="mt-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3">
        <div className="flex items-center justify-between border-b border-[#E5E7EB] pb-2 mb-2">
          <div className="flex items-center gap-1.5 text-xs font-mono font-bold text-[#171717]">
            <Layers className="w-3.5 h-3.5 text-[#3563FF]" />
            <span>Dual-Path RUL Architecture (Empirical Finding: 10.84 vs 15.47 RMSE)</span>
          </div>
          <span className="text-[10px] font-mono bg-white px-2 py-0.5 rounded border border-[#E5E7EB] text-[#4B5563]">
            Raw Sensor Bypass + HSMM Structural Dwell Synergy
          </span>
        </div>

        {/* ASCII / Schematic Flow Diagram */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-2 text-xs font-mono items-center">
          {/* Path A */}
          <div className="md:col-span-5 bg-white border border-[#E5E7EB] rounded p-2">
            <div className="text-[10px] text-[#6B7280] uppercase font-bold flex items-center justify-between">
              <span>Path A: High-Frequency Dynamics</span>
              <span className="text-[#3563FF]">18 ch × 4 steps</span>
            </div>
            <div className="text-[11px] text-[#171717] font-semibold mt-1">
              Raw Multivariate Sensor Window (72 features)
            </div>
            <div className="text-[10px] text-[#9CA3AF] mt-0.5">
              Direct bypass preserving non-quantized thermal & pressure magnitude
            </div>
          </div>

          {/* Merge operator */}
          <div className="md:col-span-2 text-center text-[#3563FF] font-bold text-sm">
            <span className="hidden md:inline">───────▶ ⊕ ◀───────</span>
            <span className="md:hidden">▼ MERGED VIA RIDGE ▼</span>
          </div>

          {/* Path B */}
          <div className="md:col-span-5 bg-white border border-[#E5E7EB] rounded p-2">
            <div className="text-[10px] text-[#6B7280] uppercase font-bold flex items-center justify-between">
              <span>Path B: Low-Frequency Grammar</span>
              <span className="text-[#8B5CF6]">HSMM Features</span>
            </div>
            <div className="text-[11px] text-[#171717] font-semibold mt-1">
              State Posterior [M=5] + Dwell Ratio [τ / D_max]
            </div>
            <div className="text-[10px] text-[#9CA3AF] mt-0.5">
              Captures cumulative phase-type degradation hazard over long horizons
            </div>
          </div>
        </div>

        <div className="mt-2 text-[11px] text-[#4B5563] font-mono flex items-center justify-between bg-white border border-[#E5E7EB] rounded p-2">
          <span>
            <strong className="text-[#171717]">Architectural Honesty Note:</strong> RUL information does not pass exclusively through the discrete token bottleneck. Combining both paths reduces RMSE by ~30% over raw alone.
          </span>
          <button
            onClick={onOpenEvaluation}
            className="text-[10px] text-[#3563FF] underline ml-2 shrink-0 font-semibold"
          >
            Inspect 39-window ablation
          </button>
        </div>
      </div>

      {/* Stage Inspection Detail Drawer (If User Clicks Any Stage) */}
      {inspectedStage && (
        <div className="mt-3 bg-[#EFF6FF] border border-[#BFDBFE] rounded p-3 text-xs font-mono text-[#1E3A8A]">
          <div className="flex items-center justify-between font-bold mb-1">
            <span>Stage {inspectedStage} Technical Specifications & Verification</span>
            <button
              onClick={() => setInspectedStage(null)}
              className="text-[#3563FF] hover:text-[#1D4ED8]"
            >
              [Close Details]
            </button>
          </div>
          {inspectedStage === 1 && (
            <p>
              Sensor window buffers 4 consecutive 1 Hz time points across 18 channels. Normalized via training-set z-statistics. No future lookahead is permitted in accordance with causal verification protocol.
            </p>
          )}
          {inspectedStage === 2 && (
            <p>
              3-layer causal dilated 1D convolution network with exponential receptive field expansion (d=1, 2, 4). Maps 72 input floats into a 16-dimensional continuous latent feature vector z_t.
            </p>
          )}
          {inspectedStage === 3 && (
            <p>
              Vector Quantizer with codebook K=16. Uses Straight-Through Estimator (STE) for backward gradient routing. Learned tokens represent recurring operational dynamics, not named physical faults.
            </p>
          )}
          {inspectedStage === 4 && (
            <p>
              Hidden Semi-Markov Model with explicit non-exponential dwell duration density d_m(τ) up to D_max=10. Forward-backward algorithm computes causal posterior and expected remaining dwell.
            </p>
          )}
          {inspectedStage === 5 && (
            <p>
              Multi-task readouts: Ridge regression on concatenated [Raw + HSMM] features for RUL; Viterbi surprisal decomposition for Anomaly; belief-weighted ordinal index for Health; GRU for Fault class.
            </p>
          )}
        </div>
      )}
    </div>
  );
};
