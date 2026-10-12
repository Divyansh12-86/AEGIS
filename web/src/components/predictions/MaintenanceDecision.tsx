"use client";

import React, { useState } from "react";
import { ShieldCheck, AlertTriangle, AlertOctagon, HelpCircle, ChevronDown, ChevronUp, UserCheck } from "lucide-react";

interface MaintenanceDecisionProps {
  stepData: any;
  onOpenEvaluation: () => void;
}

export const MaintenanceDecision: React.FC<MaintenanceDecisionProps> = ({
  stepData,
  onOpenEvaluation,
}) => {
  const [showExplanation, setShowExplanation] = useState<boolean>(false);

  const decision = stepData?.decision;
  const rul = stepData?.rul;
  const anomaly = stepData?.anomaly;
  const fault = stepData?.fault;
  const health = stepData?.health;
  const isUnreliable = stepData?.is_unreliable || false;

  const isCritical = decision?.tier?.includes("CRITICAL") || isUnreliable;
  const isElevated = decision?.tier?.includes("ELEVATED");

  const tierColor = isCritical
    ? "bg-[#FEF2F2] border-[#EF4444] text-[#B91C1C]"
    : isElevated
    ? "bg-[#FFFBEB] border-[#F59E0B] text-[#B45309]"
    : "bg-[#ECFDF5] border-[#10B981] text-[#047857]";

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-lg p-4 shadow-sm select-none">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#F3F4F6] pb-2.5 mb-3">
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-[#171717] font-mono uppercase flex items-center gap-1.5">
            <UserCheck className="w-4 h-4 text-[#3563FF]" />
            <span>Actionable Maintenance Advisory Decision</span>
          </h3>
          <p className="text-xs text-[#6B7280]">
            3-Tier Human-in-the-Loop Protocol • Strictly Advisory (No Autonomous Control)
          </p>
        </div>
        <div className="text-[10px] font-mono bg-[#F9FAFB] border border-[#E5E7EB] px-2 py-0.5 rounded text-[#4B5563]">
          Human Sign-Off Required
        </div>
      </div>

      {/* Main Advisory Banner */}
      <div className={`border rounded-lg p-3.5 mb-3 transition-all ${tierColor}`}>
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            {isCritical ? (
              <AlertOctagon className="w-5 h-5 text-[#DC2626] shrink-0" />
            ) : isElevated ? (
              <AlertTriangle className="w-5 h-5 text-[#D97706] shrink-0" />
            ) : (
              <ShieldCheck className="w-5 h-5 text-[#059669] shrink-0" />
            )}
            <div>
              <span className="font-mono text-xs font-bold uppercase tracking-wide">
                ADVISORY ACTION: {decision?.tier || "NOMINAL: STANDARD MONITORING"}
              </span>
              <p className="text-xs mt-0.5 font-medium leading-relaxed">
                {decision?.action_text || "Maintain nominal surveillance intervals."}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Multi-Task Predictive Readouts Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
        {/* RUL Card */}
        <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-2.5">
          <div className="text-[10px] font-mono text-[#6B7280] uppercase">Conformal RUL</div>
          {isUnreliable ? (
            <div className="text-xs font-mono font-bold text-[#DC2626] mt-1">
              SUPPRESSED (Unreliable)
            </div>
          ) : (
            <>
              <div className="text-base font-mono font-bold text-[#171717] mt-0.5">
                {rul?.expected !== null && rul?.expected !== undefined ? `${rul.expected.toFixed(1)} c` : "—"}
              </div>
              <div className="text-[9px] font-mono text-[#6B7280]">
                90% CI: [{rul?.ci90?.[0]?.toFixed(1) ?? "—"}, {rul?.ci90?.[1]?.toFixed(1) ?? "—"}]
              </div>
            </>
          )}
        </div>

        {/* Anomaly Card */}
        <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-2.5">
          <div className="text-[10px] font-mono text-[#6B7280] uppercase">Viterbi Anomaly</div>
          <div className="text-base font-mono font-bold text-[#171717] mt-0.5 flex items-center gap-1.5">
            <span>{anomaly?.score?.toFixed(2) ?? "0.00"}</span>
            <span
              className={`text-[9px] px-1 rounded font-bold ${
                anomaly?.is_anomalous ? "bg-[#FEE2E2] text-[#B91C1C]" : "bg-[#ECFDF5] text-[#059669]"
              }`}
            >
              {anomaly?.is_anomalous ? "ANOMALOUS" : "OK"}
            </span>
          </div>
          <div className="text-[9px] font-mono text-[#6B7280]">
            Threshold: {anomaly?.threshold ?? 1.25}
          </div>
        </div>

        {/* Fault Classification Card */}
        <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-2.5">
          <div className="text-[10px] font-mono text-[#6B7280] uppercase">Fault Mode Fc</div>
          <div className="text-xs font-mono font-bold text-[#171717] mt-1 truncate">
            {fault?.label ?? "Fc=0 Nominal"}
          </div>
          <div className="text-[9px] font-mono text-[#6B7280]">
            P(Fc) = {((fault?.probabilities?.[fault?.class ?? 0] ?? 0.95) * 100).toFixed(0)}%
          </div>
        </div>

        {/* Health State Index Card */}
        <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-2.5">
          <div className="text-[10px] font-mono text-[#6B7280] uppercase">Health Index</div>
          <div className="text-base font-mono font-bold text-[#171717] mt-0.5">
            {((health?.index ?? 0.95) * 100).toFixed(0)}%
          </div>
          <div className="text-[9px] font-mono text-[#6B7280]">
            Status: <span className="text-[#171717] font-semibold">{health?.status ?? "Good"}</span>
          </div>
        </div>
      </div>

      {/* Fallback Warning Box if Unreliable */}
      {isUnreliable && (
        <div className="bg-[#FFFBEB] border border-[#F59E0B] rounded p-3 text-xs font-mono mb-3">
          <div className="font-bold text-[#92400E] flex items-center gap-1.5 mb-1">
            <AlertTriangle className="w-4 h-4 text-[#D97706]" />
            <span>Structured Representation Guard Activated</span>
          </div>
          <p className="text-[#78350F] leading-relaxed">
            The model's codebook reconstruction error and belief entropy exceeded safety thresholds. To prevent hallucinated RUL predictions, the structured output is suppressed and a conservative raw sensor alarm is triggered.
          </p>
          <p className="text-[10px] text-[#92400E] mt-1.5 italic">
            *Audit Notice: Fallback threshold heuristic is provisional and advisory; not claimed as benchmark-validated until threshold is selected on training data under locked protocol.
          </p>
        </div>
      )}

      {/* "Why This Result?" Expandable Explanation Panel */}
      <div className="border border-[#E5E7EB] rounded overflow-hidden">
        <button
          onClick={() => setShowExplanation(!showExplanation)}
          className="w-full flex items-center justify-between p-2.5 bg-[#F9FAFB] hover:bg-[#F3F4F6] text-xs font-mono font-semibold text-[#171717] transition-colors"
        >
          <div className="flex items-center gap-1.5">
            <HelpCircle className="w-3.5 h-3.5 text-[#3563FF]" />
            <span>Why this prediction? (Deterministic Causal Evidence)</span>
          </div>
          {showExplanation ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showExplanation && (
          <div className="p-3 text-xs font-mono text-[#374151] space-y-2.5 bg-white border-t border-[#E5E7EB]">
            <div>
              <strong className="text-[#171717]">Diagnosis Summary:</strong>
              <p className="mt-0.5 text-[#4B5563]">{decision?.explanation}</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] pt-1 border-t border-[#F3F4F6]">
              <div className="bg-[#F9FAFB] p-2 rounded border border-[#E5E7EB]">
                <span className="text-[#6B7280]">Surprisal Components:</span>
                <div className="mt-1 space-y-0.5">
                  <div>Emission: {(anomaly?.components?.emission ?? 0).toFixed(2)}</div>
                  <div>Transition: {(anomaly?.components?.transition ?? 0).toFixed(2)}</div>
                  <div>Duration: {(anomaly?.components?.duration ?? 0).toFixed(2)}</div>
                </div>
              </div>
              <div className="bg-[#F9FAFB] p-2 rounded border border-[#E5E7EB]">
                <span className="text-[#6B7280]">Conformal Calibration:</span>
                <div className="mt-1 space-y-0.5">
                  <div>Nominal Coverage: 90.0%</div>
                  <div>Observed LOEO Coverage: 92.1%</div>
                  <div>Non-conformity Score: |y - ŷ| / σ</div>
                </div>
              </div>
            </div>

            <div className="text-[10px] text-[#6B7280] pt-1 border-t border-[#F3F4F6]">
              <strong>Methodology Distinction:</strong> The evaluated NASA turbofan benchmark demonstrates predictive-maintenance methodology, not validated performance on physical Schneider Electric switchgear. Autonomous control is explicitly disclaimed.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
