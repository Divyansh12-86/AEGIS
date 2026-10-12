"use client";

import React, { useState } from "react";
import { Wrench, CheckCircle2, AlertTriangle, ArrowRight, ShieldAlert, Cpu, ListChecks, HelpCircle } from "lucide-react";
import { EngineeringInsights } from "@/lib/types";

interface EngineeringInsightsPanelProps {
  insights?: EngineeringInsights;
  cycle: number;
  isUnreliable?: boolean;
}

export const EngineeringInsightsPanel: React.FC<EngineeringInsightsPanelProps> = ({
  insights,
  cycle,
  isUnreliable,
}) => {
  const [activeTab, setActiveTab] = useState<"root_causes" | "state_causality" | "surprisal" | "maintenance">("root_causes");

  if (!insights) {
    return null;
  }

  const { root_causes, state_causality, anomaly_decomposition, actionable_maintenance } = insights;

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-lg p-4 shadow-sm select-none">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#F3F4F6] pb-3 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <Wrench className="w-4 h-4 text-[#3563FF]" />
            <h3 className="text-sm font-semibold tracking-wide text-[#171717] font-mono uppercase">
              Actionable Engineering Insights & Causal Root Cause
            </h3>
          </div>
          <p className="text-xs text-[#6B7280] mt-0.5">
            Deterministic attribution: Why anomaly & failure are predicted, and step-by-step mitigation.
          </p>
        </div>
        <div className="text-xs font-mono bg-[#EFF6FF] border border-[#BFDBFE] text-[#1D4ED8] px-2.5 py-1 rounded font-semibold">
          Cycle {cycle} Diagnosis
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-[#E5E7EB] mb-3 text-xs font-mono gap-1">
        <button
          onClick={() => setActiveTab("root_causes")}
          className={`py-1.5 px-3 border-b-2 font-semibold transition-colors ${
            activeTab === "root_causes"
              ? "border-[#3563FF] text-[#3563FF]"
              : "border-transparent text-[#6B7280] hover:text-[#171717]"
          }`}
        >
          1. Subsystem Root Causes ({root_causes.length})
        </button>
        <button
          onClick={() => setActiveTab("state_causality")}
          className={`py-1.5 px-3 border-b-2 font-semibold transition-colors ${
            activeTab === "state_causality"
              ? "border-[#3563FF] text-[#3563FF]"
              : "border-transparent text-[#6B7280] hover:text-[#171717]"
          }`}
        >
          2. State Dwell & Failure Causality
        </button>
        <button
          onClick={() => setActiveTab("surprisal")}
          className={`py-1.5 px-3 border-b-2 font-semibold transition-colors ${
            activeTab === "surprisal"
              ? "border-[#3563FF] text-[#3563FF]"
              : "border-transparent text-[#6B7280] hover:text-[#171717]"
          }`}
        >
          3. Surprisal Decomposition
        </button>
        <button
          onClick={() => setActiveTab("maintenance")}
          className={`py-1.5 px-3 border-b-2 font-semibold transition-colors ${
            activeTab === "maintenance"
              ? "border-[#3563FF] text-[#3563FF]"
              : "border-transparent text-[#6B7280] hover:text-[#171717]"
          }`}
        >
          4. Prescriptive Action Plan
        </button>
      </div>

      {/* TAB 1: Subsystem Root Causes */}
      {activeTab === "root_causes" && (
        <div className="space-y-2.5 font-mono text-xs">
          <div className="text-[11px] text-[#6B7280] mb-1">
            Specific sensor anomalies and physical components triggering the operational state change:
          </div>
          {root_causes.map((rc, idx) => {
            const isCrit = rc.severity === "CRITICAL";
            const isHigh = rc.severity === "HIGH";
            const isMod = rc.severity === "MODERATE";

            const badgeColor = isCrit
              ? "bg-[#FEE2E2] text-[#B91C1C] border-[#FECACA]"
              : isHigh
              ? "bg-[#FEF3C7] text-[#B45309] border-[#FDE68A]"
              : isMod
              ? "bg-[#EFF6FF] text-[#1D4ED8] border-[#BFDBFE]"
              : "bg-[#ECFDF5] text-[#065F46] border-[#A7F3D0]";

            return (
              <div
                key={idx}
                className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3 transition-colors hover:border-[#D1D5DB]"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-[#171717] text-xs">{rc.subsystem}</span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${badgeColor}`}>
                    {rc.severity} SEVERITY
                  </span>
                </div>
                <div className="text-[11px] text-[#4B5563] space-y-1">
                  <div>
                    <span className="text-[#6B7280]">Telemetry Evidence:</span>{" "}
                    <strong className="text-[#171717]">{rc.evidence}</strong>
                  </div>
                  <div>
                    <span className="text-[#6B7280]">Physical Degradation Mechanism:</span>{" "}
                    <span className="text-[#171717]">{rc.impact}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* TAB 2: State Dwell & Failure Causality */}
      {activeTab === "state_causality" && (
        <div className="space-y-3 font-mono text-xs">
          <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-bold text-[#171717]">
                Active Regime: State {state_causality.active_state ?? "?"} ({state_causality.state_name})
              </span>
              <span className="text-[11px] text-[#3563FF] font-bold">
                Hazard Multiplier: {state_causality.dwell_hazard_multiplier}x
              </span>
            </div>
            <p className="text-[11px] text-[#4B5563] leading-relaxed">
              {state_causality.explanation}
            </p>
          </div>

          {/* Counterfactual "What-If" Analysis */}
          <div className="bg-[#EFF6FF] border border-[#BFDBFE] rounded p-3 text-[#1E3A8A]">
            <div className="font-bold mb-1 flex items-center gap-1.5">
              <HelpCircle className="w-3.5 h-3.5 text-[#3563FF]" />
              <span>Counterfactual Temporal Attribution ("What If?"):</span>
            </div>
            <p className="text-[11px] text-[#1E40AF] leading-relaxed">
              {state_causality.counterfactual}
            </p>
            <div className="text-[10px] text-[#60A5FA] mt-1.5">
              *Proves failure is driven by cumulative non-exponential dwell time rather than transient instantaneous sensor spikes.
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Surprisal Decomposition */}
      {activeTab === "surprisal" && (
        <div className="space-y-3 font-mono text-xs">
          <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="font-bold text-[#171717]">Primary Anomaly Driver:</span>
              <span className="text-[#3563FF] font-bold">{anomaly_decomposition.primary_driver}</span>
            </div>
            <p className="text-[11px] text-[#4B5563] mt-1">
              {anomaly_decomposition.summary}
            </p>
          </div>

          <div className="space-y-2">
            {anomaly_decomposition.components.map((c, i) => (
              <div key={i} className="bg-white border border-[#E5E7EB] rounded p-2.5">
                <div className="flex justify-between items-center text-[11px] mb-1">
                  <span className="font-bold text-[#171717]">{c.type} Surprisal</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[#6B7280]">Score: {c.value.toFixed(2)}</span>
                    <span className="font-bold text-[#3563FF] bg-[#EFF6FF] px-1.5 py-0.2 rounded border border-[#BFDBFE]">
                      {c.share}
                    </span>
                  </div>
                </div>
                <p className="text-[10px] text-[#6B7280]">{c.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 4: Prescriptive Action Plan */}
      {activeTab === "maintenance" && (
        <div className="space-y-3 font-mono text-xs">
          {/* Main Action Banner */}
          <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3">
            <div className="text-[10px] text-[#6B7280] uppercase font-bold">Recommended Primary Intervention</div>
            <div className="text-sm font-bold text-[#171717] mt-0.5">
              {actionable_maintenance.primary_action}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] border-t border-[#E5E7EB] pt-2">
              <div>
                <span className="text-[#6B7280]">Operational Deadline:</span>{" "}
                <span className="font-bold text-[#B91C1C]">{actionable_maintenance.time_to_act}</span>
              </div>
            </div>
          </div>

          {/* Operational Mitigation Advice */}
          <div className="bg-[#FFFBEB] border border-[#FDE68A] rounded p-3 text-[#92400E]">
            <div className="font-bold mb-1 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-[#D97706]" />
              <span>Operational Risk Mitigation While Awaiting Maintenance:</span>
            </div>
            <p className="text-[11px] leading-relaxed">
              {actionable_maintenance.operational_mitigation}
            </p>
          </div>

          {/* Step-by-Step Procedure Checklist */}
          <div className="bg-white border border-[#E5E7EB] rounded p-3">
            <div className="font-bold text-[#171717] mb-2 flex items-center gap-1.5">
              <ListChecks className="w-3.5 h-3.5 text-[#3563FF]" />
              <span>Standard Operating Procedure (SOP) Checklist for Technician:</span>
            </div>
            <div className="space-y-1.5 text-[11px]">
              {actionable_maintenance.maintenance_procedures.map((proc, idx) => (
                <div key={idx} className="flex items-start gap-2 p-1.5 bg-[#F9FAFB] rounded border border-[#E5E7EB]">
                  <span className="text-[#3563FF] font-bold shrink-0">✓</span>
                  <span className="text-[#374151]">{proc}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
