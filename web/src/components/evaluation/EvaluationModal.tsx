"use client";

import React, { useState } from "react";
import { X, CheckCircle, AlertCircle, BarChart3, Database, FileText } from "lucide-react";

interface EvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const EvaluationModal: React.FC<EvaluationModalProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<"readout" | "conformal" | "acoustic" | "limitations">("readout");

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4 select-none">
      <div className="bg-white border border-[#E5E7EB] rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#E5E7EB] bg-[#F9FAFB]">
          <div>
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-[#3563FF]" />
              <h2 className="text-sm font-bold font-mono text-[#171717] uppercase tracking-wide">
                Scientific Verification & Empirical Benchmarks
              </h2>
            </div>
            <p className="text-xs text-[#6B7280] mt-0.5">
              Strictly auditable results from repository JSON registries with explicit provenance.
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-[#6B7280] hover:text-[#171717] hover:bg-[#E5E7EB] transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-[#E5E7EB] bg-[#FFFFFF] px-4 gap-2 text-xs font-mono">
          <button
            onClick={() => setActiveTab("readout")}
            className={`py-2.5 px-3 border-b-2 font-semibold transition-colors ${
              activeTab === "readout"
                ? "border-[#3563FF] text-[#3563FF]"
                : "border-transparent text-[#6B7280] hover:text-[#171717]"
            }`}
          >
            Dual-Path Readout (DS01 39-Win)
          </button>
          <button
            onClick={() => setActiveTab("conformal")}
            className={`py-2.5 px-3 border-b-2 font-semibold transition-colors ${
              activeTab === "conformal"
                ? "border-[#3563FF] text-[#3563FF]"
                : "border-transparent text-[#6B7280] hover:text-[#171717]"
            }`}
          >
            LOEO Conformal Coverage (6-Fold)
          </button>
          <button
            onClick={() => setActiveTab("acoustic")}
            className={`py-2.5 px-3 border-b-2 font-semibold transition-colors ${
              activeTab === "acoustic"
                ? "border-[#3563FF] text-[#3563FF]"
                : "border-transparent text-[#6B7280] hover:text-[#171717]"
            }`}
          >
            MIMII Acoustic Discretization
          </button>
          <button
            onClick={() => setActiveTab("limitations")}
            className={`py-2.5 px-3 border-b-2 font-semibold transition-colors ${
              activeTab === "limitations"
                ? "border-[#3563FF] text-[#3563FF]"
                : "border-transparent text-[#6B7280] hover:text-[#171717]"
            }`}
          >
            Honest Limitations & Responses
          </button>
        </div>

        {/* Modal Content Body */}
        <div className="p-5 overflow-y-auto space-y-4 text-xs font-mono">
          {/* TAB 1: DUAL PATH READOUT STUDY */}
          {activeTab === "readout" && (
            <div className="space-y-4">
              <div className="bg-[#EFF6FF] border border-[#BFDBFE] rounded p-3 text-[#1E3A8A]">
                <strong>Experiment Protocol:</strong> Locked test on N-CMAPSS DS01 independent units u2 & u9 across 39 evaluated windows. Source: <code className="bg-white px-1 py-0.5 rounded border border-[#BFDBFE]">results/raw_plus_structured_results.json</code>.
              </div>

              {/* Comparison Table */}
              <div className="border border-[#E5E7EB] rounded overflow-hidden">
                <table className="w-full text-left">
                  <thead className="bg-[#F9FAFB] text-[#6B7280] border-b border-[#E5E7EB] text-[11px]">
                    <tr>
                      <th className="p-2.5">Feature Representation</th>
                      <th className="p-2.5">Readout Model</th>
                      <th className="p-2.5 text-right">RMSE (cycles)</th>
                      <th className="p-2.5 text-right">MAE (cycles)</th>
                      <th className="p-2.5 text-right">Relative Delta</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E7EB] text-[#171717]">
                    <tr>
                      <td className="p-2.5 font-bold">Raw-Only Features (72 dims)</td>
                      <td className="p-2.5">Closed-Form Ridge</td>
                      <td className="p-2.5 text-right font-bold">15.47</td>
                      <td className="p-2.5 text-right">13.37</td>
                      <td className="p-2.5 text-right text-[#6B7280]">Baseline</td>
                    </tr>
                    <tr>
                      <td className="p-2.5 font-bold">Structural Features Only (F_struct)</td>
                      <td className="p-2.5">Closed-Form Ridge</td>
                      <td className="p-2.5 text-right font-bold">17.73 ± 2.83</td>
                      <td className="p-2.5 text-right">14.52 ± 2.62</td>
                      <td className="p-2.5 text-right text-[#DC2626]">+14.6% RMSE</td>
                    </tr>
                    <tr className="bg-[#ECFDF5]">
                      <td className="p-2.5 font-bold text-[#065F46]">Raw + Structural Features (Dual-Path)</td>
                      <td className="p-2.5 text-[#065F46]">Closed-Form Ridge</td>
                      <td className="p-2.5 text-right font-bold text-[#065F46]">10.84 ± 2.24</td>
                      <td className="p-2.5 text-right text-[#065F46]">9.12 ± 2.01</td>
                      <td className="p-2.5 text-right font-bold text-[#059669]">-29.9% RMSE</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Bar Visualization */}
              <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3">
                <div className="text-[11px] font-bold text-[#171717] mb-2 uppercase">
                  RMSE Error Comparison (Lower is Better)
                </div>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-[11px] text-[#4B5563] mb-0.5">
                      <span>Structural Only (F_struct)</span>
                      <span className="font-bold">17.73 cycles</span>
                    </div>
                    <div className="w-full bg-white rounded-full h-3 border border-[#E5E7EB]">
                      <div className="bg-[#9CA3AF] h-full rounded-full" style={{ width: "88%" }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-[11px] text-[#4B5563] mb-0.5">
                      <span>Raw Only (Sensor Bypass)</span>
                      <span className="font-bold">15.47 cycles</span>
                    </div>
                    <div className="w-full bg-white rounded-full h-3 border border-[#E5E7EB]">
                      <div className="bg-[#3563FF] h-full rounded-full" style={{ width: "77%" }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-[11px] text-[#065F46] mb-0.5">
                      <span className="font-bold">Raw + Structural (EGPM Synergistic Readout)</span>
                      <span className="font-bold">10.84 cycles</span>
                    </div>
                    <div className="w-full bg-white rounded-full h-3 border border-[#E5E7EB]">
                      <div className="bg-[#10B981] h-full rounded-full" style={{ width: "54%" }} />
                    </div>
                  </div>
                </div>
              </div>

              <div className="text-[10px] text-[#6B7280]">
                *Note: These figures reflect the specific 39-window test protocol and are not claimed as universal performance. Results demonstrate that combining raw sensor telemetry with HSMM dwell hazard features yields superior predictions over either representation in isolation.
              </div>
            </div>
          )}

          {/* TAB 2: CONFORMAL COVERAGE */}
          {activeTab === "conformal" && (
            <div className="space-y-3">
              <div className="bg-[#EFF6FF] border border-[#BFDBFE] rounded p-3 text-[#1E3A8A]">
                <strong>Protocol:</strong> 6-Fold Leave-One-Engine-Out (LOEO) Conformal Prediction Calibration. Non-conformity score: α_i = |y_i - ŷ_i| / σ_i. Source: <code className="bg-white px-1 py-0.5 rounded border border-[#BFDBFE]">results/conformal_rul_results.json</code>.
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3 text-center">
                  <div className="text-[10px] text-[#6B7280] uppercase">Nominal Coverage</div>
                  <div className="text-xl font-bold text-[#171717] mt-1">90.0%</div>
                  <div className="text-[10px] text-[#9CA3AF] mt-0.5">Specified Significance (α=0.10)</div>
                </div>
                <div className="bg-[#ECFDF5] border border-[#A7F3D0] rounded p-3 text-center">
                  <div className="text-[10px] text-[#065F46] uppercase">Observed Coverage</div>
                  <div className="text-xl font-bold text-[#059669] mt-1">92.1%</div>
                  <div className="text-[10px] text-[#047857] mt-0.5">Conservative Empirical Validity</div>
                </div>
                <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded p-3 text-center">
                  <div className="text-[10px] text-[#6B7280] uppercase">Mean Interval Width</div>
                  <div className="text-xl font-bold text-[#171717] mt-1">11.8 cycles</div>
                  <div className="text-[10px] text-[#9CA3AF] mt-0.5">Operationally Actionable Window</div>
                </div>
              </div>
              <p className="text-[#4B5563] leading-relaxed">
                LOEO conformal prediction guarantees finite-sample coverage without distributional assumptions on error residuals, providing plant operators with reliable safety bounds rather than point predictions.
              </p>
            </div>
          )}

          {/* TAB 3: ACOUSTIC MIMII STUDY */}
          {activeTab === "acoustic" && (
            <div className="space-y-3">
              <div className="bg-[#FFFBEB] border border-[#FDE68A] rounded p-3 text-[#92400E]">
                <strong>Scientific Finding:</strong> Discrete representation trade-off on acoustic sound signals (MIMII Fan #06). Source: <code className="bg-white px-1 py-0.5 rounded border border-[#FDE68A]">results/mimii_fan_id_06_3seeds.json</code>.
              </div>
              <div className="border border-[#E5E7EB] rounded p-3 bg-[#F9FAFB]">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-bold text-[#171717]">Discrete VQ Codebook AUROC:</span>
                  <span className="font-bold text-[#D97706]">0.628 ± 0.014</span>
                </div>
                <p className="text-[#6B7280] text-[11px] leading-relaxed">
                  Vector-quantizing 64 log-mel acoustic spectrograms into K=16 discrete tokens discards high-frequency phase and impulse dynamics required for subtle acoustic bearing flaw detection.
                </p>
                <div className="mt-2.5 p-2 bg-white rounded border border-[#E5E7EB] text-[11px] text-[#171717]">
                  <strong>Architectural Response:</strong> Continuous residual path is retained alongside discrete grammar to maintain high-frequency sensitivity.
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: LIMITATIONS */}
          {activeTab === "limitations" && (
            <div className="space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="bg-[#ECFDF5] border border-[#A7F3D0] rounded p-3">
                  <div className="text-[#065F46] font-bold mb-1.5 flex items-center gap-1.5">
                    <CheckCircle className="w-4 h-4 text-[#059669]" />
                    <span>What Works Rigorously</span>
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[#047857] text-[11px]">
                    <li>Discrete events + explicit dwell improve RUL over HMM.</li>
                    <li>Codebook vocabulary K=16 is stable across random seeds.</li>
                    <li>DS01 conformal calibration achieves 92.1% observed coverage.</li>
                    <li>Dual-path raw+struct regression beats both single paths.</li>
                  </ul>
                </div>

                <div className="bg-[#FEF2F2] border border-[#FECACA] rounded p-3">
                  <div className="text-[#991B1B] font-bold mb-1.5 flex items-center gap-1.5">
                    <AlertCircle className="w-4 h-4 text-[#DC2626]" />
                    <span>What Does Not Yet Work Universally</span>
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-[#7F1D1D] text-[11px]">
                    <li>Discrete tokens lose acoustic nuance (MIMII 0.628 AUROC).</li>
                    <li>Health state head remains uncalibrated without ground truth.</li>
                    <li>DS02 conformal intervals limited by small unit sample size.</li>
                    <li>Physical deployment on Schneider switchgear is unvalidated.</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-3 border-t border-[#E5E7EB] bg-[#F9FAFB] flex justify-end">
          <button
            onClick={onClose}
            className="text-xs font-mono font-semibold px-4 py-1.5 bg-[#171717] text-white rounded hover:bg-[#374151] transition-colors"
          >
            Close Scientific Evidence View
          </button>
        </div>
      </div>
    </div>
  );
};
