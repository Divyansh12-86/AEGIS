"use client";

import React, { useState, useEffect, useRef } from "react";
import { TurbofanRig } from "@/components/factory/TurbofanRig";
import { SensorInspector } from "@/components/telemetry/SensorInspector";
import { EGPMPipeline } from "@/components/pipeline/EGPMPipeline";
import { TemporalExplorer } from "@/components/temporal/TemporalExplorer";
import { MaintenanceDecision } from "@/components/predictions/MaintenanceDecision";
import { EngineeringInsightsPanel } from "@/components/predictions/EngineeringInsightsPanel";
import { SimulationControls } from "@/components/simulation/SimulationControls";
import { EvaluationModal } from "@/components/evaluation/EvaluationModal";
import { NCMAPSS_SENSORS, SCENARIOS } from "@/lib/scenarios";
import { SensorMetadata, StepData } from "@/lib/types";
import { checkBackendHealth, fetchInferenceStep } from "@/lib/api";
import { Cpu, Activity, Info } from "lucide-react";

export default function EGPMIndustrialSimulation() {
  const [selectedSensor, setSelectedSensor] = useState<SensorMetadata>(NCMAPSS_SENSORS[6]); // Default: T50
  const [currentScenarioId, setCurrentScenarioId] = useState<string>("scenario_a");
  const [stepIndex, setStepIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1.0);
  const [isLiveBackend, setIsLiveBackend] = useState<boolean>(false);
  const [stepData, setStepData] = useState<StepData>(SCENARIOS.scenario_a.steps[0]);
  const [history, setHistory] = useState<StepData[]>([SCENARIOS.scenario_a.steps[0]]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeStageIndex, setActiveStageIndex] = useState<number>(0);
  const [showEvalModal, setShowEvalModal] = useState<boolean>(false);

  // Check backend availability on mount
  useEffect(() => {
    checkBackendHealth().then((status) => {
      setIsLiveBackend(status.isLive);
    });
  }, []);

  // Fetch step data when stepIndex or scenario changes
  useEffect(() => {
    let isCancelled = false;
    let stageInterval: NodeJS.Timeout | null = null;
    setIsProcessing(true);
    setActiveStageIndex(0);

    fetchInferenceStep("DS01_u2", currentScenarioId, stepIndex).then((res) => {
      if (!isCancelled) {
        setStepData(res.response.step_data);
        setIsLiveBackend(res.isLiveBackend);
        setHistory((prev) => {
          const updated = [...prev, res.response.step_data];
          return updated.slice(-10); // keep last 10 steps
        });

        // Stage animation: deliberate, observable progression through all 5 neuro-symbolic stages
        let stage = 0;
        const stageDuration = Math.max(250, Math.round(450 / speed));
        stageInterval = setInterval(() => {
          stage += 1;
          if (stage <= 4) {
            setActiveStageIndex(stage);
          } else {
            if (stageInterval) clearInterval(stageInterval);
            setIsProcessing(false);
          }
        }, stageDuration);
      }
    });

    return () => {
      isCancelled = true;
      if (stageInterval) clearInterval(stageInterval);
    };
  }, [stepIndex, currentScenarioId, speed]);

  // Simulation playback timer
  useEffect(() => {
    if (!isPlaying) return;

    const currentScenario = SCENARIOS[currentScenarioId] || SCENARIOS.scenario_a;
    // Allow the 5 stages to complete (5 * 450ms = 2250ms), dwell on the prediction, then advance
    const intervalTime = Math.max(2600, Math.round(3400 / speed));

    const timer = setInterval(() => {
      setStepIndex((prev) => {
        if (prev < currentScenario.steps.length - 1) {
          return prev + 1;
        } else {
          setIsPlaying(false);
          return prev;
        }
      });
    }, intervalTime);

    return () => clearInterval(timer);
  }, [isPlaying, speed, currentScenarioId]);

  const handleStepForward = () => {
    const currentScenario = SCENARIOS[currentScenarioId] || SCENARIOS.scenario_a;
    if (stepIndex < currentScenario.steps.length - 1) {
      setStepIndex(stepIndex + 1);
    }
  };

  const handleReset = () => {
    setIsPlaying(false);
    setStepIndex(0);
    const scen = SCENARIOS[currentScenarioId] || SCENARIOS.scenario_a;
    setStepData(scen.steps[0]);
    setHistory([scen.steps[0]]);
  };

  const handleSelectScenario = (scenId: string) => {
    setIsPlaying(false);
    setCurrentScenarioId(scenId);
    setStepIndex(0);
    const scen = SCENARIOS[scenId] || SCENARIOS.scenario_a;
    setStepData(scen.steps[0]);
    setHistory([scen.steps[0]]);
  };

  const currentScenario = SCENARIOS[currentScenarioId] || SCENARIOS.scenario_a;

  return (
    <div className="min-h-screen bg-[#F3F4F6] text-[#171717] font-sans antialiased p-3 sm:p-5">
      {/* Top Application Bar */}
      <header className="bg-white border border-[#E5E7EB] rounded-lg px-4 py-3 mb-3 shadow-xs flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded bg-[#171717] flex items-center justify-center text-white font-mono font-bold text-sm shadow-xs">
            EG
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-[#171717] font-mono tracking-tight">
                EGPM Industrial Predictive Simulation
              </h1>
              <span className="text-[10px] font-mono uppercase bg-[#171717] text-white px-2 py-0.5 rounded font-semibold">
                Schneider PredictX 2026
              </span>
            </div>
            <p className="text-xs text-[#6B7280]">
              Interpretable Neuro-Symbolic Event Grammar Probabilistic Model (EGPM)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <button
            onClick={() => setShowEvalModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#EFF6FF] border border-[#BFDBFE] text-[#1D4ED8] hover:bg-[#DBEAFE] font-semibold transition-colors"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Empirical Evidence (DS01 Locked Test)</span>
          </button>
        </div>
      </header>

      {/* Simulation Playback & Scenario Controls */}
      <div className="mb-3">
        <SimulationControls
          isPlaying={isPlaying}
          onTogglePlay={() => setIsPlaying(!isPlaying)}
          onStepForward={handleStepForward}
          onReset={handleReset}
          speed={speed}
          onChangeSpeed={setSpeed}
          currentScenarioId={currentScenarioId}
          onSelectScenario={handleSelectScenario}
          isLiveBackend={isLiveBackend}
          currentStepIndex={stepIndex}
          totalSteps={currentScenario.steps.length}
        />
      </div>

      {/* Main Layout Grid */}
      <main className="space-y-3">
        {/* Row 1: The Machine Test Stand (Left) & Sensor Inspector (Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
          <div className="lg:col-span-7">
            <TurbofanRig
              selectedSensor={selectedSensor}
              onSelectSensor={setSelectedSensor}
              allSensors={NCMAPSS_SENSORS}
              isProcessing={isProcessing}
              stepData={stepData}
            />
          </div>
          <div className="lg:col-span-5">
            <SensorInspector
              selectedSensor={selectedSensor}
              onSelectSensor={setSelectedSensor}
              allSensors={NCMAPSS_SENSORS}
              stepData={stepData}
              history={history}
            />
          </div>
        </div>

        {/* Row 2: Central Visual Centerpiece — The 5-Stage EGPM Pipeline */}
        <div>
          <EGPMPipeline
            stepData={stepData}
            isProcessing={isProcessing}
            activeStageIndex={activeStageIndex}
            onOpenEvaluation={() => setShowEvalModal(true)}
          />
        </div>

        {/* Row 3: Temporal Event Explorer (Left) & Actionable Maintenance Advisory (Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
          <div className="lg:col-span-6">
            <TemporalExplorer
              history={history}
              stepData={stepData}
              onSelectStepIndex={setStepIndex}
            />
          </div>
          <div className="lg:col-span-6">
            <MaintenanceDecision
              stepData={stepData}
              onOpenEvaluation={() => setShowEvalModal(true)}
            />
          </div>
        </div>

        {/* Row 4: Actionable Engineering Insights & Root-Cause Attribution */}
        <div>
          <EngineeringInsightsPanel
            insights={stepData?.engineering_insights}
            cycle={stepData?.cycle ?? 10}
            isUnreliable={stepData?.is_unreliable}
          />
        </div>
      </main>

      {/* Scientific Evaluation Modal */}
      <EvaluationModal
        isOpen={showEvalModal}
        onClose={() => setShowEvalModal(false)}
      />

      {/* Engineering Footer */}
      <footer className="mt-4 text-center text-xs font-mono text-[#6B7280] py-2 border-t border-[#E5E7EB]">
        <span>AEGIS-1 (EGPM) • Schneider Electric PredictX 2026 Hackathon • Methodological Surrogate Benchmark (NASA N-CMAPSS) • Human-in-the-Loop Advisory</span>
      </footer>
    </div>
  );
}
