import { SCENARIOS } from "./scenarios";
import { InferenceResponse } from "./types";

let activeApiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface BackendStatus {
  isLive: boolean;
  modeLabel: string;
  version?: string;
  apiUrl?: string;
}

export async function checkBackendHealth(): Promise<BackendStatus> {
  const candidateUrls = process.env.NEXT_PUBLIC_API_URL
    ? [process.env.NEXT_PUBLIC_API_URL]
    : ["http://localhost:8000", "http://localhost:8001", "http://localhost:8002", "http://localhost:8080"];

  for (const url of candidateUrls) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 800);
      const res = await fetch(`${url}/api/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        const data = await res.json();
        activeApiBase = url;
        return {
          isLive: true,
          modeLabel: `LIVE FASTAPI BACKEND (${url.replace("http://localhost:", ":")})`,
          version: data.version || "1.0.0",
          apiUrl: url,
        };
      }
    } catch {
      // Try next port
    }
  }

  return {
    isLive: false,
    modeLabel: "RECORDED REPLAY MODE (DETERMINISTIC)",
  };
}

export async function fetchInferenceStep(
  assetId: string,
  scenarioId: string,
  stepIndex: number
): Promise<{ response: InferenceResponse; isLiveBackend: boolean }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1200);
    const res = await fetch(`${activeApiBase}/api/inference`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        asset_id: assetId,
        scenario_id: scenarioId,
        step_index: stepIndex,
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (res.ok) {
      const data = await res.json();
      return { response: data as InferenceResponse, isLiveBackend: true };
    }
  } catch {
    // Graceful fallback to deterministic local simulation
  }

  // Local fallback
  const scen = SCENARIOS[scenarioId] || SCENARIOS.scenario_a;
  const idx = Math.max(0, Math.min(stepIndex, scen.steps.length - 1));
  const step = scen.steps[idx];

  const localRes: InferenceResponse = {
    asset_id: assetId,
    scenario_id: scenarioId,
    step_index: idx,
    total_steps: scen.steps.length,
    source_mode: "recorded_replay",
    model_version: "egpm-v1-local",
    step_data: step,
    provenance: {
      dataset: "N-CMAPSS DS01",
      eval_protocol: "39 locked windows on independent units u2 & u9",
      source_result: "results/raw_plus_structured_results.json",
      surrogate_disclaimer:
        "Academic turbofan surrogate for methodology demonstration. Not validated on physical Schneider electrical switchgear.",
    },
  };

  return { response: localRes, isLiveBackend: false };
}
