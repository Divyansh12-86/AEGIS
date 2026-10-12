import { z } from "zod";

export interface SensorMetadata {
  index: number;
  name: string;
  label: string;
  unit: string;
  category: "operational" | "temperature" | "pressure" | "speed" | "fluid";
  nominal_range: [number, number];
  location: string;
}

export interface StepData {
  step_index: number;
  cycle: number;
  window_id: string;
  state_id: number | null;
  state_name: string;
  dwell: number | null;
  event_id: number | null;
  event_symbol: string;
  event_desc: string;
  sensor_readings: Record<string, number>;
  latent_vector_norm: number;
  codebook_reconstruction_err: number;
  posterior: number[];
  is_unreliable?: boolean;
  unreliable_reason?: string;
  rul: {
    expected: number | null;
    ci90: [number | null, number | null];
    raw_component?: number;
    struct_component?: number;
    fused_rmse?: number;
    dwell_effect_cycles?: number;
    suppressed?: boolean;
    fallback_alarm?: string;
    fallback_reason?: string;
  };
  anomaly: {
    score: number;
    threshold: number;
    is_anomalous: boolean;
    components: {
      emission: number;
      transition: number;
      duration: number;
    };
  };
  fault: {
    class: number | null;
    probabilities: number[];
    label: string;
  };
  health: {
    index: number;
    trend: number;
    status: string;
  };
  decision: {
    tier: string;
    action_text: string;
    human_in_the_loop: boolean;
    explanation: string;
  };
  engineering_insights?: EngineeringInsights;
}

export interface RootCauseItem {
  subsystem: string;
  severity: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  evidence: string;
  impact: string;
}

export interface StateCausalityInfo {
  active_state: number | null;
  state_name: string;
  elapsed_dwell: number | null;
  dwell_hazard_multiplier: number;
  explanation: string;
  counterfactual: string;
}

export interface AnomalyDecompositionItem {
  type: "Emission" | "Transition" | "Duration";
  value: number;
  share: string;
  description: string;
}

export interface ActionableMaintenancePlan {
  primary_action: string;
  time_to_act: string;
  operational_mitigation: string;
  maintenance_procedures: string[];
}

export interface EngineeringInsights {
  root_causes: RootCauseItem[];
  state_causality: StateCausalityInfo;
  anomaly_decomposition: {
    primary_driver: string;
    summary: string;
    components: AnomalyDecompositionItem[];
  };
  actionable_maintenance: ActionableMaintenancePlan;
}

export interface Scenario {
  id: string;
  title: string;
  description: string;
  steps: StepData[];
}

export interface Asset {
  id: string;
  name: string;
  dataset: string;
  description: string;
  channel_count: number;
  is_surrogate: boolean;
  surrogate_disclaimer: string;
}

export const InferenceResponseSchema = z.object({
  asset_id: z.string(),
  scenario_id: z.string().optional(),
  step_index: z.number(),
  total_steps: z.number(),
  source_mode: z.string(),
  model_version: z.string(),
  step_data: z.any(),
  provenance: z.object({
    dataset: z.string(),
    eval_protocol: z.string(),
    source_result: z.string(),
    surrogate_disclaimer: z.string(),
  }),
});

export type InferenceResponse = z.infer<typeof InferenceResponseSchema>;
