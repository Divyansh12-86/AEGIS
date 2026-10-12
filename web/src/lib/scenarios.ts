import { SensorMetadata, Scenario } from "./types";

export const NCMAPSS_SENSORS: SensorMetadata[] = [
  { index: 0, name: "alt", label: "Altitude", unit: "ft", category: "operational", nominal_range: [0, 40000], location: "Inlet Environment" },
  { index: 1, name: "Mach", label: "Mach Number", unit: "Mach", category: "operational", nominal_range: [0.0, 0.85], location: "Flight Dynamics" },
  { index: 2, name: "TRA", label: "Throttle Resolver Angle", unit: "%", category: "operational", nominal_range: [20, 100], location: "FADEC Control" },
  { index: 3, name: "T2", label: "Fan Inlet Temperature", unit: "°R", category: "operational", nominal_range: [480, 560], location: "Fan Face" },
  { index: 4, name: "T24", label: "LPC Outlet Temperature", unit: "°R", category: "temperature", nominal_range: [580, 680], location: "Low-Pressure Compressor" },
  { index: 5, name: "T30", label: "HPC Outlet Temperature", unit: "°R", category: "temperature", nominal_range: [1400, 1680], location: "High-Pressure Compressor" },
  { index: 6, name: "T50", label: "LPT Exhaust Temperature", unit: "°R", category: "temperature", nominal_range: [1200, 1420], location: "Turbine Exhaust" },
  { index: 7, name: "P2", label: "Fan Inlet Total Pressure", unit: "psia", category: "pressure", nominal_range: [8.0, 15.5], location: "Fan Cowling" },
  { index: 8, name: "P15", label: "Bypass Duct Pressure", unit: "psia", category: "pressure", nominal_range: [12.0, 24.0], location: "Bypass Nozzle" },
  { index: 9, name: "P30", label: "HPC Outlet Total Pressure", unit: "psia", category: "pressure", nominal_range: [350, 580], location: "Diffuser / Combustor" },
  { index: 10, name: "Nf", label: "Physical Fan Speed", unit: "rpm", category: "speed", nominal_range: [1800, 2500], location: "Low-Pressure Spool" },
  { index: 11, name: "Nc", label: "Physical Core Speed", unit: "rpm", category: "speed", nominal_range: [8200, 9600], location: "High-Pressure Spool" },
  { index: 12, name: "epr", label: "Engine Pressure Ratio", unit: "ratio", category: "pressure", nominal_range: [1.0, 2.2], location: "Core Exhaust / Inlet" },
  { index: 13, name: "Ps30", label: "HPC Outlet Static Pressure", unit: "psia", category: "pressure", nominal_range: [340, 560], location: "Combustor Liner" },
  { index: 14, name: "Wf", label: "Fuel Flow", unit: "pps", category: "fluid", nominal_range: [2.5, 6.5], location: "Fuel Spray Nozzles" },
  { index: 15, name: "Nf_dmd", label: "Demanded Fan Speed", unit: "rpm", category: "speed", nominal_range: [1800, 2500], location: "FADEC Demanded" },
  { index: 16, name: "Nc_dmd", label: "Demanded Core Speed", unit: "rpm", category: "speed", nominal_range: [8200, 9600], location: "FADEC Demanded" },
  { index: 17, name: "BPR", label: "Bypass Ratio", unit: "ratio", category: "fluid", nominal_range: [6.0, 10.5], location: "Bypass Bleed Valve" },
];

export const SCENARIOS: Record<string, Scenario> = {
  scenario_a: {
    id: "scenario_a",
    title: "Scenario A: Normal Steady-State Operation",
    description: "Stable sensor trends on DS01_u2. Steady event tokens (E03), State 1 dominant, linear aging, nominal monitoring.",
    steps: [
      {
        step_index: 0,
        cycle: 10,
        window_id: "W-010",
        state_id: 1,
        state_name: "Healthy / Nominal Operation",
        dwell: 10,
        event_id: 3,
        event_symbol: "E03",
        event_desc: "Steady-State Baseline Operation",
        sensor_readings: {
          alt: 25000.0, Mach: 0.65, TRA: 65.0, T2: 512.4,
          T24: 618.5, T30: 1490.2, T50: 1285.4, P2: 11.8,
          P15: 16.4, P30: 445.0, Nf: 2150.0, Nc: 8850.0,
          epr: 1.45, Ps30: 432.0, Wf: 3.65, Nf_dmd: 2150.0,
          Nc_dmd: 8850.0, BPR: 8.42
        },
        latent_vector_norm: 0.85,
        codebook_reconstruction_err: 0.042,
        posterior: [0.94, 0.05, 0.01, 0.00, 0.00],
        rul: {
          expected: 88.5,
          ci90: [82.1, 94.2],
          raw_component: 15.47,
          struct_component: 17.73,
          fused_rmse: 10.84,
          dwell_effect_cycles: -2.1
        },
        anomaly: { score: 0.14, threshold: 1.25, is_anomalous: false, components: { emission: 0.08, transition: 0.04, duration: 0.02 } },
        fault: { class: 0, probabilities: [0.96, 0.03, 0.01], label: "Nominal (No Fault)" },
        health: { index: 0.95, trend: 0.00, status: "Good" },
        decision: {
          tier: "NOMINAL: STANDARD MONITORING",
          action_text: "Asset operating within nominal design envelope. Maintain standard monthly inspection cadence.",
          human_in_the_loop: true,
          explanation: "State 1 dominant with low dwell hazard. No anomalous transitions detected."
        },
        engineering_insights: {
          root_causes: [
            {
              subsystem: "Core Thermal Baseline (T24, T30, T50)",
              severity: "LOW",
              evidence: "All temperatures within ±1.5% of calibrated nominal cruise baseline (T50 = 1285.4 °R).",
              impact: "No observable thermal distress or blade degradation."
            },
            {
              subsystem: "Spool Speed Synchronization (Nf, Nc)",
              severity: "LOW",
              evidence: "Actual speeds (Nf=2150, Nc=8850) match demanded speeds (Nf_dmd, Nc_dmd) exactly.",
              impact: "FADEC control loop is fully balanced without hunting."
            }
          ],
          state_causality: {
            active_state: 1,
            state_name: "Healthy / Nominal Operation",
            elapsed_dwell: 10,
            dwell_hazard_multiplier: 1.0,
            explanation: "State 1 indicates early-life nominal health. Hazard rate is constant and minimal.",
            counterfactual: "If asset had transitioned to State 2 prematurely, expected RUL would be 68.2 cycles (-20.3 cycles drop)."
          },
          anomaly_decomposition: {
            primary_driver: "Emission Surprisal (57%)",
            summary: "Total surprisal score (0.14) is well below anomaly threshold (1.25). Observations reflect nominal sensor noise.",
            components: [
              { type: "Emission", value: 0.08, share: "57%", description: "Sensor observations fit codebook centroid E03" },
              { type: "Transition", value: 0.04, share: "29%", description: "Expected high persistence in State 1" },
              { type: "Duration", value: 0.02, share: "14%", description: "Elapsed dwell τ=10 is well within expected dwell mean (25 steps)" }
            ]
          },
          actionable_maintenance: {
            primary_action: "Continue Standard Monthly Surveillance Cadence",
            time_to_act: "Next scheduled cycle (30 days)",
            operational_mitigation: "None required. Asset certified for unrestricted operation across full flight envelope.",
            maintenance_procedures: [
              "1. Log standard fuel burn and operating hour counters in Schneider asset log",
              "2. Maintain automated 1 Hz telemetry streaming",
              "3. Review rolling 30-day baseline drift at next scheduled sync"
            ]
          }
        }
      },
      {
        step_index: 1,
        cycle: 18,
        window_id: "W-018",
        state_id: 1,
        state_name: "Healthy / Nominal Operation",
        dwell: 18,
        event_id: 3,
        event_symbol: "E03",
        event_desc: "Steady-State Baseline Operation",
        sensor_readings: {
          alt: 25050.0, Mach: 0.65, TRA: 65.1, T2: 512.8,
          T24: 619.2, T30: 1492.5, T50: 1286.8, P2: 11.8,
          P15: 16.3, P30: 444.6, Nf: 2151.0, Nc: 8852.0,
          epr: 1.45, Ps30: 431.8, Wf: 3.66, Nf_dmd: 2150.0,
          Nc_dmd: 8850.0, BPR: 8.41
        },
        latent_vector_norm: 0.86,
        codebook_reconstruction_err: 0.044,
        posterior: [0.92, 0.06, 0.02, 0.00, 0.00],
        rul: {
          expected: 80.5,
          ci90: [74.3, 86.2],
          raw_component: 15.47,
          struct_component: 17.73,
          fused_rmse: 10.84,
          dwell_effect_cycles: -8.0
        },
        anomaly: { score: 0.15, threshold: 1.25, is_anomalous: false, components: { emission: 0.08, transition: 0.04, duration: 0.03 } },
        fault: { class: 0, probabilities: [0.95, 0.03, 0.02], label: "Nominal (No Fault)" },
        health: { index: 0.93, trend: -0.02, status: "Good" },
        decision: {
          tier: "NOMINAL: STANDARD MONITORING",
          action_text: "Sustained nominal dwell. Linear progression, no intervention needed.",
          human_in_the_loop: true,
          explanation: "Continuous dwell accumulation in State 1 consistent with expected flight cycle."
        },
        engineering_insights: {
          root_causes: [
            {
              subsystem: "Core Thermal Baseline (T24, T30, T50)",
              severity: "LOW",
              evidence: "T50 = 1286.8 °R (+1.4 °R drift from cycle 10), reflecting normal cumulative operating hours.",
              impact: "Healthy baseline wear rate."
            }
          ],
          state_causality: {
            active_state: 1,
            state_name: "Healthy / Nominal Operation",
            elapsed_dwell: 18,
            dwell_hazard_multiplier: 1.0,
            explanation: "Sustained dwell τ=18 in State 1 accounts for exact -8.0 cycles reduction in remaining useful life.",
            counterfactual: "Linear dwell aging: -1 cycle of RUL per elapsed flight cycle."
          },
          anomaly_decomposition: {
            primary_driver: "Emission Surprisal (53%)",
            summary: "Total surprisal score (0.15) remains flat. No early warning alarms.",
            components: [
              { type: "Emission", value: 0.08, share: "53%", description: "Sensor observations fit codebook centroid E03" },
              { type: "Transition", value: 0.04, share: "27%", description: "Persistence in State 1 is normal" },
              { type: "Duration", value: 0.03, share: "20%", description: "Dwell accumulation τ=18 within bounds" }
            ]
          },
          actionable_maintenance: {
            primary_action: "Maintain Standard Operational Cadence",
            time_to_act: "Next scheduled cycle (30 days)",
            operational_mitigation: "None required.",
            maintenance_procedures: [
              "1. Continue standard monitoring interval"
            ]
          }
        }
      }
    ]
  },
  scenario_b: {
    id: "scenario_b",
    title: "Scenario B: Changing Operating Conditions",
    description: "Altitude, Mach, and throttle ramp (TRA 65% → 78%). Sensor traces drift naturally, producing new tokens (E05) without mechanical fault.",
    steps: [
      {
        step_index: 0,
        cycle: 25,
        window_id: "W-025",
        state_id: 2,
        state_name: "Bedding-In / Thermal Drift",
        dwell: 5,
        event_id: 5,
        event_symbol: "E05",
        event_desc: "Minor Contact Resistance Drift",
        sensor_readings: {
          alt: 28000.0, Mach: 0.72, TRA: 78.0, T2: 498.2,
          T24: 635.4, T30: 1540.0, T50: 1315.0, P2: 10.4,
          P15: 17.8, P30: 482.0, Nf: 2240.0, Nc: 9120.0,
          epr: 1.58, Ps30: 468.0, Wf: 4.12, Nf_dmd: 2240.0,
          Nc_dmd: 9120.0, BPR: 7.95
        },
        latent_vector_norm: 0.92,
        codebook_reconstruction_err: 0.051,
        posterior: [0.12, 0.81, 0.06, 0.01, 0.00],
        rul: {
          expected: 68.2,
          ci90: [62.4, 74.0],
          raw_component: 15.47,
          struct_component: 17.73,
          fused_rmse: 10.84,
          dwell_effect_cycles: -20.3
        },
        anomaly: { score: 0.38, threshold: 1.25, is_anomalous: false, components: { emission: 0.22, transition: 0.11, duration: 0.05 } },
        fault: { class: 0, probabilities: [0.88, 0.09, 0.03], label: "Nominal Flight Cruise" },
        health: { index: 0.84, trend: -0.08, status: "Acceptable" },
        decision: {
          tier: "NOMINAL: CONTINUE MONITORING",
          action_text: "Expected thermal stabilization following load ramp. No immediate intervention required.",
          human_in_the_loop: true,
          explanation: "Smooth transition from State 1 -> 2. Dwell accumulation is consistent with normal aging under higher flight regime."
        },
        engineering_insights: {
          root_causes: [
            {
              subsystem: "Throttle & Flight Dynamic Ramp (TRA, Mach, Alt)",
              severity: "MODERATE",
              evidence: "TRA increased to 78%, Mach to 0.72. Resulting T50 rise (+29.6 °R) matches aerothermal scaling laws.",
              impact: "Benign aerodynamic load transition, explicitly separated by causal encoder from mechanical fault."
            },
            {
              subsystem: "Diffuser & Combustor Pressure (P30)",
              severity: "LOW",
              evidence: "P30 = 482.0 psia (+37 psia) tracks higher compressor pressure ratio proportionally.",
              impact: "Normal compressor work increase."
            }
          ],
          state_causality: {
            active_state: 2,
            state_name: "Bedding-In / Thermal Drift",
            elapsed_dwell: 5,
            dwell_hazard_multiplier: 1.2,
            explanation: "State 2 represents high-power cruise bedding-in. Causal temporal encoder normalizes out altitude effects.",
            counterfactual: "If load returned to idle/low cruise (TRA 65%), state posterior would revert toward State 1 within 3 windows."
          },
          anomaly_decomposition: {
            primary_driver: "Emission Surprisal (58%)",
            summary: "Surprisal score rose to 0.38 due to higher fuel flow and temperature, but remains far below 1.25 alarm threshold.",
            components: [
              { type: "Emission", value: 0.22, share: "58%", description: "Thermal rise produces Token E05 instead of baseline E03" },
              { type: "Transition", value: 0.11, share: "29%", description: "Permissible State 1 → State 2 transition boundary" },
              { type: "Duration", value: 0.05, share: "13%", description: "Dwell τ=5 in State 2 is early in expected duration" }
            ]
          },
          actionable_maintenance: {
            primary_action: "Continue Monitoring at Higher Operating Load",
            time_to_act: "Standard operational cycle",
            operational_mitigation: "Ensure coolant bleed valve (BPR=7.95) modulates smoothly with throttle demand.",
            maintenance_procedures: [
              "1. Verify pressure ratio (epr = 1.58) corresponds with demanded throttle",
              "2. Monitor for temperature stabilization over next 10 flight cycles"
            ]
          }
        }
      },
      {
        step_index: 1,
        cycle: 36,
        window_id: "W-036",
        state_id: 2,
        state_name: "Bedding-In / Thermal Drift",
        dwell: 16,
        event_id: 7,
        event_symbol: "E07",
        event_desc: "Thermal Rise + Contact Resistance Transient",
        sensor_readings: {
          alt: 28100.0, Mach: 0.72, TRA: 78.8, T2: 498.8,
          T24: 644.0, T30: 1555.0, T50: 1326.0, P2: 10.3,
          P15: 18.1, P30: 488.0, Nf: 2252.0, Nc: 9140.0,
          epr: 1.60, Ps30: 474.0, Wf: 4.22, Nf_dmd: 2252.0,
          Nc_dmd: 9140.0, BPR: 7.88
        },
        latent_vector_norm: 0.96,
        codebook_reconstruction_err: 0.059,
        posterior: [0.04, 0.74, 0.17, 0.04, 0.01],
        rul: {
          expected: 53.8,
          ci90: [48.1, 59.5],
          raw_component: 15.47,
          struct_component: 17.73,
          fused_rmse: 10.84,
          dwell_effect_cycles: -14.4
        },
        anomaly: { score: 0.46, threshold: 1.25, is_anomalous: false, components: { emission: 0.26, transition: 0.12, duration: 0.08 } },
        fault: { class: 0, probabilities: [0.81, 0.14, 0.05], label: "Nominal High Cruise" },
        health: { index: 0.76, trend: -0.08, status: "Acceptable" },
        decision: {
          tier: "NOMINAL: CONTINUE MONITORING",
          action_text: "Thermal and pressure stabilization verified. Continue standard surveillance cadence.",
          human_in_the_loop: true,
          explanation: "High throttle operating condition accounted for by causal encoder. No wear signature detected."
        },
        engineering_insights: {
          root_causes: [
            {
              subsystem: "High Cruise Thermal Equilibrium (T30, T50)",
              severity: "MODERATE",
              evidence: "T50 = 1326.0 °R has leveled off after load ramp. Rate of change dT/dt < 0.2 °R/min.",
              impact: "Stable high-cruise heat dissipation without runaway."
            }
          ],
          state_causality: {
            active_state: 2,
            state_name: "Bedding-In / Thermal Drift",
            elapsed_dwell: 16,
            dwell_hazard_multiplier: 1.3,
            explanation: "Sustained dwell in State 2 (τ=16) reflects normal aging under elevated flight conditions.",
            counterfactual: "If State 3 transition occurs before cycle 45, hazard multiplier will double from 1.3x to 2.6x."
          },
          anomaly_decomposition: {
            primary_driver: "Emission Surprisal (57%)",
            summary: "Surprisal stable at 0.46. Model distinguishes thermal persistence from failure onset.",
            components: [
              { type: "Emission", value: 0.26, share: "57%", description: "High cruise telemetry cluster" },
              { type: "Transition", value: 0.12, share: "26%", description: "Continued residency in State 2" },
              { type: "Duration", value: 0.08, share: "17%", description: "Dwell τ=16 approaches nominal mean (18 steps)" }
            ]
          },
          actionable_maintenance: {
            primary_action: "Maintain Normal Operational Monitoring",
            time_to_act: "Standard operational cycle",
            operational_mitigation: "None required.",
            maintenance_procedures: [
              "1. Log steady-state thermal index in fleet management database"
            ]
          }
        }
      }
    ]
  },
  scenario_c: {
    id: "scenario_c",
    title: "Scenario C: Accelerated Wear & Critical Failure Precursor",
    description: "Incipient wear in State 3 accelerates to severe State 4 thermal degradation. Exhaust gas temp (T50) spikes, RUL drops to 14.8 cycles with critical maintenance advisory.",
    steps: [
      {
        step_index: 0,
        cycle: 42,
        window_id: "W-042",
        state_id: 3,
        state_name: "Incipient Micro-Fretting & Wear",
        dwell: 8,
        event_id: 7,
        event_symbol: "E07",
        event_desc: "Thermal Rise + Contact Resistance Transient",
        sensor_readings: {
          alt: 32000.0, Mach: 0.78, TRA: 84.0, T2: 482.0,
          T24: 662.0, T30: 1610.0, T50: 1365.0, P2: 8.9,
          P15: 19.4, P30: 512.0, Nf: 2310.0, Nc: 9340.0,
          epr: 1.72, Ps30: 498.0, Wf: 4.85, Nf_dmd: 2310.0,
          Nc_dmd: 9340.0, BPR: 7.40
        },
        latent_vector_norm: 1.18,
        codebook_reconstruction_err: 0.084,
        posterior: [0.01, 0.04, 0.82, 0.12, 0.01],
        rul: {
          expected: 36.2,
          ci90: [31.8, 41.5],
          raw_component: 15.47,
          struct_component: 17.73,
          fused_rmse: 10.84,
          dwell_effect_cycles: -32.0
        },
        anomaly: { score: 0.82, threshold: 1.25, is_anomalous: false, components: { emission: 0.45, transition: 0.22, duration: 0.15 } },
        fault: { class: 1, probabilities: [0.24, 0.68, 0.08], label: "Fc=1 (Incipient Fretting)" },
        health: { index: 0.58, trend: -0.20, status: "Degrading" },
        decision: {
          tier: "ELEVATED: INCREASE TELEMETRY FREQUENCY",
          action_text: "Elevate telemetry polling from daily to hourly. Flag unit for review during upcoming weekly maintenance sync.",
          human_in_the_loop: true,
          explanation: "Event E07 triggered by elevated exhaust temp T50 (+50°R above expected). Dwell in State 3 accelerates phase-type hazard."
        },
        engineering_insights: {
          root_causes: [
            {
              subsystem: "Turbine Exhaust Temperature (T50)",
              severity: "HIGH",
              evidence: "T50 = 1365.0 °R exceeds expected cruise thermal model by +45.0 °R despite steady fuel flow.",
              impact: "Incipient thermal distress in low-pressure turbine or contact micro-fretting resistance rise."
            },
            {
              subsystem: "Compressor Discharge Static Pressure (Ps30)",
              severity: "MODERATE",
              evidence: "Ps30 static pressure ratio dropping (498.0 psia vs 512.0 total), indicating micro-leakage in compressor stage.",
              impact: "Mild aerodynamic stage loss accelerating heat buildup."
            }
          ],
          state_causality: {
            active_state: 3,
            state_name: "Incipient Micro-Fretting & Wear",
            elapsed_dwell: 8,
            dwell_hazard_multiplier: 2.1,
            explanation: "Transition into State 3 (posterior 0.82) combined with dwell τ=8 steps increases phase-type absorption hazard 2.1-fold.",
            counterfactual: "If wear progression had remained halted in State 2, RUL would be 68.2 cycles (+32.0 cycles higher)."
          },
          anomaly_decomposition: {
            primary_driver: "Emission Surprisal (55%)",
            summary: "Surprisal score (0.82) is approaching 1.25 threshold. High T50 emission probability is low under nominal health.",
            components: [
              { type: "Emission", value: 0.45, share: "55%", description: "Elevated T50 and Ps30 drift away from baseline codebook centroid" },
              { type: "Transition", value: 0.22, share: "27%", description: "Uncommon State 2 → State 3 transition in mid-life" },
              { type: "Duration", value: 0.15, share: "18%", description: "Dwell in State 3 is accumulating hazard" }
            ]
          },
          actionable_maintenance: {
            primary_action: "Elevate Telemetry Polling from Daily to Hourly & Schedule Layover Inspection",
            time_to_act: "Within 31 cycles (conformal lower bound)",
            operational_mitigation: "Avoid extended maximum-thrust climbs; limit TRA to ≤ 80% to constrain T50 exhaust rise.",
            maintenance_procedures: [
              "1. Increase automated sensor logging frequency to 10 Hz",
              "2. Review oil debris monitor (ODM) and vibration spectral peaks for bearing race flaking",
              "3. Schedule non-destructive borescope examination of LPT stator blades at next scheduled overnight layover"
            ]
          }
        }
      },
      {
        step_index: 1,
        cycle: 58,
        window_id: "W-058",
        state_id: 4,
        state_name: "Accelerated Thermal Degradation",
        dwell: 12,
        event_id: 14,
        event_symbol: "E14",
        event_desc: "High Harmonic Heat Dissipation & Arc Precursor",
        sensor_readings: {
          alt: 32050.0, Mach: 0.78, TRA: 84.5, T2: 481.5,
          T24: 688.0, T30: 1675.0, T50: 1418.0, P2: 8.8,
          P15: 19.8, P30: 535.0, Nf: 2345.0, Nc: 9480.0,
          epr: 1.81, Ps30: 521.0, Wf: 5.42, Nf_dmd: 2345.0,
          Nc_dmd: 9480.0, BPR: 6.95
        },
        latent_vector_norm: 1.54,
        codebook_reconstruction_err: 0.142,
        posterior: [0.00, 0.01, 0.09, 0.86, 0.04],
        rul: {
          expected: 14.8,
          ci90: [11.2, 18.9],
          raw_component: 15.47,
          struct_component: 17.73,
          fused_rmse: 10.84,
          dwell_effect_cycles: -21.4
        },
        anomaly: { score: 1.84, threshold: 1.25, is_anomalous: true, components: { emission: 0.98, transition: 0.44, duration: 0.42 } },
        fault: { class: 2, probabilities: [0.08, 0.22, 0.70], label: "Fc=2 (Critical Arc / Thermal Precursor)" },
        health: { index: 0.22, trend: -0.36, status: "Critical" },
        decision: {
          tier: "CRITICAL: SCHEDULE MAINTENANCE INSPECTION",
          action_text: "Schedule contact overhaul within 11 cycles (conformal lower bound). Advisory: operator must approve work order.",
          human_in_the_loop: true,
          explanation: "High dwell in State 4 (12 cycles) with elevated exhaust heat and vibration. Transition to absorbing failure imminent."
        },
        engineering_insights: {
          root_causes: [
            {
              subsystem: "Turbine Exhaust Gas Temperature (T50)",
              severity: "CRITICAL",
              evidence: "T50 = 1418.0 °R (+58.0 °R above redline thermal limit of 1360 °R).",
              impact: "Severe thermal oxidation, stator vane trailing edge burn-through, and electrical contact surface melting."
            },
            {
              subsystem: "Fuel Delivery & Core Speed Efficiency (Wf / Nc)",
              severity: "HIGH",
              evidence: "Wf = 5.42 pps (+18% fuel burn over normal) needed to maintain Nc=9480 rpm.",
              impact: "Compressor efficiency collapse forcing engine to over-fuel, creating localized hot spots."
            },
            {
              subsystem: "Compressor Outlet Heat Dissipation (T30)",
              severity: "HIGH",
              evidence: "T30 = 1675.0 °R near metallurgical limits of diffuser casing.",
              impact: "Risk of uncontained thermal casing breach or terminal electrical busbar flashover."
            }
          ],
          state_causality: {
            active_state: 4,
            state_name: "Accelerated Thermal Degradation",
            elapsed_dwell: 12,
            dwell_hazard_multiplier: 3.8,
            explanation: "Sustained dwell of τ=12 cycles in State 4 exponentially accelerates the phase-type transition probability into absorbing failure State 5. Dwell accumulation alone explains -21.4 cycles of remaining useful life loss.",
            counterfactual: "If asset had received maintenance before cycle 45 to revert to State 2, RUL would now be 48.5 cycles (+33.7 cycles longer)."
          },
          anomaly_decomposition: {
            primary_driver: "Emission Surprisal (53%)",
            summary: "Surprisal score (1.84) crossed the critical 1.25 threshold. Both emission magnitude and dwell duration violate normal bounds.",
            components: [
              { type: "Emission", value: 0.98, share: "53%", description: "Extreme T50/T30 thermal values deviate from learned VQ codebook centroids" },
              { type: "Transition", value: 0.44, share: "24%", description: "High probability mass transitioning into terminal absorbing state" },
              { type: "Duration", value: 0.42, share: "23%", description: "Elapsed dwell τ=12 exceeds the maximum expected dwell mean (8.4 steps) in State 4" }
            ]
          },
          actionable_maintenance: {
            primary_action: "Immediate Physical Borescope Inspection & Overhaul Work Order",
            time_to_act: "Within 11 cycles (Conformal 90% Lower Bound: 11.2 cycles)",
            operational_mitigation: "De-rate engine continuous throttle by 15% immediately to drop T50 below 1350 °R while awaiting technician overhaul.",
            maintenance_procedures: [
              "1. Dispatch certified technician to perform immediate borescope inspection of HPT stage 1 nozzle guide vanes",
              "2. Measure four-wire micro-ohmmeter contact resistance on main power electrical busbar contacts",
              "3. Inspect turbine thermocouple harness for calibration drift or sensor lead shorting",
              "4. Issue formal work order for hot section overhaul before cycle 69"
            ]
          }
        }
      }
    ]
  },
  scenario_d: {
    id: "scenario_d",
    title: "Scenario D: Model Unreliability Fallback (Edge Case)",
    description: "Demonstrating system behavior when structured representation is unreliable. Codebook error > 3x variance and belief entropy H=1.58 nats suppress structured RUL and trigger raw sensor heuristic.",
    steps: [
      {
        step_index: 0,
        cycle: 65,
        window_id: "W-065-EDGE",
        state_id: null,
        state_name: "Unmodeled / Out-of-Vocabulary Regime",
        dwell: null,
        event_id: null,
        event_symbol: "E??",
        event_desc: "Quantization Divergence / Unseen Dynamic",
        sensor_readings: {
          alt: 35000.0, Mach: 0.82, TRA: 95.0, T2: 465.0,
          T24: 745.0, T30: 1780.0, T50: 1490.0, P2: 7.4,
          P15: 22.0, P30: 590.0, Nf: 2480.0, Nc: 9850.0,
          epr: 2.15, Ps30: 580.0, Wf: 6.85, Nf_dmd: 2480.0,
          Nc_dmd: 9850.0, BPR: 5.80
        },
        latent_vector_norm: 2.45,
        codebook_reconstruction_err: 0.482,
        posterior: [0.20, 0.22, 0.19, 0.21, 0.18],
        is_unreliable: true,
        unreliable_reason: "High belief entropy H=1.58 nats & codebook reconstruction error 0.482 exceeds 3x variance bound.",
        rul: {
          expected: null,
          ci90: [null, null],
          suppressed: true,
          fallback_alarm: "CRITICAL RAW-SENSOR HEURISTIC TRIGGERED",
          fallback_reason: "T50 (1490 °R) exceeds 1420 °R threshold. Structured estimate suppressed to prevent hallucination."
        },
        anomaly: { score: 3.42, threshold: 1.25, is_anomalous: true, components: { emission: 2.10, transition: 0.82, duration: 0.50 } },
        fault: { class: null, probabilities: [0.33, 0.34, 0.33], label: "Unclassifiable (Out of Distribution)" },
        health: { index: 0.10, trend: -0.60, status: "Severe Anomaly" },
        decision: {
          tier: "CRITICAL: IMMEDIATE OPERATOR INSPECTION",
          action_text: "Dispatch technician immediately for physical inspection. Structured RUL suppressed due to high model entropy.",
          human_in_the_loop: true,
          explanation: "Structured grammar cannot confidently assign a physical degradation state. Fallback heuristic active; note fallback is unvalidated pending locked evaluation."
        },
        engineering_insights: {
          root_causes: [
            {
              subsystem: "Severe Out-of-Distribution Thermal Spike (T50, T30)",
              severity: "CRITICAL",
              evidence: "T50 = 1490.0 °R exceeds emergency shutdown limit of 1420.0 °R; T30 = 1780.0 °R (+180 °R over envelope).",
              impact: "Severe thermal runaway or combustion nozzle blowout resulting in out-of-vocabulary operational dynamics."
            },
            {
              subsystem: "Quantization Codebook Divergence",
              severity: "CRITICAL",
              evidence: "Latent vector distance ||z - e_k|| = 0.482 (> 3x training variance bound).",
              impact: "Latent vector falls outside known K=16 codebook space, triggering safe rejection."
            }
          ],
          state_causality: {
            active_state: null,
            state_name: "Unmodeled / Out-of-Vocabulary Regime",
            elapsed_dwell: null,
            dwell_hazard_multiplier: 5.0,
            explanation: "HSMM belief entropy is uniform (H=1.58 nats across all 5 states). The model cannot determine physical state.",
            counterfactual: "Structured RUL formula phase-type reachability diverges under uniform entropy; model correctly suppresses structured RUL rather than emitting confident hallucinations."
          },
          anomaly_decomposition: {
            primary_driver: "Emission Surprisal (61%)",
            summary: "Surprisal score (3.42) is 2.7x above the anomaly threshold, dominated by unprecedented raw sensor extremes.",
            components: [
              { type: "Emission", value: 2.10, share: "61%", description: "Raw thermal telemetry lies entirely outside training distribution" },
              { type: "Transition", value: 0.82, share: "24%", description: "Unresolvable state transition topology" },
              { type: "Duration", value: 0.50, share: "15%", description: "Dwell cannot be reliably computed" }
            ]
          },
          actionable_maintenance: {
            primary_action: "Immediate Physical Failsafe Inspection (Structured Model Suppressed)",
            time_to_act: "IMMEDIATE (0 cycles)",
            operational_mitigation: "Execute emergency engine load shedding or switchgear breaker trip. Do not rely on automated AI predictions.",
            maintenance_procedures: [
              "1. Halt automated flight cycles or isolate electrical circuit breaker immediately",
              "2. Manually test raw thermocouple voltages and RTD continuity",
              "3. Inspect combustion liner and turbine rotor for physical failure or structural breach",
              "4. Note: Fallback threshold heuristic is provisional and advisory; unvalidated pending locked evaluation"
            ]
          }
        }
      }
    ]
  }
};
