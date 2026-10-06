# NASA C-MAPSS FD001 PHM Validation Results

## 1. PHM Pipeline Reuse Architecture (No Code Duplication)

To ensure the FD001 dataset quantitatively validates the **core** AeroVigil PHM algorithms rather than functioning as an isolated pipeline, we introduced the `CMAPSSResidualAdapter`. 

Because FD001 lacks the physical operating conditions (throttle, ambient temperature) required by the AeroVigil physics-based Digital Twin, the adapter creates a **data-driven baseline** (calibrated on early, healthy engine cycles). It then feeds normalized residual z-scores directly into the exact same `EngineHealthIndex` logic used by the primary piston-engine models.

```mermaid
flowchart TD
    %% Define Styles
    classDef existing fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b
    classDef new_cmapss fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20

    %% Core AeroVigil (Piston Engine)
    subgraph AeroVigil_Piston ["AeroVigil Primary Pipeline (Aero-Piston Engine)"]
        direction TB
        sim[("Synthetic Piston Telemetry\n(RPM, CHT, EGT, Oil)")]:::existing
        dt["Physics-Based Digital Twin\n(Surrogate Estimator)"]:::existing
        sim --> dt
    end

    %% New FD001 Integration
    subgraph FD001_Validation ["C-MAPSS FD001 Validation (Turbofan HPC)"]
        direction TB
        dataset[("NASA FD001 Dataset\n(15 Informative Sensors)")]:::new_cmapss
        baseline["Data-Driven Baseline\n(First 30 Healthy Cycles)"]:::new_cmapss
        dataset --> baseline
    end

    %% The Bridge
    adapter{"CMAPSS Residual Adapter\n(Maps deviations to z-scores)"}:::new_cmapss
    
    dt --> |"Physics Residuals"| analyzer["Residual Analyzer\n(Existing Core)"]:::existing
    baseline --> |"Data-Driven Deviations"| adapter
    adapter --> |"Normalized Z-Scores"| analyzer
    
    %% Shared downstream PHM core
    analyzer --> health["Engine Health Index Module\n(Existing Core)"]:::existing
    health --> rul["Gradient Boosting Regressor\n(RUL Predictor)"]:::new_cmapss
    
    health --> h_out(["Model-Derived Health Trajectory"]):::output
    rul --> r_out(["Remaining Useful Life (RUL)"]):::output
```

> [!IMPORTANT]  
> The core `EngineHealthIndex` and downstream prognostic math remain completely unchanged. The `CMAPSSResidualAdapter` translates the FD001 HPC degradation into a compatible signature format, successfully validating the existing mathematical pipeline against a standard empirical dataset.

---

## 2. Predicted vs. True RUL (100 Test Engines)

The RUL prediction model (Gradient Boosting Regressor over a 15-cycle sliding window) was evaluated on 100 blind test engines against `RUL_FD001.txt`. 
- **Test MAE**: 15.5 cycles
- **Test RMSE**: 21.2 cycles
- **Test R²**: 0.741

![Predicted vs True RUL (FD001)](/C:/Users/Harshali%20g/.gemini/antigravity-ide/brain/0bac27fb-82e1-4039-89e6-a8eb600c4545/rul_validation_plots.png)

---

## 3. Engine Health Degradation Trajectories

Because we routed the C-MAPSS data through the AeroVigil health pipeline, we can extract the mathematically-derived Health Index (0-100) for the FD001 engines. 

As expected, all engines start near 100 (healthy) and monotonically degrade towards the critical thresholds as the HPC degradation progresses toward failure.

![Health Trajectories (FD001)](/C:/Users/Harshali%20g/.gemini/antigravity-ide/brain/0bac27fb-82e1-4039-89e6-a8eb600c4545/health_trajectories.png)
