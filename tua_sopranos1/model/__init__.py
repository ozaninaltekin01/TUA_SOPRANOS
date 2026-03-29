"""
TUA SOPRANOS — K2 Model Modülü
================================
CARA tabanlı çarpışma risk analizi, manevra karar motoru,
ML risk sınıflandırması ve LSTM yörünge tahmini.

Kullanım (K3 için):
    from model.cara_engine   import run_cara_from_k1, generate_cdm
    from model.maneuver      import suggest_maneuver, shadow_zone_analysis
    from model.game_theory   import who_should_dodge, fuel_budget_manager
    from model.threat_analysis import detect_ghost_maneuver, fragmentation_warning
    from model.ml_model      import predict_risk, hybrid_risk_assessment
    from model.ml_model      import load_lstm_model, predict_orbit, hybrid_full_pipeline
    from model.ml_training   import run_training
    from model.model_evaluation import run_full_evaluation, quick_sanity_check
"""

from .cara_engine import (
    run_cara_assessment,
    run_cara_from_k1,
    batch_cara_assessment,
    generate_cdm,
    compute_pc,
    assess_cara_status,
    find_tca,
    build_encounter_frame,
    project_to_2d,
    create_mock_scenario,
    CARA_THRESHOLDS,
)

from .maneuver import (
    suggest_maneuver,
    tsiolkovsky_fuel_mass,
    estimate_lifetime_impact,
    compute_delta_v_options,
    recalculate_pc_after_maneuver,
    shadow_zone_analysis,
)

from .threat_analysis import (
    detect_ghost_maneuver,
    fragmentation_warning,
    prioritize_threats,
    j2_expected_drift,
)

from .game_theory import (
    who_should_dodge,
    fuel_budget_manager,
    build_payoff_matrix,
    find_nash_equilibrium,
)

from .ml_model import (
    # XGBoost
    predict_risk,
    batch_predict,
    hybrid_risk_assessment,
    extract_features,
    load_model,
    get_model_info,
    # LSTM
    load_lstm_model,
    predict_orbit,
    compare_with_sgp4,
    # End-to-end
    hybrid_full_pipeline,
)

from .model_evaluation import (
    run_full_evaluation,
    evaluate_xgboost,
    evaluate_lstm,
    quick_sanity_check,
    plot_feature_importance,
)
