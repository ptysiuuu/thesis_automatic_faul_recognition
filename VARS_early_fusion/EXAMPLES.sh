# VARS Explainability Configuration Examples
# Copy and modify these commands for your specific use case

# =============================================================================
# EXAMPLE 1: Full Pipeline with Gemini API (Recommended)
# =============================================================================
# High-quality explanations, no local GPU memory constraint

python explain.py \
    --path /data/SoccerNet \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --split test \
    --num_views 5 \
    --frames_per_view 16 \
    --fps 12 \
    --fusion_mode \
    --llm_backend gemini \
    --gemini_api_key YOUR_GOOGLE_API_KEY \
    --llm_temperature 0.7 \
    --output_dir ./explanations_test


# =============================================================================
# EXAMPLE 2: Quick Test - Evidence Only (No LLM)
# =============================================================================
# Fastest way to test the pipeline, no API key needed

python explain.py \
    --path /data/SoccerNet \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --split test \
    --skip_llm \
    --output_dir ./explanations_test


# =============================================================================
# EXAMPLE 3: Multi-View Architecture with Transformer Aggregation
# =============================================================================
# For MVNetwork models instead of EarlyFusionNetwork

python explain.py \
    --path /data/SoccerNet \
    --checkpoint ./models/VARS_model/5/mvit_v2_s/0.0001/_B4_F16_G0.1_Step3/model_best.pth \
    --split test \
    --num_views 5 \
    --backbone mvit_v2_s \
    --aggregation transformer \
    --graph_topology structured \
    --llm_backend gemini \
    --gemini_api_key YOUR_GOOGLE_API_KEY \
    --output_dir ./explanations_multiview


# =============================================================================
# EXAMPLE 4: Using Local LLM (Mistral-7B)
# =============================================================================
# Requires ~16GB GPU VRAM, slower but fully private

python explain.py \
    --path /data/SoccerNet \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --split test \
    --fusion_mode \
    --llm_backend local \
    --llm_model mistralai/Mistral-7B-Instruct-v0.2 \
    --llm_temperature 0.5 \
    --max_num_worker 0 \
    --output_dir ./explanations_local


# =============================================================================
# EXAMPLE 5: Using Anthropic Claude API
# =============================================================================
# Premium quality (slower, costs money)

python explain.py \
    --path /data/SoccerNet \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --split test \
    --fusion_mode \
    --llm_backend anthropic \
    --anthropic_api_key YOUR_ANTHROPIC_API_KEY \
    --output_dir ./explanations_claude


# =============================================================================
# EXAMPLE 6: Challenge Set Evaluation
# =============================================================================
# For final evaluation on challenge set (if available)

python explain.py \
    --path /data/SoccerNet \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --split chall \
    --fusion_mode \
    --use_ema \
    --use_tta \
    --llm_backend gemini \
    --gemini_api_key YOUR_GOOGLE_API_KEY \
    --output_dir ./explanations_challenge


# =============================================================================
# EXAMPLE 7: Quick-Start Launcher Script (Simplest)
# =============================================================================
# Uses sensible defaults, minimal configuration needed

python run_explainability.py \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --dataset_path /data/SoccerNet \
    --split test \
    --gemini_api_key YOUR_GOOGLE_API_KEY \
    --create_eval_template


# =============================================================================
# ENVIRONMENT VARIABLES (Alternative to --api_key Arguments)
# =============================================================================
# Instead of passing --gemini_api_key on command line, you can set:

export GOOGLE_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here

# Then run without the key arguments:
python explain.py \
    --path /data/SoccerNet \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --split test \
    --fusion_mode \
    --llm_backend gemini \
    --output_dir ./explanations_test


# =============================================================================
# Useful shell aliases for common tasks
# =============================================================================

# In ~/.bashrc or ~/.zshrc:

alias explain_test='python explain.py \
    --path /data/SoccerNet \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --split test \
    --fusion_mode \
    --llm_backend gemini \
    --output_dir ./explanations_test'

alias explain_no_llm='python explain.py \
    --path /data/SoccerNet \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --split test \
    --skip_llm \
    --output_dir ./explanations_test'

alias explain_launcher='python run_explainability.py \
    --checkpoint ./models/VARS_early_fusion/5/early_fusion/0.0001/_B4_F8_G0.1_Step3/model_best.pth \
    --dataset_path /data/SoccerNet \
    --gemini_api_key $GOOGLE_API_KEY'

# Usage:
# $ explain_test                    # Run full pipeline
# $ explain_no_llm                  # Evidence only
# $ explain_launcher --split chall  # Challenge set
