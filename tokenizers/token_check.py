#%%
import numpy as np
import matplotlib.pyplot as plt
import os

#################################################################
# 1. CONFIGURATION
#################################################################

# --- ⚠️ SET THESE 2 VARIABLES ---
# Specify the patient ID you want to look at
PATIENT_ID_TO_INSPECT = "01" # <--- CHANGE THIS
# Path to the directory where you saved the .npy files
TOKENS_DIR = r"D:/PULSE/tokens"
# --------------------------------

# Construct file paths
cls_file = os.path.join(TOKENS_DIR, f"{PATIENT_ID_TO_INSPECT}_cls.npy")
patch_file = os.path.join(TOKENS_DIR, f"{PATIENT_ID_TO_INSPECT}_patch.npy")

#################################################################
# 2. LOAD AND PROCESS DATA
#################################################################

try:
    # --- Load the CLS token ---
    # This is the (512,) vector from [CLS] token
    cls_token_vec = np.load(cls_file)
    
    # --- Load the PATCH tokens ---
    # This is the (S, 512) matrix, where S is the sequence length
    patch_tokens_matrix = np.load(patch_file)
    
    # --- Average the PATCH tokens (Global Average Pooling) ---
    # This is what your main script does for FEATURE_MODE = 'patch'
    # np.mean(patch_tokens_matrix, axis=0) gives a (512,) vector
    avg_patch_token_vec = np.mean(patch_tokens_matrix, axis=0)

    print(f"--- Successfully loaded data for Patient: {PATIENT_ID_TO_INSPECT} ---")
    print(f"CLS Token Shape:     {cls_token_vec.shape}")
    print(f"Patch Tokens Shape:  {patch_tokens_matrix.shape} (S, embed_dim)")
    print(f"Avg. Patch Shape:  {avg_patch_token_vec.shape}")

except FileNotFoundError:
    print(f"ERROR: Files not found for patient '{PATIENT_ID_TO_INSPECT}'.")
    print(f"Checked for: {cls_file}")
    print(f"And:         {patch_file}")
    print("Please make sure the patient ID is correct and files exist.")
    # Stop execution if files aren't found
    raise

#################################################################
# 3. VISUALIZE DISTRIBUTIONS
#################################################################

print("\nGenerating plots...")

# === PLOT 1: Overlaid Histograms (Direct Comparison) ===
plt.figure(figsize=(12, 6))
plt.hist(cls_token_vec, bins=50, alpha=0.7, label='[CLS] Token Vector', density=True)
plt.hist(avg_patch_token_vec, bins=50, alpha=0.7, label='Averaged Patch Token Vector', density=True)
plt.title(f'Feature Distributions for Patient {PATIENT_ID_TO_INSPECT}\n(CLS vs. Averaged Patch)')
plt.xlabel('Feature Value')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# === PLOT 2: Heatmap of Raw Patch Tokens ===
# This shows the "stuff" that gets averaged
S, C = patch_tokens_matrix.shape
plt.figure(figsize=(15, 7))
# We use imshow to visualize the (S, C) matrix
# We transpose it to (C, S) so features are on y-axis, sequence on x-axis
im = plt.imshow(patch_tokens_matrix.T, aspect='auto', cmap='viridis')
plt.colorbar(im, label='Feature Value')
plt.title(f'Raw Patch Token Matrix for Patient {PATIENT_ID_TO_INSPECT}')
plt.xlabel(f'Sequence of Patches (S = {S})')
plt.ylabel(f'Feature Dimension (C = {C})')
plt.tight_layout()
plt.show()

print("✅ Analysis complete.")