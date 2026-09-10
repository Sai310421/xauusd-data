# FEATURES

DexG feature vectors come from the Python ML Builder only.

Allowed:
- feature_order saved by the builder model JSON
- causal reconstruction via the builder's own feature modules
- custom_features / advanced_features computed by that builder

Forbidden in this lab:
- G75 geometry or G75 artifacts
- Hydra66 66-dim contract
- TSUGI / Negative Memory features
- mixing builder columns with Hydra66 or G75 columns in one vector
- renaming Hydra66 fields to look like builder fields

If a vector cannot be rebuilt from builder feature_order + builder functions, it is not a DexG feature vector.
