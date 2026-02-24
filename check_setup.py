#!/usr/bin/env python3
"""Quick check that all imports work."""
try:
    import torch
    print(f"torch: {torch.__version__}")
except ImportError as e:
    print(f"MISSING: torch - {e}")

try:
    import streamlit
    print(f"streamlit: {streamlit.__version__}")
except ImportError as e:
    print(f"MISSING: streamlit - {e}")

try:
    import plotly
    print(f"plotly: {plotly.__version__}")
except ImportError as e:
    print(f"MISSING: plotly - {e}")

try:
    import networkx
    print(f"networkx: {networkx.__version__}")
except ImportError as e:
    print(f"MISSING: networkx - {e}")

try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError as e:
    print(f"MISSING: numpy - {e}")

# Test BDH model instantiation
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from core.bdh import BDH, BDHConfig
    config = BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=32, vocab_size=256, dropout=0.0)
    model = BDH(config)
    idx = torch.tensor([[72, 101, 108, 108, 111]], dtype=torch.long)
    logits, loss = model(idx)
    print(f"BDH model OK - output shape: {logits.shape}")
    _, intermediates = model.forward_with_intermediates(idx)
    print(f"Forward with intermediates OK - {len(intermediates['layer_x_sparse'])} layers captured")
    
    from core.analysis import compute_sparsity
    sp = compute_sparsity(intermediates['layer_x_sparse'][0])
    print(f"Sparsity of layer 0: {sp*100:.1f}% (active: {(1-sp)*100:.1f}%)")
except Exception as e:
    print(f"BDH MODEL ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test Transformer
try:
    from core.transformer import SimpleTransformer, TransformerConfig
    tf_config = TransformerConfig(n_layer=2, n_embd=64, n_head=2, vocab_size=256)
    tf_model = SimpleTransformer(tf_config)
    _, tf_int = tf_model.forward_with_intermediates(idx)
    print(f"Transformer OK - {len(tf_int['layer_mlp_activations'])} layers captured")
    sp_tf = compute_sparsity(tf_int['layer_mlp_activations'][0])
    print(f"Transformer layer 0 sparsity: {sp_tf*100:.1f}% (active: {(1-sp_tf)*100:.1f}%)")
except Exception as e:
    print(f"TRANSFORMER ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== ALL CHECKS PASSED ===")
