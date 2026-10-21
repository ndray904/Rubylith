# Third-party components

| Component | Where | License |
|---|---|---|
| BiRefNet-lite (ZhengPeng7/BiRefNet) | `docs/m/bir6/`, converted as described in the README | MIT: see `licenses/BiRefNet-MIT.txt` |
| ISNet, quantized, from `@imgly/background-removal-data` 1.4.5 | `docs/m/small/` | AGPL-3.0: see `licenses/ISNet-imgly-AGPL-3.0.md` |
| ONNX Runtime Web 1.22.0 (Microsoft) | `docs/ort/jsep/` (WebAssembly); JavaScript loaded from jsDelivr | MIT: see `licenses/onnxruntime-MIT.txt` |
| onnx-simplifier | used by `tools/model/simp.py` (not bundled) | Apache-2.0 |

The edge-refinement approach (a color-guided filter in a band around the edge) follows the matting code in PhotoCraft.
It was reimplemented here, not copied.
