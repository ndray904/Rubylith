# bir_web3 (fp16-stored weights + Cast) -> fp32 weights -> onnxsim at fixed 1x3x1024x1024 -> fp16-stored again
import onnx, numpy as np, sys, collections
from onnx import helper, numpy_helper
import onnxsim
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
# 1. fp16 initializers -> fp32, drop the Cast-to-float nodes that consumed them
f16 = {t.name for t in g.initializer if t.data_type == onnx.TensorProto.FLOAT16}
new = []
for t in g.initializer:
    new.append(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name) if t.name in f16 else t)
del g.initializer[:]; g.initializer.extend(new)
for n in g.node:
    if n.op_type == 'Cast' and n.input[0] in f16:
        n.op_type = 'Identity'; del n.attribute[:]
for i in g.input:
    d = i.type.tensor_type.shape.dim
    for k, v in enumerate([1, 3, 1024, 1024]): d[k].ClearField('dim_param'); d[k].dim_value = v
print('fp32 nodes', len(g.node))
ms, ok = onnxsim.simplify(m, skip_shape_inference=False)
print('simplified ok', ok, 'nodes', len(ms.graph.node))
# 2. store large float initializers as fp16 + Cast (only when exact enough)
g = ms.graph
casts, inits = [], []
for t in g.initializer:
    a = numpy_helper.to_array(t)
    if a.dtype == np.float32 and a.size >= 1024:
        h = a.astype(np.float16)
        if np.isfinite(h).all() and np.abs(h.astype(np.float32) - a).max() <= 1e-3 * max(1.0, np.abs(a).max()):
            inits.append(numpy_helper.from_array(h, t.name + '_f16'))
            casts.append(helper.make_node('Cast', [t.name + '_f16'], [t.name], to=onnx.TensorProto.FLOAT))
            continue
    inits.append(t)
del g.initializer[:]; g.initializer.extend(inits)
nodes = casts + list(g.node); del g.node[:]; g.node.extend(nodes)
onnx.save(ms, dst)
print(collections.Counter(n.op_type for n in ms.graph.node).most_common())
print('fp32 kept', sum(1 for t in g.initializer if t.data_type == 1 and np.prod(t.dims) >= 1024))
