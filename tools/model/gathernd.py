# ORT-web 1.22's WebGPU GatherND shader fails to compile ("i32(vec4<u32>)"), so those nodes silently do nothing.
# BiRefNet's deformable-conv sampling uses GatherND(data[1,1,H,W,C], idx[1,1,N,2], batch_dims=2) = data[0,0,y,x,:].
# Rewrite as Gather(data reshaped [H*W, C], y*W + x) in int32, using only Split/Mul/Add (no int ReduceSum).
import onnx, numpy as np, sys
from onnx import helper, numpy_helper, TensorProto
m = onnx.shape_inference.infer_shapes(onnx.load(sys.argv[1])); g = m.graph
vi = {v.name: v for v in list(g.value_info) + list(g.input)}
dims = lambda n: [d.dim_value for d in vi[n].type.tensor_type.shape.dim]
out, inits, k = [], [], 0
def const(name, arr): inits.append(numpy_helper.from_array(np.array(arr), name)); return name
for n in g.node:
    if n.op_type != 'GatherND': out.append(n); continue
    data, idx = n.input; _, _, H, W, C = dims(data); N = dims(idx)[2]
    assert [a.i for a in n.attribute if a.name == 'batch_dims'] == [2]
    p = f'gnd{k}_'; k += 1
    out += [
        helper.make_node('Cast', [idx], [p + 'i32'], to=TensorProto.INT32),
        helper.make_node('Reshape', [p + 'i32', const(p + 's1', np.array([N, 2], np.int64))], [p + 'yx']),
        helper.make_node('Split', [p + 'yx', const(p + 'sp', np.array([1, 1], np.int64))], [p + 'y', p + 'x'], axis=1),
        helper.make_node('Mul', [p + 'y', const(p + 'w', np.array(W, np.int32))], [p + 'yw']),
        helper.make_node('Add', [p + 'yw', p + 'x'], [p + 'lin2']),
        helper.make_node('Reshape', [p + 'lin2', const(p + 's4', np.array([N], np.int64))], [p + 'lin']),
        helper.make_node('Reshape', [data, const(p + 's2', np.array([H * W, C], np.int64))], [p + 'flat']),
        helper.make_node('Gather', [p + 'flat', p + 'lin'], [p + 'rows'], axis=0),
        helper.make_node('Reshape', [p + 'rows', const(p + 's3', np.array([1, 1, N, C], np.int64))], list(n.output)),
    ]
del g.node[:]; g.node.extend(out); g.initializer.extend(inits)
del g.value_info[:]
onnx.checker.check_model(m); onnx.save(m, sys.argv[2]); print('rewrote', k)
