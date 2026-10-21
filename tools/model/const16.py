# Large float Constant nodes whose values are exact in fp16 (attention masks, sampling grids) -> fp16 initializer + Cast.
import onnx, numpy as np, sys
from onnx import helper, numpy_helper
m = onnx.load(sys.argv[1]); g = m.graph
out, n16 = [], 0
for n in g.node:
    if n.op_type == 'Constant' and n.attribute[0].name == 'value':
        a = numpy_helper.to_array(n.attribute[0].t)
        if a.dtype == np.float32 and a.size > 1000 and np.array_equal(a.astype(np.float16).astype(np.float32), a):
            nm = n.output[0] + '__c16'
            g.initializer.append(numpy_helper.from_array(a.astype(np.float16), nm))
            out.append(helper.make_node('Cast', [nm], [n.output[0]], name=nm + '_cast', to=onnx.TensorProto.FLOAT)); n16 += 1
            continue
    out.append(n)
del g.node[:]; g.node.extend(out)
onnx.checker.check_model(m); onnx.save(m, sys.argv[2]); print('converted', n16)
