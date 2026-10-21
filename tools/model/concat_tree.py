# WebGPU shaders get at most 8 storage buffers: split Concat nodes with more than 6 inputs into a tree.
import onnx, sys
from onnx import helper
m = onnx.load(sys.argv[1]); g = m.graph
out, k = [], 0
for n in g.node:
    if n.op_type != 'Concat' or len(n.input) <= 6: out.append(n); continue
    axis = [a.i for a in n.attribute if a.name == 'axis'][0]
    level = list(n.input)
    while len(level) > 6:
        nxt = []
        for i in range(0, len(level), 6):
            grp = level[i:i + 6]
            if len(grp) == 1: nxt.append(grp[0]); continue
            name = f'{n.name}_tree{k}'; k += 1
            out.append(helper.make_node('Concat', grp, [name], name=name, axis=axis)); nxt.append(name)
        level = nxt
    out.append(helper.make_node('Concat', level, list(n.output), name=n.name, axis=axis))
del g.node[:]; g.node.extend(out)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2]); print('added', k, 'concat nodes; max inputs', max(len(n.input) for n in g.node if n.op_type == 'Concat'))
