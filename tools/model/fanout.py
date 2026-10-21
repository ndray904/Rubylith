# Rewrite Concat/Split nodes with many tensors into trees of <=6, so each WebGPU kernel stays under 8 storage buffers.
import onnx, numpy as np, sys
from onnx import helper, numpy_helper, TensorProto
K = 6
m = onnx.load(sys.argv[1]); g = m.graph
consts = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in g.node if n.op_type == 'Constant' and n.attribute and n.attribute[0].name == 'value'}
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
uid = [0]
def nm(base): uid[0] += 1; return f'{base}__fx{uid[0]}'
out = []; nc = ns = 0
def concat_tree(inputs, output, axis, name):
    global nc
    if len(inputs) <= K:
        out.append(helper.make_node('Concat', inputs, [output], axis=axis, name=name)); return
    groups = [inputs[i:i + K] for i in range(0, len(inputs), K)]
    mids = []
    for gi in groups:
        if len(gi) == 1: mids.append(gi[0]); continue
        t = nm(output); concat_tree(gi, t, axis, nm(name)); mids.append(t)
    concat_tree(mids, output, axis, name)
def split_tree(inp, sizes, outputs, axis, name):
    if len(outputs) <= K:
        sname = nm(name + '_sizes')
        out.append(helper.make_node('Constant', [], [sname], value=numpy_helper.from_array(np.array(sizes, dtype=np.int64)), name=sname))
        out.append(helper.make_node('Split', [inp, sname], outputs, axis=axis, name=name)); return
    # group outputs so the top split has <= K parts
    per = -(-len(outputs) // K)
    groups = [list(range(i, min(i + per, len(outputs)))) for i in range(0, len(outputs), per)]
    mids, msizes = [], []
    for gi in groups:
        msizes.append(int(sum(sizes[j] for j in gi)))
        mids.append(outputs[gi[0]] if len(gi) == 1 else nm(name + '_grp'))
    split_tree(inp, msizes, mids, axis, nm(name))
    for gi, mid in zip(groups, mids):
        if len(gi) > 1: split_tree(mid, [sizes[j] for j in gi], [outputs[j] for j in gi], axis, nm(name))
for n in g.node:
    axis = next((a.i for a in n.attribute if a.name == 'axis'), 0)
    if n.op_type == 'Concat' and len(n.input) > K:
        concat_tree(list(n.input), n.output[0], axis, n.name); nc += 1; continue
    if n.op_type == 'Split' and len(n.output) > K:
        src = n.input[1] if len(n.input) > 1 else None
        sizes = consts.get(src, inits.get(src)) if src else None
        if sizes is None: print('cannot resolve sizes for', n.name); out.append(n); continue
        split_tree(n.input[0], [int(v) for v in sizes], list(n.output), axis, n.name); ns += 1; continue
    out.append(n)
del g.node[:]; g.node.extend(out)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2]); print('rewrote concat', nc, 'split', ns)
