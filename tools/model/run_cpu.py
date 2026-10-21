import onnxruntime as ort, numpy as np, sys, time
from PIL import Image
im = Image.open(sys.argv[1]).convert('RGB'); W,H = im.size
o = ort.SessionOptions(); o.intra_op_num_threads = 1
s = ort.InferenceSession(sys.argv[2], o, providers=['CPUExecutionProvider'])
x = np.asarray(im.resize((1024,1024), Image.BILINEAR)).astype(np.float32)/255
x = ((x-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])).astype(np.float32).transpose(2,0,1)[None]
t=time.time(); out = s.run(None, {s.get_inputs()[0].name: x})[0][0,0]; print('1-thread secs', round(time.time()-t,1), out.min(), out.max())
if out.min() < 0 or out.max() > 1: out = 1/(1+np.exp(-out))
Image.fromarray((out*255).astype(np.uint8)).resize((W,H), Image.BILINEAR).save(sys.argv[3])
