import unittest
from tinygrad.helpers import Timing, CI, Profiling
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad import dtypes

class TestTensorcore(unittest.TestCase):
  def setUp(self): print()
  def tearDown(self): print()

  def test_simple(self):
    x = Tensor.empty((256,256), dtype=dtypes.half)
    y = Tensor.empty((256,256), dtype=dtypes.half)
    #x = Tensor.empty(512,512)
    #y = Tensor.empty(512,512)

    with Timing("running conv: "):
      out = x.matmul(y)

    with Timing("scheduling: "):
      sched = out.lazydata.schedule()

    for i,s in enumerate(sched):
      if s.ast.op in LoadOps: continue
      ops = s.ast.lazyops
      with Timing(f"linearize {i} with {len(ops):4d} ops: "):
        l = Linearizer(s.ast)
        print(l.apply_tensor_cores())
        #l.hand_coded_optimizations()
        l.linearize()

if __name__ == '__main__':
  unittest.main(verbosity=2)