#!/usr/bin/env python3
import os, ctypes, ctypes.util, io, mmap
from tinygrad.helpers import Timing, from_mv
libc = ctypes.CDLL(ctypes.util.find_library("c"))

#from extra.hip_gpu_driver import hip_ioctl

# sudo su -c "echo 3 > /proc/sys/vm/drop_caches"

# sudo su -c 'echo 8 > /proc/sys/kernel/printk'
# sudo su -c "echo 'module amdgpu +p' > /sys/kernel/debug/dynamic_debug/control"

libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.read.restype = ctypes.c_size_t

libc.malloc.argtypes = [ctypes.c_size_t]
libc.malloc.restype = ctypes.c_void_p

def read_direct(fd, sz):
  with Timing("mmap: ", lambda x: f", {sz/x:.2f} GB/s"):
    buf = mmap.mmap(-1, sz, flags=mmap.MAP_SHARED|mmap.MAP_POPULATE)
  with Timing("read: ", lambda x: f", {sz/x:.2f} GB/s"):
    ret = libc.read(fd, from_mv(buf), sz)
  assert ret == sz

def read_mmap(fd, sz):
  with Timing("mmfd:       ", lambda x: f", {sz/x:.2f} GB/s"):
    buf = mmap.mmap(fd, sz, flags=mmap.MAP_SHARED|mmap.MAP_POPULATE) #|MAP_LOCKED)
    t = 0
    for i in range(0, sz, 0x1000): t += buf[i]

def read_to_gpu_mmap(fd, sz, gpubuf):
  with Timing("gpu copyin: ", lambda x: f", {sz/x:.2f} GB/s"):
    with Timing("mmfd:       ", lambda x: f", {sz/x:.2f} GB/s"):
      buf = mmap.mmap(fd, sz, flags=mmap.MAP_SHARED|mmap.MAP_POPULATE) #|MAP_LOCKED)
    dev.allocator._copyin_async(gpubuf, from_mv(buf), sz)
    dev.synchronize()

def read_to_gpu_single(fd, sz, gpubuf):
  os.lseek(fd, 0, os.SEEK_SET)
  with Timing("total: ", lambda x: f", {sz/x:.2f} GB/s"):
    with Timing("gpu host alloc: ", lambda x: f", {sz/x:.2f} GB/s"):
      hst = dev.allocator._hostalloc(sz)
    with Timing("read to host:   ", lambda x: f", {sz/x:.2f} GB/s"):
      ret = libc.read(fd, hst, sz)
    with Timing("gpu host copy:  ", lambda x: f", {sz/x:.2f} GB/s"):
      dev.allocator._copyin_async(gpubuf, hst, sz)
      dev.synchronize()

def read_to_gpu_pingpong(fd, sz, gpubuf):
  PIECE = 8
  psz = sz//PIECE
  print(f"piece size {psz:#x}")
  with Timing("gpu host alloc: ", lambda x: f", {sz/x:.2f} GB/s"):
    hst1 = dev.allocator._hostalloc(psz)
    hst2 = dev.allocator._hostalloc(psz)

  os.lseek(fd, 0, os.SEEK_SET)
  with Timing("total: ", lambda x: f", {sz/x:.2f} GB/s"):
    for i in range(PIECE//2):
      with Timing("tfer(0):           ", lambda x: f", {psz/x:.2f} GB/s"):
        ret = libc.read(fd, hst1, psz)
        dev.synchronize()
        dev.allocator._copyin_async(gpubuf, hst1, psz)
      with Timing("tfer(1):           ", lambda x: f", {psz/x:.2f} GB/s"):
        ret = libc.read(fd, hst2, psz)
        dev.synchronize()
        dev.allocator._copyin_async(gpubuf, hst2, psz)
    dev.synchronize()

MAP_LOCKED = 0x2000
MAP_HUGETLB = 0x40000

from tinygrad.runtime.ops_hip import HIPDevice

if __name__ == "__main__":
  # 4GB of random numbers
  fd = os.open("/home/tiny/tinygrad/weights/rng", os.O_RDWR|os.O_DIRECT)
  #sz = (os.fstat(fd).st_size) // 4
  #sz = 128*1024*1024
  #sz = 256*1024*1024
  sz = 1024*1024*1024
  print(f"read {sz} from {fd}")

  dev = HIPDevice()
  with Timing("gpu alloc:  ", lambda x: f", {sz/x:.2f} GB/s"):
    gpubuf = dev.allocator._alloc(sz)
  # warmup
  dev.allocator._copyin_async(gpubuf, from_mv(bytearray(b"\x00\x00\x00\x00"*0x1000)), 0x4000)
  print("copying, is warm")

  print("****** read direct")
  read_direct(fd, sz)

  print("****** read mmap")
  read_mmap(fd, sz)

  print("****** read to gpu pingpong")
  read_to_gpu_pingpong(fd, sz, gpubuf)

  print("****** read to gpu single")
  read_to_gpu_single(fd, sz, gpubuf)

  print("****** read to gpu mmap")
  read_to_gpu_mmap(fd, sz, gpubuf)

  os._exit(0)
