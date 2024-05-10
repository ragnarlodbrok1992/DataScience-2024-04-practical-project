import pycuda.driver as drv
import pycuda.autoinit

drv.init()

print(f"Number of devices: {drv.Device.count()}")

for i in range(drv.Device.count()):
    dev = drv.Device(i)
    print(f"Device {i}: {dev.name()}")
    print(f"  Compute capability: {dev.compute_capability()}")
    print(f"  Total memory: {dev.total_memory() / 1024**2} MB")
