import multiprocessing
import platform

print("CPU cores:", multiprocessing.cpu_count())
print("Platform:", platform.system(), platform.release())
