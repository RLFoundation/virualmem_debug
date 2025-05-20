// Implementation of functions extracted from cumem_allocator.cpp
#define USE_ROCM

#include <iostream>
#include <sched.h>       // For CPU affinity functions
#include <unistd.h>      // For syscall
#include <sys/syscall.h> // For SYS_gettid
#include <hip/hip_runtime.h>

// Define some required macros
#define MEMCREATE_CHUNK_SIZE (2 * 1024 * 1024)
#define MIN(a, b) (a < b ? a : b)
#define ENABLE_DEBUG_CUMEM

// Include compatibility layer
#include "cumem_allocator_compat.h"

// Implementation of ensure_context
void ensure_context(unsigned long long device) {
  CUcontext pctx;
  CUresult result = cuCtxGetCurrent(&pctx);
  if (result != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(result, &error_string);
    std::cerr << "CUDA Error in cuCtxGetCurrent: " << error_string << std::endl;
    return;
  }

  if (!pctx) {
    // Ensure device context.
    result = cuDevicePrimaryCtxRetain(&pctx, device);
    if (result != CUDA_SUCCESS) {
      const char* error_string;
      cuGetErrorString(result, &error_string);
      std::cerr << "CUDA Error in cuDevicePrimaryCtxRetain: " << error_string << std::endl;
      return;
    }
    
    result = cuCtxSetCurrent(pctx);
    if (result != CUDA_SUCCESS) {
      const char* error_string;
      cuGetErrorString(result, &error_string);
      std::cerr << "CUDA Error in cuCtxSetCurrent: " << error_string << std::endl;
      return;
    }
  }
}

// Helper function to set CPU affinity based on GPU device ID
void set_cpu_affinity_for_gpu(unsigned long long device) {
  // Get current thread ID
  pid_t tid = syscall(SYS_gettid);
  
  cpu_set_t mask;
  CPU_ZERO(&mask);
  unsigned long long _device = device;
  // GPUs 0-3 are on NUMA node 0 (CPUs 0-47)
  // GPUs 4-7 are on NUMA node 1 (CPUs 48-95)
  if (_device <= 3) {
    // Set affinity to NUMA node 0 CPUs
    for (int i = 0; i < 48; i++) {
      CPU_SET(i, &mask);
    }
    std::cout << "Setting affinity for GPU " << device << " to NUMA node 0 (CPUs 0-47)" << std::endl;
  } else {
    // Set affinity to NUMA node 1 CPUs
    for (int i = 48; i < 96; i++) {
      CPU_SET(i, &mask);
    }
    std::cout << "Setting affinity for GPU " << device << " to NUMA node 1 (CPUs 48-95)" << std::endl;
  }
  
  // Set the CPU affinity for this thread
  int result = sched_setaffinity(tid, sizeof(mask), &mask);
  if (result != 0) {
    std::cerr << "Failed to set CPU affinity for thread " << tid << ": " << strerror(errno) << std::endl;
  }
}

// Implementation of create_and_map
void create_and_map(unsigned long long device, ssize_t size, CUdeviceptr d_mem,
                   CUmemGenericAllocationHandle** p_memHandle,
                   unsigned long long* chunk_sizes, size_t num_chunks) {
  ensure_context(device);
  
  // Set CPU affinity based on the GPU device
  set_cpu_affinity_for_gpu(device);

  // Define memory allocation properties
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

  // Create memory handles for each chunk
  for (auto i = 0; i < num_chunks; ++i) {
    CUresult result = cuMemCreate(p_memHandle[i], chunk_sizes[i], &prop, 0);
    if (result != CUDA_SUCCESS) {
      const char* error_string;
      cuGetErrorString(result, &error_string);
      std::cerr << "CUDA Error in cuMemCreate for chunk " << i << ": " << error_string << std::endl;
#ifdef ENABLE_DEBUG_CUMEM
      std::cout << "cuMemCreate failed: " << i << std::endl;
#endif
      return;
    }
#ifdef ENABLE_DEBUG_CUMEM
    std::cout << "p_memHandle[" << i << "] = " << *p_memHandle[i] << std::endl;
#endif
  }

  // Map each chunk to device memory
  unsigned long long allocated_size = 0;
  for (auto i = 0; i < num_chunks; ++i) {
    void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
    CUresult result = cuMemMap(map_addr, chunk_sizes[i], 0, *(p_memHandle[i]), 0);
    if (result != CUDA_SUCCESS) {
      const char* error_string;
      cuGetErrorString(result, &error_string);
      std::cerr << "CUDA Error in cuMemMap for chunk " << i << ": " << error_string << std::endl;
#ifdef ENABLE_DEBUG_CUMEM
      std::cout << "cuMemMap failed: " << i << std::endl;
#endif
      return;
    }
    allocated_size += chunk_sizes[i];
#ifdef ENABLE_DEBUG_CUMEM
    std::cout << "allocated_size = " << allocated_size << std::endl;
#endif
  }

  // Set memory access permissions
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUresult result = cuMemSetAccess(d_mem, size, &accessDesc, 1);
  if (result != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(result, &error_string);
    std::cerr << "CUDA Error in cuMemSetAccess: " << error_string << std::endl;
    return;
  }
  
#ifdef ENABLE_DEBUG_CUMEM
  std::cout << "create_and_map: device=" << device << ", size=" << size 
            << ", d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
#endif
}

// Implementation of unmap_and_release
void unmap_and_release(unsigned long long device, ssize_t size,
                       CUdeviceptr d_mem,
                       CUmemGenericAllocationHandle** p_memHandle,
                       unsigned long long* chunk_sizes, size_t num_chunks) {
#ifdef ENABLE_DEBUG_CUMEM
  std::cout << "unmap_and_release: device=" << device << ", size=" << size 
            << ", d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
#endif
  ensure_context(device);

  // Unmap each chunk
  unsigned long long allocated_size = 0;
  for (auto i = 0; i < num_chunks; ++i) {
    void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
    CUresult result = cuMemUnmap(map_addr, chunk_sizes[i]);
    if (result != CUDA_SUCCESS) {
      const char* error_string;
      cuGetErrorString(result, &error_string);
      std::cerr << "CUDA Error in cuMemUnmap for chunk " << i << ": " << error_string << std::endl;
#ifdef ENABLE_DEBUG_CUMEM
      std::cout << "cuMemUnmap failed" << std::endl;
#endif
      return;
    }
    allocated_size += chunk_sizes[i];
  }

  // Release each memory handle
  for (auto i = 0; i < num_chunks; ++i) {
    CUresult result = cuMemRelease(*(p_memHandle[i]));
    if (result != CUDA_SUCCESS) {
      const char* error_string;
      cuGetErrorString(result, &error_string);
      std::cerr << "CUDA Error in cuMemRelease for chunk " << i << ": " << error_string << std::endl;
#ifdef ENABLE_DEBUG_CUMEM
      std::cout << "cuMemRelease failed" << std::endl;
#endif
      return;
    }
  }
} 