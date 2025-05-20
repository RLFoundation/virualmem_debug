// Test program for create_and_map and unmap_and_release functions on ROCM
#define USE_ROCM

#include <iostream>
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <chrono>

// Include the compatibility header from local directory
#include "cumem_allocator_compat.h"

// Function prototypes from cumem_allocator.cpp
void create_and_map(unsigned long long device, ssize_t size, CUdeviceptr d_mem,
                    CUmemGenericAllocationHandle** p_memHandle,
                    unsigned long long* chunk_sizes, size_t num_chunks);

void unmap_and_release(unsigned long long device, ssize_t size, CUdeviceptr d_mem,
                       CUmemGenericAllocationHandle** p_memHandle,
                       unsigned long long* chunk_sizes, size_t num_chunks);

void ensure_context(unsigned long long device);

// Helper function to get memory allocation granularity
size_t get_memory_granularity(unsigned long long device) {
    // Define memory allocation properties
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

    // Get granularity
    size_t granularity;
    CUresult result = cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        std::cerr << "Error getting allocation granularity: " << error_str << std::endl;
        return 0;
    }
    
    return granularity;
}

// Helper function to format size in human-readable form
std::string format_size(size_t size_bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(size_bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
    return std::string(buffer);
}

// Structure to hold all the memory allocation information for one device
struct DeviceMemory {
    unsigned long long device;
    size_t size;
    size_t alignedSize;
    CUdeviceptr d_mem;
    CUmemGenericAllocationHandle** p_memHandle;
    unsigned long long* chunk_sizes;
    size_t num_chunks;
    bool allocated;

    DeviceMemory() : allocated(false) {}

    ~DeviceMemory() {
        if (p_memHandle) {
            for (size_t i = 0; i < num_chunks; i++) {
                if (p_memHandle[i]) {
                    free(p_memHandle[i]);
                }
            }
            free(p_memHandle);
        }
        if (chunk_sizes) {
            free(chunk_sizes);
        }
    }
};

// Allocate memory on a specific device
bool allocate_device_memory(DeviceMemory& mem, size_t size, size_t granularity, bool verify = true) {
    // Align the size
    mem.size = size;
    mem.alignedSize = ((size + granularity - 1) / granularity) * granularity;
    
    // Reserve memory address
    CUresult result = cuMemAddressReserve(&mem.d_mem, mem.alignedSize, granularity, 0, 0);
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        std::cerr << "Error reserving memory address for device " << mem.device 
                  << ": " << error_str << std::endl;
        return false;
    }
    
    // Define chunk sizes for ROCM (AMD) implementation
    // Use 128MB chunks for better management of large memory
    const size_t CHUNK_SIZE_MB = 128;
    size_t aligned_chunk_size = ((CHUNK_SIZE_MB * 1024 * 1024 + granularity - 1) / granularity) * granularity;
    mem.num_chunks = (mem.alignedSize + aligned_chunk_size - 1) / aligned_chunk_size;
    
    // Allocate arrays for handles and chunk sizes
    mem.p_memHandle = (CUmemGenericAllocationHandle**)malloc(mem.num_chunks * sizeof(CUmemGenericAllocationHandle*));
    mem.chunk_sizes = (unsigned long long*)malloc(mem.num_chunks * sizeof(unsigned long long));
    
    // Initialize handles and chunk sizes
    for (size_t i = 0; i < mem.num_chunks; i++) {
        mem.p_memHandle[i] = (CUmemGenericAllocationHandle*)malloc(sizeof(CUmemGenericAllocationHandle));
        mem.chunk_sizes[i] = (i == mem.num_chunks - 1) ? 
            (mem.alignedSize - (mem.num_chunks - 1) * aligned_chunk_size) : aligned_chunk_size;
    }
    
    // Call create_and_map
    create_and_map(mem.device, mem.alignedSize, mem.d_mem, mem.p_memHandle, mem.chunk_sizes, mem.num_chunks);
    
    // Verify memory is accessible (optional)
    if (verify) {
        // Test with a small amount of data (1MB)
        const size_t test_size = 1 * 1024 * 1024;
        int* h_data = new int[test_size / sizeof(int)];
        for (size_t i = 0; i < test_size / sizeof(int); i++) {
            h_data[i] = i & 0xFF;
        }
        
        hipError_t hip_result = hipMemcpy((void*)mem.d_mem, h_data, test_size, hipMemcpyHostToDevice);
        if (hip_result != hipSuccess) {
            std::cerr << "Error copying to device " << mem.device 
                      << " memory: " << hipGetErrorString(hip_result) << std::endl;
            delete[] h_data;
            return false;
        }
        
        int* h_result = new int[test_size / sizeof(int)];
        hip_result = hipMemcpy(h_result, (void*)mem.d_mem, test_size, hipMemcpyDeviceToHost);
        if (hip_result != hipSuccess) {
            std::cerr << "Error copying from device " << mem.device 
                      << " memory: " << hipGetErrorString(hip_result) << std::endl;
            delete[] h_data;
            delete[] h_result;
            return false;
        }
        
        bool data_correct = true;
        for (size_t i = 0; i < test_size / sizeof(int); i++) {
            if (h_data[i] != h_result[i]) {
                std::cerr << "Device " << mem.device << " data verification failed at index " 
                          << i << ": expected " << h_data[i] << ", got " << h_result[i] << std::endl;
                data_correct = false;
                break;
            }
        }
        
        delete[] h_data;
        delete[] h_result;
        
        if (!data_correct) {
            return false;
        }
    }
    
    mem.allocated = true;
    return true;
}

// Free memory on a specific device
bool free_device_memory(DeviceMemory& mem) {
    if (!mem.allocated) {
        return true;
    }
    
    unmap_and_release(mem.device, mem.alignedSize, mem.d_mem, mem.p_memHandle, mem.chunk_sizes, mem.num_chunks);
    
    // Free the address
    CUresult result = cuMemAddressFree(mem.d_mem, mem.alignedSize);
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        std::cerr << "Error freeing memory address for device " << mem.device 
                  << ": " << error_str << std::endl;
        return false;
    }
    
    mem.allocated = false;
    return true;
}

int main() {
    std::cout << "ROCM Memory Mapping Test - Simultaneous Allocation of 120GB on All Devices" << std::endl;
    
    // Initialize HIP
    hipError_t hip_result = hipInit(0);
    if (hip_result != hipSuccess) {
        std::cerr << "Failed to initialize HIP runtime: " << hipGetErrorString(hip_result) << std::endl;
        return 1;
    }
    
    // Get device count
    int deviceCount;
    hip_result = hipGetDeviceCount(&deviceCount);
    if (hip_result != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(hip_result) << std::endl;
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " HIP device(s)" << std::endl;
    
    // Limit to 8 devices maximum (0-7)
    int max_devices = std::min(deviceCount, 8);
    
    // Set allocation size to 120GB
    size_t allocation_size = 120ULL * 1024 * 1024 * 1024;
    
    // Array to store device memory information
    std::vector<DeviceMemory> device_memories(max_devices);
    
    // First pass: initialize contexts and get granularity for all devices
    std::vector<size_t> granularities(max_devices);
    for (int i = 0; i < max_devices; i++) {
        hipDeviceProp_t deviceProp;
        hip_result = hipGetDeviceProperties(&deviceProp, i);
        if (hip_result != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << i 
                     << ": " << hipGetErrorString(hip_result) << std::endl;
            continue;
        }
        
        std::cout << "Device " << i << ": " << deviceProp.name 
                  << " (Memory: " << format_size(deviceProp.totalGlobalMem) << ")" << std::endl;
        
        // Initialize context
        ensure_context(i);
        
        // Get granularity
        granularities[i] = get_memory_granularity(i);
        if (granularities[i] == 0) {
            std::cerr << "Failed to get memory granularity for device " << i << std::endl;
            continue;
        }
        
        // Initialize device memory structure
        device_memories[i].device = i;
    }
    
    std::cout << "\nSimultaneously allocating " << format_size(allocation_size) 
              << " on each device..." << std::endl;
    
    // Start timing
    auto alloc_start_time = std::chrono::high_resolution_clock::now();
    
    // Second pass: allocate memory on all devices
    for (int i = 0; i < max_devices; i++) {
        if (granularities[i] == 0) {
            continue;  // Skip devices with granularity errors
        }
        
        std::cout << "Allocating on device " << i << " (" << format_size(allocation_size) << ")..." << std::endl;
        if (!allocate_device_memory(device_memories[i], allocation_size, granularities[i], true)) {
            std::cerr << "Failed to allocate memory on device " << i << std::endl;
        } else {
            std::cout << "Successfully allocated " << format_size(allocation_size) 
                      << " on device " << i << std::endl;
        }
    }
    
    auto alloc_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> alloc_time = alloc_end_time - alloc_start_time;
    std::cout << "\nTotal allocation time for all devices: " << alloc_time.count() << " seconds" << std::endl;
    
    // Wait a moment to let the system stabilize
    std::cout << "\nGiving the system a moment to stabilize..." << std::endl;
    sleep(5);
    
    // Release memory on all devices
    std::cout << "\nSimultaneously releasing memory from all devices..." << std::endl;
    auto free_start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < max_devices; i++) {
        if (!device_memories[i].allocated) {
            continue;
        }
        
        std::cout << "Releasing memory from device " << i << "..." << std::endl;
        if (!free_device_memory(device_memories[i])) {
            std::cerr << "Failed to release memory from device " << i << std::endl;
        } else {
            std::cout << "Successfully released memory from device " << i << std::endl;
        }
    }
    
    auto free_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> free_time = free_end_time - free_start_time;
    std::cout << "\nTotal release time for all devices: " << free_time.count() << " seconds" << std::endl;
    
    // Overall timing
    std::chrono::duration<double> total_time = free_end_time - alloc_start_time;
    std::cout << "\nTotal test time: " << total_time.count() << " seconds" << std::endl;
    
    return 0;
} 