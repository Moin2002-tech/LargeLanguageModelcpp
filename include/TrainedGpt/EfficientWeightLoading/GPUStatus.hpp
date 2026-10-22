//
// Created by moinshaikh on 10/17/26.
//
#pragma once
#ifndef LARGELANGUAGEMODELCPP_GPUSTATUS_HPP
#include<torch/torch.h>

#include "c10/cuda/CUDACachingAllocator.h"

namespace gpuStatus
{
    void start_memory_tracking()
    {
        if (torch::cuda::is_available())
        {
            unsigned int device_id = torch::cuda::device_count();
            c10::cuda::CUDACachingAllocator::resetPeakStats(device_id);
        }
        else
        {
            std::cout<<"cuda is not available"<<"\n";
        }
    }

    void printMemoryStatus()
    {
        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        if (device.is_cuda())
            {
            c10::cuda::CUDACachingAllocator::emptyCache();
            torch::cuda::synchronize();

            // Diagnostic only — never let it crash the benchmark.
            try {
                int device_id = c10::cuda::current_device();  // actual device index

                c10::cuda::CUDACachingAllocator::DeviceStats stats =
                    c10::cuda::CUDACachingAllocator::getDeviceStats(device_id);

                int64_t allocated_bytes = stats.allocated_bytes[0].current;
                int64_t cached_bytes    = stats.reserved_bytes[0].current;

                std::cout << "    [CUDA allocated before model: "
                          << allocated_bytes << " bytes | "
                          << cached_bytes << " bytes cached]\n";
            } catch (const std::exception&) {
                // ignore — diagnostic only
            }
        }
    }
    void cleanup_cuda_memory(int device) {
        // 1. torch.cuda.empty_cache()
        //    (Python's `gc.collect()` has no equivalent in C++.)
        c10::cuda::CUDACachingAllocator::emptyCache();

        // 2. time.sleep(3)  -- buffer time to allow memory to clear
        std::this_thread::sleep_for(std::chrono::seconds(3));

        // 3. torch.cuda.reset_peak_memory_stats(device)
        c10::cuda::CUDACachingAllocator::resetPeakStats(device);

        // 4. torch.cuda.max_memory_allocated(device)
        const auto device_stats =
            c10::cuda::CUDACachingAllocator::getDeviceStats(device);
        const std::size_t peak_bytes =
            device_stats.allocated_bytes
                [static_cast<std::size_t>(c10::CachingAllocator::StatType::AGGREGATE)]
                .peak;

        const double max_memory_allocated_gb =
            static_cast<double>(peak_bytes) / (1024.0 * 1024.0 * 1024.0);

        // Mirror Python's f"{max_memory_allocated:.1f} GB"
        std::cout << std::fixed << std::setprecision(1)
                  << "Maximum GPU memory allocated: " << max_memory_allocated_gb
                  << " GB" << std::endl;
    }

}

#endif //LARGELANGUAGEMODELCPP_GPUSTATUS_HPP
