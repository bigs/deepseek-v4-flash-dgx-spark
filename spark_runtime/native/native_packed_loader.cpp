#include <torch/extension.h>

#ifdef DEEPSEEK_SPARK_WITH_CUDA
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>
#include <unistd.h>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

int64_t read_into_unlocked(int64_t fd, torch::Tensor destination, int64_t size, int64_t offset) {
    TORCH_CHECK(fd >= 0, "fd must be non-negative");
    TORCH_CHECK(size >= 0, "size must be non-negative");
    TORCH_CHECK(offset >= 0, "offset must be non-negative");
    TORCH_CHECK(destination.device().is_cpu(), "destination must be a CPU tensor");
    TORCH_CHECK(destination.dtype() == torch::kUInt8, "destination must be uint8");
    TORCH_CHECK(destination.is_contiguous(), "destination must be contiguous");
    TORCH_CHECK(destination.numel() >= size, "destination tensor is too small");

    auto* output = static_cast<char*>(destination.data_ptr());
    int64_t completed = 0;

    while (completed < size) {
        const auto remaining = static_cast<size_t>(size - completed);
        const auto current_offset = static_cast<off_t>(offset + completed);
        const ssize_t result = ::pread(
            static_cast<int>(fd),
            output + completed,
            remaining,
            current_offset
        );
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::system_error(
                errno,
                std::generic_category(),
                "pread failed"
            );
        }
        if (result == 0) {
            break;
        }
        completed += static_cast<int64_t>(result);
    }
    return completed;
}

int64_t read_into(int64_t fd, torch::Tensor destination, int64_t size, int64_t offset) {
    py::gil_scoped_release release;
    return read_into_unlocked(fd, destination, size, offset);
}

int64_t copy_storage_to_tensors_unlocked(
    torch::Tensor storage,
    const std::vector<torch::Tensor>& targets,
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>* expected_sizes,
    bool non_blocking
) {
    TORCH_CHECK(storage.device().is_cpu(), "storage must be a CPU tensor");
    TORCH_CHECK(storage.dtype() == torch::kUInt8, "storage must be uint8");
    TORCH_CHECK(storage.is_contiguous(), "storage must be contiguous");
    TORCH_CHECK(targets.size() == offsets.size(), "targets and offsets length mismatch");
    if (expected_sizes != nullptr) {
        TORCH_CHECK(
            targets.size() == expected_sizes->size(),
            "targets and expected sizes length mismatch"
        );
    }

    auto* base = static_cast<char*>(storage.data_ptr());
    const int64_t storage_bytes = storage.numel();
    int64_t copied_bytes = 0;

    for (size_t index = 0; index < targets.size(); ++index) {
        auto target = targets[index];
        const int64_t offset = offsets[index];
        TORCH_CHECK(offset >= 0, "offset must be non-negative");
        TORCH_CHECK(target.is_contiguous(), "target must be contiguous");

        const int64_t bytes = target.numel() * target.element_size();
        if (expected_sizes != nullptr) {
            TORCH_CHECK(
                bytes == expected_sizes->at(index),
                "target size does not match expert copy plan"
            );
        }
        TORCH_CHECK(
            offset + bytes <= storage_bytes,
            "source slice exceeds storage tensor"
        );

        auto options = torch::TensorOptions()
            .dtype(target.scalar_type())
            .device(torch::kCPU);
        auto sizes = target.sizes().vec();
        auto source = torch::from_blob(base + offset, sizes, options);
        target.copy_(source, non_blocking);
        copied_bytes += bytes;
    }
    return copied_bytes;
}

#ifdef DEEPSEEK_SPARK_WITH_CUDA
int64_t copy_storage_to_tensors_cuda_unlocked(
    torch::Tensor storage,
    const std::vector<torch::Tensor>& targets,
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>* expected_sizes
) {
    TORCH_CHECK(storage.device().is_cpu(), "storage must be a CPU tensor");
    TORCH_CHECK(storage.dtype() == torch::kUInt8, "storage must be uint8");
    TORCH_CHECK(storage.is_contiguous(), "storage must be contiguous");
    TORCH_CHECK(targets.size() == offsets.size(), "targets and offsets length mismatch");
    if (expected_sizes != nullptr) {
        TORCH_CHECK(
            targets.size() == expected_sizes->size(),
            "targets and expected sizes length mismatch"
        );
    }

    auto* base = static_cast<char*>(storage.data_ptr());
    const int64_t storage_bytes = storage.numel();
    int64_t copied_bytes = 0;

    for (size_t index = 0; index < targets.size(); ++index) {
        auto target = targets[index];
        const int64_t offset = offsets[index];
        TORCH_CHECK(offset >= 0, "offset must be non-negative");
        TORCH_CHECK(target.is_cuda(), "CUDA memcpy target must be a CUDA tensor");
        TORCH_CHECK(target.is_contiguous(), "target must be contiguous");

        const int64_t bytes = target.numel() * target.element_size();
        if (expected_sizes != nullptr) {
            TORCH_CHECK(
                bytes == expected_sizes->at(index),
                "target size does not match expert copy plan"
            );
        }
        TORCH_CHECK(
            offset + bytes <= storage_bytes,
            "source slice exceeds storage tensor"
        );

        const auto stream = c10::cuda::getCurrentCUDAStream(target.get_device());
        C10_CUDA_CHECK(cudaMemcpyAsync(
            target.data_ptr(),
            base + offset,
            static_cast<size_t>(bytes),
            cudaMemcpyHostToDevice,
            stream.stream()
        ));
        copied_bytes += bytes;
    }
    return copied_bytes;
}
#endif

int64_t copy_storage_to_tensors(
    torch::Tensor storage,
    std::vector<torch::Tensor> targets,
    std::vector<int64_t> offsets,
    bool non_blocking
) {
    py::gil_scoped_release release;
    return copy_storage_to_tensors_unlocked(
        storage,
        targets,
        offsets,
        nullptr,
        non_blocking
    );
}

double seconds_between(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end
) {
    return std::chrono::duration<double>(end - start).count();
}

class ExpertCopyPlan {
  public:
    ExpertCopyPlan(std::vector<int64_t> offsets, std::vector<int64_t> sizes)
        : offsets_(std::move(offsets)), sizes_(std::move(sizes)) {
        TORCH_CHECK(offsets_.size() == sizes_.size(), "offsets and sizes length mismatch");
        for (size_t index = 0; index < offsets_.size(); ++index) {
            TORCH_CHECK(offsets_[index] >= 0, "copy-plan offset must be non-negative");
            TORCH_CHECK(sizes_[index] >= 0, "copy-plan size must be non-negative");
        }
    }

    int64_t count() const {
        return static_cast<int64_t>(offsets_.size());
    }

    int64_t copy_storage_to_tensors(
        torch::Tensor storage,
        std::vector<torch::Tensor> targets,
        bool non_blocking
    ) const {
        py::gil_scoped_release release;
        return copy_storage_to_tensors_unlocked(
            storage,
            targets,
            offsets_,
            &sizes_,
            non_blocking
        );
    }

    int64_t copy_storage_to_tensors_cuda(
        torch::Tensor storage,
        std::vector<torch::Tensor> targets
    ) const {
#ifdef DEEPSEEK_SPARK_WITH_CUDA
        py::gil_scoped_release release;
        return copy_storage_to_tensors_cuda_unlocked(
            storage,
            targets,
            offsets_,
            &sizes_
        );
#else
        TORCH_CHECK(false, "native extension was not built with CUDA support");
        return 0;
#endif
    }

    std::tuple<int64_t, int64_t, double, double> read_into_and_copy(
        int64_t fd,
        torch::Tensor staging,
        int64_t read_size,
        int64_t read_offset,
        std::vector<torch::Tensor> targets,
        bool non_blocking
    ) const {
        int64_t read_bytes = 0;
        int64_t copied_bytes = 0;
        double read_seconds = 0.0;
        double copy_seconds = 0.0;

        {
            py::gil_scoped_release release;
            const auto read_start = std::chrono::steady_clock::now();
            read_bytes = read_into_unlocked(fd, staging, read_size, read_offset);
            const auto read_end = std::chrono::steady_clock::now();
            TORCH_CHECK(read_bytes == read_size, "short packed expert read");
            copied_bytes = copy_storage_to_tensors_unlocked(
                staging.slice(0, 0, read_size),
                targets,
                offsets_,
                &sizes_,
                non_blocking
            );
            const auto copy_end = std::chrono::steady_clock::now();
            read_seconds = seconds_between(read_start, read_end);
            copy_seconds = seconds_between(read_end, copy_end);
        }

        return std::make_tuple(read_bytes, copied_bytes, read_seconds, copy_seconds);
    }

    std::tuple<int64_t, int64_t, double, double> read_into_and_copy_cuda(
        int64_t fd,
        torch::Tensor staging,
        int64_t read_size,
        int64_t read_offset,
        std::vector<torch::Tensor> targets
    ) const {
#ifdef DEEPSEEK_SPARK_WITH_CUDA
        int64_t read_bytes = 0;
        int64_t copied_bytes = 0;
        double read_seconds = 0.0;
        double copy_seconds = 0.0;

        {
            py::gil_scoped_release release;
            const auto read_start = std::chrono::steady_clock::now();
            read_bytes = read_into_unlocked(fd, staging, read_size, read_offset);
            const auto read_end = std::chrono::steady_clock::now();
            TORCH_CHECK(read_bytes == read_size, "short packed expert read");
            copied_bytes = copy_storage_to_tensors_cuda_unlocked(
                staging.slice(0, 0, read_size),
                targets,
                offsets_,
                &sizes_
            );
            const auto copy_end = std::chrono::steady_clock::now();
            read_seconds = seconds_between(read_start, read_end);
            copy_seconds = seconds_between(read_end, copy_end);
        }

        return std::make_tuple(read_bytes, copied_bytes, read_seconds, copy_seconds);
#else
        TORCH_CHECK(false, "native extension was not built with CUDA support");
        return std::make_tuple(0, 0, 0.0, 0.0);
#endif
    }

  private:
    std::vector<int64_t> offsets_;
    std::vector<int64_t> sizes_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "read_into",
        &read_into,
        "Read bytes from an fd and offset directly into a contiguous CPU uint8 tensor"
    );
    module.def(
        "copy_storage_to_tensors",
        &copy_storage_to_tensors,
        "Copy packed CPU storage slices into destination tensors"
    );
    py::class_<ExpertCopyPlan>(module, "ExpertCopyPlan")
        .def(py::init<std::vector<int64_t>, std::vector<int64_t>>())
        .def_property_readonly("count", &ExpertCopyPlan::count)
        .def(
            "copy_storage_to_tensors",
            &ExpertCopyPlan::copy_storage_to_tensors,
            "Copy packed storage slices into tensors using a precomputed plan"
        )
        .def(
            "copy_storage_to_tensors_cuda",
            &ExpertCopyPlan::copy_storage_to_tensors_cuda,
            "Copy packed storage slices into CUDA tensors with cudaMemcpyAsync"
        )
        .def(
            "read_into_and_copy",
            &ExpertCopyPlan::read_into_and_copy,
            "Read a packed expert block into staging and copy planned slices into tensors"
        )
        .def(
            "read_into_and_copy_cuda",
            &ExpertCopyPlan::read_into_and_copy_cuda,
            "Read a packed expert block and cudaMemcpyAsync planned slices into CUDA tensors"
        );
}
