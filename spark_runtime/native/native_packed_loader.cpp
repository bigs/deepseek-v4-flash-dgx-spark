#include <torch/extension.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>

#include <pybind11/stl.h>

namespace {

int64_t read_into(int64_t fd, torch::Tensor destination, int64_t size, int64_t offset) {
    TORCH_CHECK(fd >= 0, "fd must be non-negative");
    TORCH_CHECK(size >= 0, "size must be non-negative");
    TORCH_CHECK(offset >= 0, "offset must be non-negative");
    TORCH_CHECK(destination.device().is_cpu(), "destination must be a CPU tensor");
    TORCH_CHECK(destination.dtype() == torch::kUInt8, "destination must be uint8");
    TORCH_CHECK(destination.is_contiguous(), "destination must be contiguous");
    TORCH_CHECK(destination.numel() >= size, "destination tensor is too small");

    auto* output = static_cast<char*>(destination.data_ptr());
    int64_t completed = 0;

    pybind11::gil_scoped_release release;
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

int64_t copy_storage_to_tensors(
    torch::Tensor storage,
    std::vector<torch::Tensor> targets,
    std::vector<int64_t> offsets,
    bool non_blocking
) {
    TORCH_CHECK(storage.device().is_cpu(), "storage must be a CPU tensor");
    TORCH_CHECK(storage.dtype() == torch::kUInt8, "storage must be uint8");
    TORCH_CHECK(storage.is_contiguous(), "storage must be contiguous");
    TORCH_CHECK(targets.size() == offsets.size(), "targets and offsets length mismatch");

    auto* base = static_cast<char*>(storage.data_ptr());
    const int64_t storage_bytes = storage.numel();
    int64_t copied_bytes = 0;

    pybind11::gil_scoped_release release;
    for (size_t index = 0; index < targets.size(); ++index) {
        auto target = targets[index];
        const int64_t offset = offsets[index];
        TORCH_CHECK(offset >= 0, "offset must be non-negative");
        TORCH_CHECK(target.is_contiguous(), "target must be contiguous");

        const int64_t bytes = target.numel() * target.element_size();
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
}
