#include <cstdint>
#include <numeric>

#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.hpp>

#include "util.hpp"

std::array layers = {
#ifdef DEBUG
    "VK_LAYER_KHRONOS_validation",
#endif
};

int main(int argc, char **argv) {
    // Create device and initialize objects
    auto app_info             = vk::ApplicationInfo("ProRes-iDCT", 0, "", 0, VK_API_VERSION_1_2);
    auto instance_create_info = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app_info, layers);

    auto instance = VK_CHECK_RV(vk::createInstanceUnique(instance_create_info));
    auto physdev  = VK_CHECK_RV(instance->enumeratePhysicalDevices()).front();

    auto dev_props = physdev.getProperties();
    std::println("Starting on device {}, api {}.{}.{}, driver {}.{}.{}", dev_props.deviceName.data(),
                 VK_VERSION_MAJOR(dev_props.apiVersion),    VK_VERSION_MINOR(dev_props.apiVersion),    VK_VERSION_PATCH(dev_props.apiVersion),
                 VK_VERSION_MAJOR(dev_props.driverVersion), VK_VERSION_MINOR(dev_props.driverVersion), VK_VERSION_PATCH(dev_props.driverVersion));

    auto queue_idx = util::find_in_family(physdev.getQueueFamilyProperties(), [](auto &&e) {
        return !!(e.queueFlags & (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer));
    });

    auto mem_props = physdev.getMemoryProperties();
    auto host_buf_idx = util::find_in_family(mem_props.memoryTypes, [](auto &&e) {
        return !!(e.propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent);
    });

    auto dev_buf_idx = util::find_in_family(mem_props.memoryTypes, [](auto &&e) {
        return !!(e.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    });

    assert(queue_idx >= 0 && host_buf_idx >= 0 && dev_buf_idx >= 0);

    std::array queue_prios = { 1.0f };
    auto queue_create_info = vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queue_idx, queue_prios);
    auto dev_features    = vk::PhysicalDeviceFeatures().setShaderFloat64(VK_TRUE);
    auto dev_create_info = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), queue_create_info, nullptr, nullptr, &dev_features);

    auto dev   = VK_CHECK_RV(physdev.createDeviceUnique(dev_create_info));
    auto queue = dev->getQueue(queue_idx, 0);

    auto command_pool = VK_CHECK_RV(dev->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                                                           queue_idx)));
    auto cmdbuf       = VK_CHECK_RV(dev->allocateCommandBuffers(vk::CommandBufferAllocateInfo(*command_pool, vk::CommandBufferLevel::ePrimary, 1))).front();

    auto fence = VK_CHECK_RV(dev->createFenceUnique(vk::FenceCreateInfo()));

    // Create memory buffers
    constexpr auto buf_size = 8 * 8 * sizeof(double);

    auto inbuf  = VK_CHECK_RV(dev->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), buf_size,
                                                                           vk::BufferUsageFlagBits::eStorageBuffer |
                                                                           vk::BufferUsageFlagBits::eTransferDst)));
    auto outbuf = VK_CHECK_RV(dev->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), buf_size,
                                                                           vk::BufferUsageFlagBits::eStorageBuffer |
                                                                           vk::BufferUsageFlagBits::eTransferSrc   |
                                                                           vk::BufferUsageFlagBits::eTransferDst)));

    auto inmem  = VK_CHECK_RV(dev->allocateMemoryUnique(vk::MemoryAllocateInfo(dev->getBufferMemoryRequirements(*inbuf ).size,
                                                                               host_buf_idx)));
    auto outmem = VK_CHECK_RV(dev->allocateMemoryUnique(vk::MemoryAllocateInfo(dev->getBufferMemoryRequirements(*outbuf).size,
                                                                               dev_buf_idx )));

    VK_CHECK(dev->bindBufferMemory(*inbuf,  *inmem,  0));
    VK_CHECK(dev->bindBufferMemory(*outbuf, *outmem, 0));

    auto *addr = VK_CHECK_RV(dev->mapMemory(*inmem, 0, buf_size));
    SCOPEGUARD([&] { dev->unmapMemory(*inmem); });

    std::ranges::iota(std::span(static_cast<double *>(addr), 16), 1);

    // Load shader code
    std::vector<std::uint32_t> shader_bin;
    util::read_whole_file(shader_bin, "build/prores-idct.spv");
    if (shader_bin.empty())
        return -1;

    auto shader = VK_CHECK_RV(dev->createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
                                                                                       shader_bin.size() * sizeof(decltype(shader_bin)::value_type),
                                                                                       shader_bin.data())));

    // Set up descriptors
    auto desc_sizes = vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2);
    auto desc_pool = VK_CHECK_RV(dev->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                                                                              1, desc_sizes)));

    std::array desc_bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)
    };
    auto desc_layout = VK_CHECK_RV(dev->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), desc_bindings)));
    auto desc_sets = VK_CHECK_RV(dev->allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(*desc_pool, *desc_layout)));
    auto descriptors = std::move(desc_sets.front());

	auto in_info = vk::DescriptorBufferInfo(*inbuf, 0, buf_size), out_info = vk::DescriptorBufferInfo(*outbuf, 0, buf_size);
    dev->updateDescriptorSets({
        {*descriptors, 0, 0, vk::DescriptorType::eStorageBuffer, nullptr, in_info },
        {*descriptors, 1, 0, vk::DescriptorType::eStorageBuffer, nullptr, out_info},
    }, nullptr);

    // Set up pipeline
    auto pipeline_layout = VK_CHECK_RV(dev->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *desc_layout)));

    auto pipeline_create_info = vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                                              vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                                                vk::ShaderStageFlagBits::eCompute, *shader, "main"),
                                                              *pipeline_layout);
    auto pipeline = VK_CHECK_RV(dev->createComputePipelineUnique(nullptr, pipeline_create_info));

    // Finally, write commands and kickoff
    vk::BufferMemoryBarrier transfer_barrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                                             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *outbuf, 0, buf_size);

    VK_CHECK(cmdbuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipeline_layout, 0, *descriptors, nullptr);
    cmdbuf.dispatch(1, 1, 1);
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                           vk::DependencyFlagBits::eByRegion, nullptr, transfer_barrier, nullptr);
    cmdbuf.copyBuffer(*outbuf, *inbuf, vk::BufferCopy(0, 0, buf_size));
    VK_CHECK(cmdbuf.end());

    VK_CHECK(queue.submit(vk::SubmitInfo(nullptr, nullptr, cmdbuf), *fence));
    VK_CHECK(dev->waitForFences(*fence, VK_TRUE, UINT64_MAX));

    auto *p = static_cast<double *>(addr);
    std::println("{:5.1f} {:5.1f} {:5.1f} {:5.1f}", p[ 0], p[ 1], p[ 2], p[ 3]);
    std::println("{:5.1f} {:5.1f} {:5.1f} {:5.1f}", p[ 4], p[ 5], p[ 6], p[ 7]);
    std::println("{:5.1f} {:5.1f} {:5.1f} {:5.1f}", p[ 8], p[ 9], p[10], p[11]);
    std::println("{:5.1f} {:5.1f} {:5.1f} {:5.1f}", p[12], p[13], p[14], p[15]);

    return 0;
}
