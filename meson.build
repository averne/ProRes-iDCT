project('Vulkan-iDCT', 'cpp',
    default_options: ['cpp_std=c++23']
)

if get_option('buildtype') == 'debug'
    add_global_arguments('-DDEBUG', language: 'cpp')
endif

vk_dep = dependency('vulkan', required: true)

executable('prores-idct', 'src/main.cpp',
    dependencies: vk_dep,
)
