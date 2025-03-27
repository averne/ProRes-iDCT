project('Vulkan-iDCT', 'cpp',
    default_options: ['cpp_std=c++23']
)

if get_option('buildtype') == 'debug'
    add_global_arguments('-DDEBUG', language: 'cpp')
endif

glsllang = find_program('glslangValidator')
shader = custom_target('shader',
    command : [glsllang, '-V', '@INPUT@', '-o',  '@OUTPUT@'],
    input : 'src/prores-idct.comp',
    output : '@BASENAME@.spv',
)

executable('prores-idct', 'src/main.cpp',
    dependencies: [dependency('vulkan', required: true), declare_dependency(sources: shader)],
)
