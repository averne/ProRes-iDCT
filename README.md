# Vulkan-iDCT

Compute shader-based implementation of the Apple ProRes inverse DCT.  
This was written in fulfillment of the qualification task for the project "ProRes Vulkan Decoder", as part of the FFmpeg/Google Summer of Code 2025 event.

## Building/Running

[glslang](https://github.com/KhronosGroup/glslang) is necessary to compile the compute shader.

1. Setup the project: `meson setup build`.
2. Compile the code: `meson compile -C build`.
3. Run the program: `build/prores-idct`. Note that your current working directory must be the base of this repository, otherwise the shader binary will not be found.

The program will generate randomized DCT coefficients, and apply the inverse transform (both using the GPU and a reference software implementation).  
The reconstructed data is then validated against its reference, using criteria outlined in the SMPTE specification (Apple ProRes Bitstream Syntax and Decoding Process, annex A "IDCT Implementation Accuracy Qualification").
