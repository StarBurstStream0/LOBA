import os
import glob
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 定义 convex_ext 的编译配置
def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

# 主 setup 函数
setup(
    name="convex",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=[
            make_cuda_ext(
                name='convex_ext',
                module='',
                sources=[
                    './src/convex_cpu.cpp',
                    './src/convex_ext.cpp'
                ],
                sources_cuda=['./src/convex_cuda.cu'])
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=1.7"],
)