# 2025并行处理课程大作业

## 构建项目

使用 `./make-debug.sh` 或 `./make-release.sh` 构建 Debug 或 Release 版本的程序。

## 测试数据生成和导入

生成一个 $n$ 行 $m$ 列的随机矩阵，矩阵的序列号为 $id$ 。

```bash
python ./test/data/gen.py <n> <m> <id>
```

在程序中，使用以下方法读取生成的数据：

```cpp
// 读取一个 4000 x 160000 的矩阵，矩阵的序列号为 1 。
Matrix a = Matrix::from_file("./test/data/4000x16000_1.txt");
```

## 自动化测试

### 功能测试

使用 `./test-func.sh` 运行功能性测试集。功能性测试集不开启任何编译优化，仅测试各部分功能是否正确实现、代码中是否有未定义的行为。

此部分代码会读取 `test/data` 目录下的数据，测试前请先运行数据生成脚本。

目前包含的功能测试：
- `func_kernel_8x16x16000`: 测试针对 $8 \times 16 \times 16000$ 规模的矩阵乘法的优化是否正确。
  > 运行本测试需要先生成以下文件：
  > - `test/data/8x16_1.txt`
  > - `test/data/16x16000_1.txt`
  
- `func_kernel_32x16000x16`: 测试针对 $32 \times 16000 \times 16$ 规模的矩阵乘法的优化是否正确。
  > 运行本测试需要先生成以下文件：
  > - `test/data/32x16000_1.txt`
  > - `test/data/16000x16_1.txt`
  
- `func_kernel_144x144x144`: 测试针对 $144 \times 144 \times 144$ 规模的矩阵乘法的优化是否正确。
  > 运行本测试需要先生成以下文件：
  > - `test/data/144x144_1.txt`
  
- `func_kernel_4000x16000x128`: 测试针对 $4000 \times 16000 \times 128$ 规模的矩阵乘法的优化是否正确。
  > 运行本测试需要先生成以下文件：
  > - `test/data/4000x16000_1.txt`
  > - `test/data/16000x128_1.txt`
  >
  > 运行本测试可能需要较长的时间。
  
- `func_matmul_block`: 测试分块矩阵乘法的实现是否正确。
  > 运行本测试需要先生成以下文件：
  > - `test/data/16x16_1.txt`
  > - `test/data/200x400_1.txt`
  > - `test/data/400x300_1.txt`
  
- `func_pack`: 测试data packing的实现是否正确。

  > 运行本测试不需要任何文件。

### 性能测试

使用 `./test-perf.sh` 运行性能测试集。性能测试集开启 `-O3` 优化，测试矩阵乘法优化的效果。

测试程序会将结果输出到标准输出。默认情况下， `ctest` 将它们重定向到了 `build-release/Testing/Temporary/LastTest.log` ，你可以在这里查看性能评测的结果。

性能测试不读取任何文件。

目前包含的性能测试：
- `perf_matmul_4kx16kx128`: $4000 \times 16000 \times 128$ 规模的矩阵乘法内核。
- `perf_matmul_8x16x16k`: $8 \times 16 \times 16000$ 规模的矩阵乘法内核。
- `perf_matmul_32x16kx16`: $32 \times 16000 \times 16$ 规模的矩阵乘法内核。
- `perf_matmul_144x144x144`: $144 \times 144 \times 144$ 规模的矩阵乘法内核。
- `perf_pack_4kx16kx128`: 对 $4000 \times 16000 \times 128$ 规模的矩阵进行packing。
- `perf_pack_144x144x144`: 对 $144 \times 144 \times 144$ 规模的矩阵进行packing。
- `perf_submatmul_4kx16kx128`: 对 $4000 \times 16000 \times 128$ 规模的计算矩阵乘法（省略packing）。
