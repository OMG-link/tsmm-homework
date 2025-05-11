# 2025并行处理课程大作业

## 构建项目

使用 `./make-debug.sh` 或 `./make-release.sh` 构建 Debug 或 Release 版本的程序。

使用 `./run.sh` 运行构建好的程序。

## 测试数据生成和导入

生成一个 $n$ 行 $m$ 列的矩阵，矩阵的序列号为 $id$ 。

```bash
python ./tests/data/gen.py <n> <m> <id>
```

在程序中，使用以下方法读取生成的数据：

```cpp
// 读取一个 4000 x 160000 的矩阵，矩阵的序列号为 1 。
Matrix a = Matrix::from_file("./tests/data/4000x16000_1.txt");
```
