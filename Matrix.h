// Matrix.cuh
// 创建人;邱孝兵
//
// 矩阵定义（Matrix）
// 功能说明：定义了矩阵的数据结构和对矩阵的基本操作
//
// 修订历史：
// 2015年04月08日（邱孝兵）
//     初始版本。

#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct Matrix_st{
    size_t width;    // 矩阵宽度
    size_t height;   // 矩阵高度 
    float *matData;  // 矩阵数据。
} Matrix;

// 结构体：MatrixCuda（矩阵的 CUDA 相关数据）
// 该结构体定义了与 CUDA 相关的矩阵数据。该结构体通常在算法内部使用，上层用户通
// 常见不到也不需要了解此结构体。
typedef struct MatrixCuda_st {
    Matrix matMeta;    // 矩阵数据，保存了对应的矩阵逻辑数据。
    int deviceId;      // 当前数据所处的内存，如果数据在 GPU 的内存上，则
                       // deviceId 为对应设备的编号；若 deviceId < 0，则说明数
                       // 据存储在 Host 内存上。
    size_t pitchWords; // Padding 后矩阵每行数据所占内存的字数（4 字节），要求
                       // pitchBytes >= width
} MatrixCuda;

// 宏：MATRIX_CUDA
// 给定 Matrix 型指针，返回对应的 MatrixCuda 型指针。该宏通常在算法内部使用，用
// 来获取关于 CUDA 的矩阵数据。
#define MATRIX_CUDA(mat)                                                      \
    ((MatrixCuda *)((unsigned char *)mat -                                    \
                    (unsigned long)(&(((MatrixCuda *)NULL)->matMeta))))
// 类：MatrixBasicOp（矩阵基本操作）
// 继承自：无
// 该类包含了对于矩阵的基本操作，如矩阵的创建与销毁、矩阵的读取、矩阵在各地址空
// 间之间的拷贝等。要求所有的矩阵实例，都需要通过该类创建，否则，将会导致系统运
// 行的紊乱（需要保证每一个 Matrix 型的数据都有对应的 MatrixCuda 数据）。
class MatrixBasicOp {

public:

    // Host 静态方法：newMatrix（创建矩阵）
    // 创建一个新的矩阵实例，并通过参数 outmat 返回。
    static __host__ int newMatrix(Matrix **outmat);

    // Host 静态方法：deleteMatrix（销毁矩阵）
    // 销毁一个不再被使用的矩阵实例。
    static __host__ int deleteMatrix(Matrix *inmat);

    // Host 静态方法：makeAtCurrentDevice（在当前 Device 内存中构建数据）
    // 针对空矩阵，在当前 Device 内存中为其申请一段指定的大小的空间，这段空间中
    // 的数据是未被赋值的混乱数据。如果不是空矩阵，则该方法会报错。
    static __host__ int makeAtCurrentDevice(Matrix *mat, 
                                    size_t width,size_t height);

    // Host 静态方法：makeAtHost（在 Host 内存中构建数据）
    // 针对空矩阵，在 Host 内存中为其申请一段指定的大小的空间，这段空间中的数据
    // 是未被赋值的混乱数据。如果不是空矩阵，则该方法会报错。
    static __host__ int makeAtHost(Matrix *mat,size_t width,size_t height);
              
    // Host 静态方法：readFromFile（从CSV文件读取矩阵）
    // 从指定的文件路径中读取一个矩阵。该矩阵必须事先经过 newMatrix 创建。读取
    // 后矩阵默认放于 Host 内存中，这里要求文件为纯文本格式，且矩阵数据之间使用
    // ，分割——类似于CSV格式
    static __host__ int readFromFile(const char *filepath, unsigned int width, unsigned int height, Matrix *outmat);                                  
    

    // Host 静态方法：writeToFile（将矩阵写入CSV文件）
    // 矩阵写入到指定的文件中。该矩阵中必须包含有数据,写入的格式是按照CSV格式进行写入的
    static __host__ int writeToFile(const char *filepath, Matrix *inmat);

    // Host 静态方法：copyToCurrentDevice（将矩阵拷贝到当前 Device 内存上）
    // 这是一个 In-Place 形式的拷贝。如果矩阵数据本来就在当前的 Device 上，则该
    // 函数不会进行任何操作，直接返回。如果矩阵数据不在当前 Device 上，则会将数
    // 据拷贝到当前 Device 上，并更新 matData 指针。原来的数据将会被释放。
    static __host__ int copyToCurrentDevice(Matrix *mat);

    // Host 静态方法：copyToCurrentDevice（将矩阵拷贝到当前 Device 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srcmat 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstmat，且数据是存储于当前的 Device 上的。如果
    // dstmat 中原来存在有数据，且原来的数据同新的数据尺寸相同，也存放在当前
    // Device 上，则覆盖原内容，不重新分配空间；否则原数据将会被释放，并重新申
    // 请空间。
    static __host__ int copyToCurrentDevice(Matrix *srcmat, Matrix *dstmat);

    // Host 静态方法：copyToHost（将矩阵拷贝到 Host 内存上）
    // 这是一个 In-Place 形式的拷贝。如果矩阵数据本来就在 Host 上，则该函数不会
    // 进行任何操作，直接返回。如果矩阵数据不在 Host 上，则会将数据拷贝到 Host
    // 上，并更新 matData 指针。原来的数据将会被释放。
    static __host__ int copyToHost(Matrix *mat);

    // Host 静态方法：copyToHost（将矩阵拷贝到 Host 内存上）
    // 这是一个 Out-Place 形式的拷贝。无论 srcmat 位于哪一个内存空间中，都会得
    // 到一个和其内容完全一致的 dstmat，且数据是存储于 Host 上的。如果 dstmat
    // 中原来存在有数据，如果原来的数据同新的数据尺寸相同，且也存放在 Host 上，
    // 则覆盖原内容，但不重新分配空间；否则原数据将会被释放，并重新申请空间。
    static __host__ int copyToHost(Matrix *srcmat, Matrix *dstmat);       
};


#endif