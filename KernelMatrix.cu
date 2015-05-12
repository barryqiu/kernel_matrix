// KernelMatrix.cu
// 实现计算样例矩阵的核矩阵

#include "KernelMatrix.h"
#include "math.h"
#include <iostream>
using namespace std;

// 宏：DEF_BLOCK_X 和 DEF_BLOCK_Y
// 定义了默认的线程块的尺寸。
#define DEF_BLOCK_X  32 
#define DEF_BLOCK_Y   8 

// 核函数： _calcKernelMatrixKer（计算样例矩阵的核矩阵）
// 根据给定的样例矩阵，和操作类，计算其对应的核矩阵
static __global__ void 
_calcKernelMatrixKer (
        MatrixCuda inMatrix,        // 输入的样例矩阵
        KernelMatrix km,            // 操作类
        MatrixCuda outKernelMatrix  // 输出的核矩阵
);

// 核函数： _calcKernelMatrixKer（计算样例矩阵的核矩阵）
static __global__ void _calcKernelMatrixKer (
        MatrixCuda inMatrix, KernelMatrix km, MatrixCuda outKernelMatrix)
{
    // 计算线程对应的输出点的位置，其中 c 和 r 分别表示输出核矩阵的横纵坐标位置
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查像素点是否越界
    if (c >= outKernelMatrix.matMeta.width || r >= outKernelMatrix.matMeta.height)
        return;

    // 由于核矩阵是一个对称阵，所以只需要计算其一半的点就可以，在这儿仅仅计算上半部分的点
    // 即仅仅计算 c >= r 的点，如果 c < r 直接返回
    if (c < r)
    {
        outKernelMatrix.matMeta.matData[r * outKernelMatrix.pitchWords + c] = 0;
        return;
    }

    // 计算两个样例的差的2-范数的平方
    float sum = 0;

    for(int i = 0; i < inMatrix.matMeta.width; i++)
    {
        // 计算两个样例对应点的差
        float difference = inMatrix.matMeta.matData[c * inMatrix.pitchWords + i] - 
               inMatrix.matMeta.matData[r * inMatrix.pitchWords + i];
        
        // 计算平方并且求和
        sum += difference * difference;               
    }

    // 计算 RBF 的值
    float rbf = expf(-0.5 * sum / (km.getSigma() * km.getSigma()));

    // 将结果写入输出矩阵
    outKernelMatrix.matMeta.matData[r * outKernelMatrix.pitchWords + c] = rbf;
}

// Host 成员函数 calcKernelMatrix（计算核矩阵）
__host__ int KernelMatrix::calcKernelMatrix(Matrix *inSample, Matrix *outKernelMatrix)
{
    // 判断输入指针是否为 NULL， 如果为 NULL 直接报错返回
    if (NULL == inSample || NULL == outKernelMatrix)
        return NULL_POINTER;

    // 判断输入矩阵尺寸是否正确， 如果不正确直接报错返回
    if (inSample->height < 1 || inSample->width < 1 || 
        outKernelMatrix->height < 1 || outKernelMatrix->width < 1 || 
        inSample->height != outKernelMatrix->width || 
        inSample->height != outKernelMatrix->height)
        return INVALID_DATA;

    // 将输入输出矩阵拷贝到当前设备、
    //MatrixBasicOp::copyToCurrentDevice(inSample);
    //MatrixBasicOp::copyToCurrentDevice(outKernelMatrix);

    MatrixCuda *inSampleCud, *outKernelMatrixCud; // 输入输出设备端矩阵

    // 通过预定义的宏将 Matrix 指针转化为 MatrixCuda 类型的指针
    inSampleCud = MATRIX_CUDA(inSample);
    outKernelMatrixCud = MATRIX_CUDA(outKernelMatrix);

    // 计算调用 Kernel 函数的线程块的尺寸和线程块的数量
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = inSample->height / blocksize.x + 1;
    gridsize.y = inSample->height / blocksize.y + 1;

    // 调用和函数，计算核矩阵
    _calcKernelMatrixKer<<<gridsize, blocksize>>>(*inSampleCud, 
            *this, *outKernelMatrixCud);

    // 若调用 CUDA 出错返回错误代码    
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << cudaGetErrorString( cudaGetLastError()) << endl;
        return CUDA_ERROR;
    }
           
    return NO_ERROR;
}

// Host 成员函数 calcKernelMatrixSeriel（串行计算核矩阵）
__host__ int KernelMatrix::calcKernelMatrixSeriel(Matrix *inSample, Matrix *outKernelMatrix)
{
    // 判断输入指针是否为 NULL， 如果为 NULL 直接报错返回
    if (NULL == inSample || NULL == outKernelMatrix)
        return NULL_POINTER;

    // 判断输入矩阵尺寸是否正确， 如果不正确直接报错返回
    if (inSample->height < 1 || inSample->width < 1 || 
        outKernelMatrix->height < 1 || outKernelMatrix->width < 1 || 
        inSample->height != outKernelMatrix->width || 
        inSample->height != outKernelMatrix->height)
        return INVALID_DATA;

    for (int r = 0; r < outKernelMatrix->height; r++)
    {
        for (int c = 0; c < outKernelMatrix->width; c++)
        {
            if (c < r)
            {
                outKernelMatrix->matData[r * outKernelMatrix->width + c] = 0;
                continue;
            }

                 // 计算两个样例的差的2-范数的平方
                float sum = 0;

                for(int i = 0; i < inSample->width; i++)
                {
                    // 计算两个样例对应点的差
                    float difference = inSample->matData[c * inSample->width + i] - 
                        inSample->matData[r * inSample->width + i];      

                    // 计算平方并且求和
                    sum += difference * difference;               
                }

                // 计算 RBF 的值
                float rbf = expf(-0.5 * sum / (this->getSigma() * this->getSigma()));

                // 将结果写入输出矩阵
                outKernelMatrix->matData[r * outKernelMatrix->width + c] = rbf;
        }
    }
    return NO_ERROR;
}
    
