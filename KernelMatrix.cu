// KernelMatrix.cu
// ʵ�ּ�����������ĺ˾���

#include "KernelMatrix.h"
#include "math.h"
#include <iostream>
using namespace std;

// �꣺DEF_BLOCK_X �� DEF_BLOCK_Y
// ������Ĭ�ϵ��߳̿�ĳߴ硣
#define DEF_BLOCK_X  32 
#define DEF_BLOCK_Y   8 

// �˺����� _calcKernelMatrixKer��������������ĺ˾���
// ���ݸ������������󣬺Ͳ����࣬�������Ӧ�ĺ˾���
static __global__ void 
_calcKernelMatrixKer (
        MatrixCuda inMatrix,        // �������������
        KernelMatrix km,            // ������
        MatrixCuda outKernelMatrix  // ����ĺ˾���
);

// �˺����� _calcKernelMatrixKer��������������ĺ˾���
static __global__ void _calcKernelMatrixKer (
        MatrixCuda inMatrix, KernelMatrix km, MatrixCuda outKernelMatrix)
{
    // �����̶߳�Ӧ��������λ�ã����� c �� r �ֱ��ʾ����˾���ĺ�������λ��
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // ������ص��Ƿ�Խ��
    if (c >= outKernelMatrix.matMeta.width || r >= outKernelMatrix.matMeta.height)
        return;

    // ���ں˾�����һ���Գ�������ֻ��Ҫ������һ��ĵ�Ϳ��ԣ���������������ϰ벿�ֵĵ�
    // ���������� c >= r �ĵ㣬��� c < r ֱ�ӷ���
    if (c < r)
    {
        outKernelMatrix.matMeta.matData[r * outKernelMatrix.pitchWords + c] = 0;
        return;
    }

    // �������������Ĳ��2-������ƽ��
    float sum = 0;

    for(int i = 0; i < inMatrix.matMeta.width; i++)
    {
        // ��������������Ӧ��Ĳ�
        float difference = inMatrix.matMeta.matData[c * inMatrix.pitchWords + i] - 
               inMatrix.matMeta.matData[r * inMatrix.pitchWords + i];
        
        // ����ƽ���������
        sum += difference * difference;               
    }

    // ���� RBF ��ֵ
    float rbf = expf(-0.5 * sum / (km.getSigma() * km.getSigma()));

    // �����д���������
    outKernelMatrix.matMeta.matData[r * outKernelMatrix.pitchWords + c] = rbf;
}

// Host ��Ա���� calcKernelMatrix������˾���
__host__ int KernelMatrix::calcKernelMatrix(Matrix *inSample, Matrix *outKernelMatrix)
{
    // �ж�����ָ���Ƿ�Ϊ NULL�� ���Ϊ NULL ֱ�ӱ�����
    if (NULL == inSample || NULL == outKernelMatrix)
        return NULL_POINTER;

    // �ж��������ߴ��Ƿ���ȷ�� �������ȷֱ�ӱ�����
    if (inSample->height < 1 || inSample->width < 1 || 
        outKernelMatrix->height < 1 || outKernelMatrix->width < 1 || 
        inSample->height != outKernelMatrix->width || 
        inSample->height != outKernelMatrix->height)
        return INVALID_DATA;

    // ������������󿽱�����ǰ�豸��
    //MatrixBasicOp::copyToCurrentDevice(inSample);
    //MatrixBasicOp::copyToCurrentDevice(outKernelMatrix);

    MatrixCuda *inSampleCud, *outKernelMatrixCud; // ��������豸�˾���

    // ͨ��Ԥ����ĺ꽫 Matrix ָ��ת��Ϊ MatrixCuda ���͵�ָ��
    inSampleCud = MATRIX_CUDA(inSample);
    outKernelMatrixCud = MATRIX_CUDA(outKernelMatrix);

    // ������� Kernel �������߳̿�ĳߴ���߳̿������
    dim3 blocksize, gridsize;
    blocksize.x = DEF_BLOCK_X;
    blocksize.y = DEF_BLOCK_Y;
    gridsize.x = inSample->height / blocksize.x + 1;
    gridsize.y = inSample->height / blocksize.y + 1;

    // ���úͺ���������˾���
    _calcKernelMatrixKer<<<gridsize, blocksize>>>(*inSampleCud, 
            *this, *outKernelMatrixCud);

    // ������ CUDA �����ش������    
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << cudaGetErrorString( cudaGetLastError()) << endl;
        return CUDA_ERROR;
    }
           
    return NO_ERROR;
}

// Host ��Ա���� calcKernelMatrixSeriel�����м���˾���
__host__ int KernelMatrix::calcKernelMatrixSeriel(Matrix *inSample, Matrix *outKernelMatrix)
{
    // �ж�����ָ���Ƿ�Ϊ NULL�� ���Ϊ NULL ֱ�ӱ�����
    if (NULL == inSample || NULL == outKernelMatrix)
        return NULL_POINTER;

    // �ж��������ߴ��Ƿ���ȷ�� �������ȷֱ�ӱ�����
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

                 // �������������Ĳ��2-������ƽ��
                float sum = 0;

                for(int i = 0; i < inSample->width; i++)
                {
                    // ��������������Ӧ��Ĳ�
                    float difference = inSample->matData[c * inSample->width + i] - 
                        inSample->matData[r * inSample->width + i];      

                    // ����ƽ���������
                    sum += difference * difference;               
                }

                // ���� RBF ��ֵ
                float rbf = expf(-0.5 * sum / (this->getSigma() * this->getSigma()));

                // �����д���������
                outKernelMatrix->matData[r * outKernelMatrix->width + c] = rbf;
        }
    }
    return NO_ERROR;
}
    
