// Matrix.cuh
// ������;��Т��
//
// �����壨Matrix��
// ����˵���������˾�������ݽṹ�ͶԾ���Ļ�������
//
// �޶���ʷ��
// 2015��04��08�գ���Т����
//     ��ʼ�汾��

#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct Matrix_st{
    size_t width;    // ������
    size_t height;   // ����߶� 
    float *matData;  // �������ݡ�
} Matrix;

// �ṹ�壺MatrixCuda������� CUDA ������ݣ�
// �ýṹ�嶨������ CUDA ��صľ������ݡ��ýṹ��ͨ�����㷨�ڲ�ʹ�ã��ϲ��û�ͨ
// ��������Ҳ����Ҫ�˽�˽ṹ�塣
typedef struct MatrixCuda_st {
    Matrix matMeta;    // �������ݣ������˶�Ӧ�ľ����߼����ݡ�
    int deviceId;      // ��ǰ�����������ڴ棬��������� GPU ���ڴ��ϣ���
                       // deviceId Ϊ��Ӧ�豸�ı�ţ��� deviceId < 0����˵����
                       // �ݴ洢�� Host �ڴ��ϡ�
    size_t pitchWords; // Padding �����ÿ��������ռ�ڴ��������4 �ֽڣ���Ҫ��
                       // pitchBytes >= width
} MatrixCuda;

// �꣺MATRIX_CUDA
// ���� Matrix ��ָ�룬���ض�Ӧ�� MatrixCuda ��ָ�롣�ú�ͨ�����㷨�ڲ�ʹ�ã���
// ����ȡ���� CUDA �ľ������ݡ�
#define MATRIX_CUDA(mat)                                                      \
    ((MatrixCuda *)((unsigned char *)mat -                                    \
                    (unsigned long)(&(((MatrixCuda *)NULL)->matMeta))))
// �ࣺMatrixBasicOp���������������
// �̳��ԣ���
// ��������˶��ھ���Ļ��������������Ĵ��������١�����Ķ�ȡ�������ڸ���ַ��
// ��֮��Ŀ����ȡ�Ҫ�����еľ���ʵ��������Ҫͨ�����ഴ�������򣬽��ᵼ��ϵͳ��
// �е����ң���Ҫ��֤ÿһ�� Matrix �͵����ݶ��ж�Ӧ�� MatrixCuda ���ݣ���
class MatrixBasicOp {

public:

    // Host ��̬������newMatrix����������
    // ����һ���µľ���ʵ������ͨ������ outmat ���ء�
    static __host__ int newMatrix(Matrix **outmat);

    // Host ��̬������deleteMatrix�����پ���
    // ����һ�����ٱ�ʹ�õľ���ʵ����
    static __host__ int deleteMatrix(Matrix *inmat);

    // Host ��̬������makeAtCurrentDevice���ڵ�ǰ Device �ڴ��й������ݣ�
    // ��Կվ����ڵ�ǰ Device �ڴ���Ϊ������һ��ָ���Ĵ�С�Ŀռ䣬��οռ���
    // ��������δ����ֵ�Ļ������ݡ�������ǿվ�����÷����ᱨ��
    static __host__ int makeAtCurrentDevice(Matrix *mat, 
                                    size_t width,size_t height);

    // Host ��̬������makeAtHost���� Host �ڴ��й������ݣ�
    // ��Կվ����� Host �ڴ���Ϊ������һ��ָ���Ĵ�С�Ŀռ䣬��οռ��е�����
    // ��δ����ֵ�Ļ������ݡ�������ǿվ�����÷����ᱨ��
    static __host__ int makeAtHost(Matrix *mat,size_t width,size_t height);
              
    // Host ��̬������readFromFile����CSV�ļ���ȡ����
    // ��ָ�����ļ�·���ж�ȡһ�����󡣸þ���������Ⱦ��� newMatrix ��������ȡ
    // �����Ĭ�Ϸ��� Host �ڴ��У�����Ҫ���ļ�Ϊ���ı���ʽ���Ҿ�������֮��ʹ��
    // ���ָ��������CSV��ʽ
    static __host__ int readFromFile(const char *filepath, unsigned int width, unsigned int height, Matrix *outmat);                                  
    

    // Host ��̬������writeToFile��������д��CSV�ļ���
    // ����д�뵽ָ�����ļ��С��þ����б������������,д��ĸ�ʽ�ǰ���CSV��ʽ����д���
    static __host__ int writeToFile(const char *filepath, Matrix *inmat);

    // Host ��̬������copyToCurrentDevice�������󿽱�����ǰ Device �ڴ��ϣ�
    // ����һ�� In-Place ��ʽ�Ŀ���������������ݱ������ڵ�ǰ�� Device �ϣ����
    // ������������κβ�����ֱ�ӷ��ء�����������ݲ��ڵ�ǰ Device �ϣ���Ὣ��
    // �ݿ�������ǰ Device �ϣ������� matData ָ�롣ԭ�������ݽ��ᱻ�ͷš�
    static __host__ int copyToCurrentDevice(Matrix *mat);

    // Host ��̬������copyToCurrentDevice�������󿽱�����ǰ Device �ڴ��ϣ�
    // ����һ�� Out-Place ��ʽ�Ŀ��������� srcmat λ����һ���ڴ�ռ��У������
    // ��һ������������ȫһ�µ� dstmat���������Ǵ洢�ڵ�ǰ�� Device �ϵġ����
    // dstmat ��ԭ�����������ݣ���ԭ��������ͬ�µ����ݳߴ���ͬ��Ҳ����ڵ�ǰ
    // Device �ϣ��򸲸�ԭ���ݣ������·���ռ䣻����ԭ���ݽ��ᱻ�ͷţ���������
    // ��ռ䡣
    static __host__ int copyToCurrentDevice(Matrix *srcmat, Matrix *dstmat);

    // Host ��̬������copyToHost�������󿽱��� Host �ڴ��ϣ�
    // ����һ�� In-Place ��ʽ�Ŀ���������������ݱ������� Host �ϣ���ú�������
    // �����κβ�����ֱ�ӷ��ء�����������ݲ��� Host �ϣ���Ὣ���ݿ����� Host
    // �ϣ������� matData ָ�롣ԭ�������ݽ��ᱻ�ͷš�
    static __host__ int copyToHost(Matrix *mat);

    // Host ��̬������copyToHost�������󿽱��� Host �ڴ��ϣ�
    // ����һ�� Out-Place ��ʽ�Ŀ��������� srcmat λ����һ���ڴ�ռ��У������
    // ��һ������������ȫһ�µ� dstmat���������Ǵ洢�� Host �ϵġ���� dstmat
    // ��ԭ�����������ݣ����ԭ��������ͬ�µ����ݳߴ���ͬ����Ҳ����� Host �ϣ�
    // �򸲸�ԭ���ݣ��������·���ռ䣻����ԭ���ݽ��ᱻ�ͷţ�����������ռ䡣
    static __host__ int copyToHost(Matrix *srcmat, Matrix *dstmat);       
};


#endif