// Matrix.cu
// �����˾�������ݽṹ�ͶԾ���Ļ�������

#include "Matrix.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

#include "ErrorCode.h"

__host__ int MatrixBasicOp::newMatrix(Matrix **outmat)
{
     MatrixCuda *resmatCud;  // ��Ӧ�ڷ��ص� outmat �� MatrixCuda �����ݡ�

    // ���װ����������ָ���Ƿ�Ϊ NULL��
    if (outmat == NULL)
        return NULL_POINTER;

    // �������Ԫ���ݵĿռ䡣
    resmatCud = new MatrixCuda;

    // ��ʼ�������ϵ�����Ϊ�վ���
    resmatCud->matMeta.width = 0;
    resmatCud->matMeta.height = 0;
    resmatCud->matMeta.matData = NULL;
    resmatCud->deviceId = -1;
    resmatCud->pitchWords = 0;

    // �� Matrix ��ֵ�����������
    *outmat = &(resmatCud->matMeta);

    // ������ϣ����ء�
    return NO_ERROR;
}

// Host ��̬������deleteMatrix�����پ���
__host__ int MatrixBasicOp::deleteMatrix(Matrix *inmat)
{
    // �������ָ���Ƿ�Ϊ NULL��
    if (inmat == NULL)
        return NULL_POINTER;

    // ������������� Matrix ��ָ�룬�õ���Ӧ�� MatrixCuda �����ݡ�
    MatrixCuda *inmatCud = MATRIX_CUDA(inmat);

    // ���������ڵĵ�ַ�ռ��Ƿ�Ϸ�������������ڵ�ַ�ռ䲻���� Host ���κ�һ
    // �� Device����ú�������������������󣬱�ʾ�޷�����
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (inmatCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // ��ȡ��ǰ Device ID��
    int curdevid;
    cudaGetDevice(&curdevid);

    // �ͷž������ݣ����������ݡ�
    if (inmat->matData == NULL || inmat->width == 0 || inmat->height == 0 ||
        inmatCud->pitchWords == 0) {
        // �����������ǿյģ��򲻽��о��������ͷŲ�������Ϊ����Ҳû�����ݿɱ�
        // �ͷţ���
        // Do Nothing;
    } if (inmatCud->deviceId < 0) {
        // �������ݴ洢�� Host �ڴ棬ֱ������ delete �ؼ����ͷž������ݡ�
        delete[] inmat->matData;
    } else if (inmatCud->deviceId == curdevid) {
        // �������ݴ洢�ڵ�ǰ Device �ڴ��У���ֱ������ cudaFree �ӿ��ͷŸþ���
        // ���ݡ�
        cudaFree(inmat->matData);
    } else {
        // �������ݴ洢�ڷǵ�ǰ Device �ڴ��У�����Ҫ�����л��豸�������豸��Ϊ
        // ��ǰ Device��Ȼ���ͷ�֮�������Ҫ���豸�л������Ա�֤�����������
        // ȷ�ԡ�
        cudaSetDevice(inmatCud->deviceId);
        cudaFree(inmat->matData);
        cudaSetDevice(curdevid);
    }

    // �ͷž����Ԫ���ݡ�
    delete inmatCud;

    // ������ϣ����ء�
    return NO_ERROR;
}

// Host ��̬������makeAtCurrentDevice���ڵ�ǰ Device �ڴ��й������ݣ�
__host__ int MatrixBasicOp::makeAtCurrentDevice(Matrix *mat,
                                                size_t width, size_t height)
{
    // �����������Ƿ�Ϊ NULL
    if (mat == NULL)
        return NULL_POINTER;

    // �������ľ���ĳ����Ƿ�Ϸ�
    if (width < 1 || height < 1)
        return INVALID_DATA;

    // �������Ƿ�Ϊ�վ���
    if (mat->matData != NULL)
        return INVALID_DATA;

    // ��ȡ mat ��Ӧ�� MatrixCuda �����ݡ�
    MatrixCuda *matCud = MATRIX_CUDA(mat);

    // �ڵ�ǰ�� Device ������洢ָ���ߴ��������Ҫ���ڴ�ռ䡣
    cudaError_t cuerrcode;
    float *newspace;
    size_t pitchbytes;
    cuerrcode = cudaMallocPitch((void **)(&newspace), &pitchbytes,
                                width * sizeof (float), height);
    if (cuerrcode != cudaSuccess)
        return CUDA_ERROR;

    // ��ȡ��ǰ Device ID��
    int curdevid;
    cudaGetDevice(&curdevid);

    // �޸ľ����Ԫ���ݡ����� ROI ����Ϊ����ͼƬ��
    mat->width = width;
    mat->height = height;
    mat->matData = newspace;
    matCud->deviceId = curdevid;
    matCud->pitchWords = pitchbytes / sizeof (float);
    // �������Ǽ����� cudaMallocPitch �õ��� pitch �ǿ��Ա� sizeof (float) ����
    // �ġ������Ĳ�����������������г�����������󣨵�Ȼ������ʵ�� CUDA ����
    // ������ֵģ���

    // ������ϣ��˳���
    return NO_ERROR;
}

// Host ��̬������makeAtHost���� Host �ڴ��й������ݣ�
__host__ int MatrixBasicOp::makeAtHost(Matrix *mat,
                                       size_t width, size_t height)
{
    // �����������Ƿ�Ϊ NULL
    if (mat == NULL)
        return NULL_POINTER;

    // �������ľ���ĳ����Ƿ�Ϸ�
    if (width < 1 || height < 1)
        return INVALID_DATA;

    // �������Ƿ�Ϊ�վ���
    if (mat->matData != NULL)
        return INVALID_DATA;

    // ��ȡ mat ��Ӧ�� MatrixCuda �����ݡ�
    MatrixCuda *matCud = MATRIX_CUDA(mat);

    // Ϊ���������� Host �ڴ�������ռ�
    mat->matData = new float[width * height];
    if (mat->matData == NULL)
        return OUT_OF_MEM;

    // ���þ����е�Ԫ����
    mat->width = width;
    mat->height = height;
    matCud->deviceId = -1;
    matCud->pitchWords = width;

    // ������ϣ��˳�
    return NO_ERROR;
}

// Host ��̬������readFromFile�����ļ���ȡ����
__host__ int MatrixBasicOp::readFromFile(const char *filepath, unsigned int width, unsigned int height, Matrix *outmat)
{
    // ����ļ�·���;����Ƿ�Ϊ NULL��
    if (filepath == NULL || outmat == NULL)
        return NULL_POINTER;

    // �������ĳߴ粻�Ϸ����򱨴��˳���
    if (width < 1 || height < 1)
        return WRONG_FILE;

    // ������������� Matrix ��ָ�룬�õ���Ӧ�� MatrixCuda �����ݡ�
    MatrixCuda *outmatCud = MATRIX_CUDA(outmat);

    // ���������ڵĵ�ַ�ռ��Ƿ�Ϸ�������������ڵ�ַ�ռ䲻���� Host ���κ�һ
    // �� Device����ú�������������������󣬱�ʾ�޷�����
    int devcnt;
    cudaGetErrorString(cudaGetDeviceCount(&devcnt));
    if (outmatCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // ��ȡ��ǰ Device ID��
    int curdevid;
    cudaGetDevice(&curdevid);

    // �򿪾����ļ���
    ifstream fin(filepath);
    if (!fin)
        return NO_FILE;

    // ��ȡ��ž������ݵ� Host �ڴ�ռ䡣���ž������õ�˼�룬���ԭ���ľ���
    // �ڴ������Ǵ洢�� Host �ڴ棬�ҳߴ���µľ���ߴ�һ��ʱ������������
    // Host �ڴ�ռ䣬ֱ������ԭ���Ŀռ����µľ������ݡ�
    float *matdata = outmat->matData;
    bool reusedata = true;
    if (outmat->matData == NULL || outmatCud->deviceId >= 0 || 
        outmat->width != width || outmat->height != height) {
        matdata = new float[width * height];
        // ���û�����뵽�µ����ݣ��򱨴�
        if (matdata == NULL)
            return OUT_OF_MEM;
        reusedata = false;
    }

    // ��ȡ�����ļ��е�����
	string line;  
    for(int r = 0; r < height; r++){
        // ��ÿһ�е����ݶ��뵽�ַ���line��
		if(getline(fin, line)){            
			istringstream sin(line);    						
            for(int c = 0; c < width; c++){
				string field;
                // ���������ݰ���,�ָ��һ��������
				if (getline(sin, field, ',')) 
                    // ������д�뵽���ݾ����Ӧ��λ��
                    matdata[r * width + c] = atof(const_cast<const char *>(field.c_str()));    				
			}		
		}
	}

    // ����Ϊֹ���������ݶ�ȡ��ϣ����ǿ��԰�ȫ���ͷŵ�����ԭ�������ݡ�һֱ�ϵ�
    // �����ͷ�ԭ�������ݣ�����Ϊ�˷�ֹһ�������ȡʧ�ܣ���������ϵͳ����һ��
    // ���ҵ�״̬����Ϊԭ�������ݻ��Ǵ���һ�����õ�״̬��
    if (reusedata == false || outmat->matData != NULL) {
        if (outmatCud->deviceId < 0) {
            // ���ԭ�������ݴ���� Host �ڴ��У���ʹ�� delete �ؼ����ͷš�
            delete[] outmat->matData;
        } else {
            // ���ԭ�������ݴ���� Device �ڴ��У������ȵ�����Ӧ�� Device��Ȼ
            // ��ʹ�� cudaFree �ͷŵ��ڴ档
            cudaSetDevice(outmatCud->deviceId);
            cudaFree(outmat->matData);
            cudaSetDevice(curdevid);
        }
    }

    // Ϊ����ֵ�µ�Ԫ���ݡ����� ROI ������Ϊ��������
    outmat->width = width;
    outmat->height = height;    
    outmat->matData = matdata;
    outmatCud->deviceId = -1;    
    outmatCud->pitchWords = width;

    // ������ϣ����ء�
    return NO_ERROR;
}

// Host ��̬������writeToFile��������д���ļ���
__host__ int MatrixBasicOp::writeToFile(const char *filepath, Matrix *inmat)
{
    // ����ļ�·���;����Ƿ�Ϊ NULL��
    if (filepath == NULL || inmat == NULL)
        return NULL_POINTER;

    // ����Ҫд����ļ���
    ofstream matfile(filepath,ios::app);
    if (!matfile) 
        return NO_FILE;

    // ������������� Matrix ��ָ�룬�õ���Ӧ�� MatrixCuda �����ݡ�
    MatrixCuda *inmatCud = MATRIX_CUDA(inmat);

    // ����������ݿ����� Host �ڴ��У������������ݾͿ��Ա�����Ĵ�������ȡ��Ȼ��
    // ���������д�뵽�ļ��С�������Ҫע����ǣ������ļ��Ŀ����������ļ���֮
    // ������Ϊ�����һ���ļ���ʧ�ܣ��򲻻�ı�������ڴ��еĴ洢״̬�������
    // ��Ժ����������������
    int errcode;
    errcode = MatrixBasicOp::copyToHost(inmat);
    if (errcode < 0)
        return errcode;

    // ����д���������ݡ�
    for (int r = 0; r < inmat->height; r++) {
        for (int c = 0; c < inmat->width; c++) {
           // ����Ӧλ�õ���Ϣд���ļ�
           matfile << inmat->matData[r * inmatCud->pitchWords + c] << ",";                    
        }
        matfile << endl;
    }
    matfile.close();
    // ������ϣ����ء�
    return NO_ERROR;
}

// Host ��̬������copyToCurrentDevice�������󿽱�����ǰ Device �ڴ��ϣ�
__host__ int MatrixBasicOp::copyToCurrentDevice(Matrix *mat)
{
    // �������Ƿ�Ϊ NULL��
    if (mat == NULL)
        return NULL_POINTER;

    // ������������� Matrix ��ָ�룬�õ���Ӧ�� MatrixCuda �����ݡ�
    MatrixCuda *matCud = MATRIX_CUDA(mat);

    // ���������ڵĵ�ַ�ռ��Ƿ�Ϸ�������������ڵ�ַ�ռ䲻���� Host ���κ�һ
    // �� Device����ú�������������������󣬱�ʾ�޷�����
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (matCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // ��ȡ��ǰ Device ID��
    int curdevid;
    cudaGetDevice(&curdevid);

    // ���������һ�����������ݵĿվ����򱨴�
    if (mat->matData == NULL || mat->width == 0 || mat->height == 0 || 
        matCud->pitchWords == 0) 
        return INVALID_DATA;

    // ���ڲ�ͬ����������������ݿ�������ǰ�豸�ϡ�
    if (matCud->deviceId < 0) {
        // ������������λ�� Host �ڴ��ϣ�����Ҫ�ڵ�ǰ Device ���ڴ�ռ�������
        // �ռ䣬Ȼ�� Host �ڴ��ϵ����ݽ��� Padding �󿽱�����ǰ Device �ϡ�
        float *devptr;  // �µ����ݿռ䣬�ڵ�ǰ Device �ϡ�
        size_t pitch;           // Padding ���ÿ�гߴ�
        cudaError_t cuerrcode;  // CUDA ���÷��صĴ����롣

        // �ڵ�ǰ�豸������ռ䣬ʹ�� Pitch �汾�����뺯������������ Padding��
        cuerrcode = cudaMallocPitch((void **)(&devptr), &pitch, 
                                    mat->width * sizeof (float), mat->height);
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // ���� Padding ���������ݵ���ǰ Device �ϡ�ע�⣬���� mat->pitchWords
        // == mat->width��
        cuerrcode = cudaMemcpy2D(devptr, pitch, 
                                 mat->matData, 
                                 matCud->pitchWords * sizeof (float),
                                 mat->width * sizeof (float), mat->height,
                                 cudaMemcpyHostToDevice);
        
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }

        // �ͷŵ�ԭ���洢�� Host �ڴ��ϵľ������ݡ�
        delete[] mat->matData;

        // ���¾������ݣ����µ��ڵ�ǰ Device ����������ݺ��������д�����Ԫ��
        // ���С�
        mat->matData = devptr;
        matCud->deviceId = curdevid;
        matCud->pitchWords = pitch / sizeof (float);

        // ������ϣ����ء�
        return NO_ERROR;

    } else if (matCud->deviceId != curdevid) {
        // �������ݴ������� Device ��������Ծ�Ҫ�ڵ�ǰ Device ���������ݿռ䣬
        // ������һ�� Device �Ͽ������ݵ�������ĵ�ǰ Device �����ݿռ��С�
        float *devptr;  // ������ĵ�ǰ Device �ϵ����ݡ�
        size_t datasize = matCud->pitchWords * mat->height *  // ���ݳߴ硣
                          sizeof (float);
        cudaError_t cuerrcode;  // CUDA ���÷��صĴ����롣

        // �ڵ�ǰ Device ������ռ䡣
        cuerrcode = cudaMalloc((void **)(&devptr), datasize);
        if (cuerrcode != cudaSuccess)
            return CUDA_ERROR;

        // �����ݴӾ���ԭ���Ĵ洢λ�ÿ�������ǰ�� Device �ϡ�
        cuerrcode = cudaMemcpyPeer(devptr, curdevid, 
                                   mat->matData, matCud->deviceId,
                                   datasize);
        if (cuerrcode != cudaSuccess) {
            cudaFree(devptr);
            return CUDA_ERROR;
        }

        // �ͷŵ�������ԭ���� Device �ϵ����ݡ�
        cudaFree(mat->matData);

        // ���µľ���������Ϣд�뵽����Ԫ�����С�
        mat->matData = devptr;
        matCud->deviceId = curdevid;

        // ������ɣ����ء�
        return NO_ERROR;
    }

    // ����������������������ݱ������ڵ�ǰ Device �ϣ���ֱ�ӷ��أ��������κε�
    // ������
    return NO_ERROR;
}

// Host ��̬������copyToCurrentDevice�������󿽱�����ǰ Device �ڴ��ϣ�
__host__ int MatrixBasicOp::copyToCurrentDevice(Matrix *srcmat, Matrix *dstmat)
{
    // �����������Ƿ�Ϊ NULL��
    if (srcmat == NULL)
        return NULL_POINTER;

    // ����������Ϊ NULL���������������������Ϊͬһ����������� In-place
    // �汾�ĺ�����
    if (dstmat == NULL || dstmat == srcmat)
        return copyToCurrentDevice(srcmat);

    // ��ȡ srcmat �� dstmat ��Ӧ�� MatrixCuda ��ָ�롣
    MatrixCuda *srcmatCud = MATRIX_CUDA(srcmat);
    MatrixCuda *dstmatCud = MATRIX_CUDA(dstmat);

    // ������žɵ� dstmat ���ݣ�ʹ���ڿ�������ʧ��ʱ���Իָ�Ϊԭ���Ŀ��õ�����
    // ��Ϣ����ֹϵͳ����һ�����ҵ�״̬��
    MatrixCuda olddstmatCud = *dstmatCud;  // �ɵ� dstmat ����
    bool reusedata = true;                // ��¼�Ƿ�������ԭ���ľ������ݿռ䡣
                                          // ��ֵΪ ture����ԭ�������ݿռ䱻��
                                          // �ã�����Ҫ��֮���ͷ����ݣ�������Ҫ
                                          // ������ͷžɵĿռ䡣

    // ���Դ������һ���վ����򲻽����κβ�����ֱ�ӱ���
    if (srcmat->matData == NULL || srcmat->width == 0 || srcmat->height == 0 ||
        srcmatCud->pitchWords == 0)
        return INVALID_DATA;

    // ���������ڵĵ�ַ�ռ��Ƿ�Ϸ�������������ڵ�ַ�ռ䲻���� Host ���κ�һ
    // �� Device����ú�������������������󣬱�ʾ�޷�����
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srcmatCud->deviceId >= devcnt || dstmatCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // ��ȡ��ǰ Device ID��
    int curdevid;
    cudaGetDevice(&curdevid);

    // ���Ŀ������д��������ݣ�����Ҫ�����������ԭ�������ݲ��洢�ڵ�ǰ��
    // Device �ϣ����߼�ʹ�洢�ڵ�ǰ�� Device �ϣ������ݳߴ粻ƥ�䣬����Ҫ�ͷ�
    // ��ԭ������Ŀռ䣬�Ա�����������ʵ��ڴ�ռ䡣�˴��������������ͷŲ�����
    // ��Ŀ�����ڵ������������ִ���ʱ�����Ժܿ�Ļָ� dstmat ��ԭ������Ϣ��ʹ��
    // ����ϵͳ���ᴦ��һ�����ҵ�״̬���������������ȷ�� dstmat ���ɹ��ĸ���
    // Ϊ���µ������Ժ󣬲Ż������Ľ�ԭ���ľ��������ͷŵ���
    if (dstmatCud->deviceId != curdevid) {
        // �������ݴ��� Host �������� Device �ϣ���ֱ���ͷŵ�ԭ�������ݿռ䡣
        reusedata = 0;
        dstmat->matData = NULL;
    } else if (!(((srcmatCud->deviceId < 0 && 
                   srcmat->width == dstmat->width) ||
                  dstmatCud->pitchWords == srcmatCud->pitchWords) &&
                 srcmat->height == dstmat->height)) {
        // �������ݴ����ڵ�ǰ Device �ϣ�����Ҫ������ݵĳߴ��Ƿ��Դ������ƥ
        // �䡣���ı�׼������Ҫ��Դ����� Padding ����п�Ⱥ�Ŀ��������
        // ͬ��Դ�����Ŀ�����ĸ߶���ͬ�����Դ�����Ǵ洢�� Host �ڴ��еģ���
        // ��Ҫ��Դ�����Ŀ�����Ŀ����ͬ���ɡ����Ŀ������Դ����ĳߴ粻ƥ
        // �����Ծ���Ҫ�ͷ�Ŀ�����ԭ�������ݿռ䡣
        reusedata = 0;
        dstmat->matData = NULL;
    }

    // ��Ŀ�����ĳߴ����ΪԴ����ĳߴ硣
    dstmat->width = srcmat->width;
    dstmat->height = srcmat->height;

    // ����Ŀ���������ݴ洢λ��Ϊ��ǰ Device��
    dstmatCud->deviceId = curdevid;

    // ���������ݴ�Դ�����п�����Ŀ������С�
    if (srcmatCud->deviceId < 0) {
        // ���Դ�������ݴ洢�� Host �ڴ棬��ʹ�� cudaMemcpy2D ���� Padding ��
        // ʽ�Ŀ�����
        cudaError_t cuerrcode;  // CUDA ���÷��صĴ����롣

        // ���Ŀ������ matData == NULL��˵��Ŀ�����ԭ��Ҫô��һ���վ���Ҫ
        // ôĿ�����ԭ�������ݿռ䲻���ʣ���Ҫ�������롣��ʱ����ҪΪĿ�������
        // ���ڵ�ǰ Device ������һ�����ʵ����ݿռ䡣
        if (dstmat->matData == NULL) {
            cuerrcode = cudaMallocPitch((void **)(&dstmat->matData), 
                                        &dstmatCud->pitchWords,
                                        dstmat->width * sizeof (float),
                                        dstmat->height);
            if (cuerrcode != cudaSuccess) {
                // ��������ڴ�Ĳ���ʧ�ܣ����ٱ�����ǰ��Ҫ���ɵ�Ŀ���������
                // �ָ���Ŀ������У��Ա�֤ϵͳ���µĲ��������ڻ��ҡ�
                *dstmatCud = olddstmatCud;
                return CUDA_ERROR;
            }

            // ��ͨ�� cudaMallocPitch �õ������ֽ�Ϊ��λ�� pitch ֵת��Ϊ����Ϊ
            // ��λ��ֵ��
            dstmatCud->pitchWords /= sizeof (float);
        }

        // ʹ�� cudaMemcpy2D ���� Padding ��ʽ�Ŀ�����
        cuerrcode = cudaMemcpy2D(dstmat->matData,
                                 dstmatCud->pitchWords * sizeof (float),
                                 srcmat->matData,
                                 srcmatCud->pitchWords * sizeof (float),
                                 srcmat->width * sizeof (float),
                                 srcmat->height,
                                 cudaMemcpyHostToDevice);
        if (cuerrcode != cudaSuccess) {
            // �����������ʧ�ܣ����ٱ����˳�ǰ����Ҫ���ɵ�Ŀ��������ݻָ���Ŀ
            // ������С����⣬������ݲ������õģ�����Ҫ�ͷ�����������ݿռ䣬
            // ��ֹ�ڴ�й©��
            if (!reusedata)
                cudaFree(dstmat->matData);
            *dstmatCud = olddstmatCud;
            return CUDA_ERROR;
        }
    } else {
        
        // ���Դ�������ݴ洢�� Device �ڴ棨�����ǵ�ǰ Device ���������� 
        // Device���������ö˵��˵Ŀ�����
        cudaError_t cuerrcode;             // CUDA ���÷��صĴ����롣
        size_t datasize = srcmatCud->pitchWords * srcmat->height *
                          sizeof (float);  // ���ݳߴ磬��λ���ֽڡ�

        // ���Ŀ�������Ҫ�������ݿռ䣬��������롣
        if (dstmat->matData == NULL) {
            cuerrcode = cudaMalloc((void **)(&dstmat->matData), datasize);
            if (cuerrcode != cudaSuccess) {
                // ���������������Ҫ���Ȼָ��ɵľ������ݣ�֮�󱨴��ָ��ɵ�
                // ���������Է�ֹϵͳ�������״̬��
                *dstmatCud = olddstmatCud;
                return CUDA_ERROR;
            }
        }

        // ����Ŀ������ Padding �ߴ���Դ������ͬ��ע�⣬��ΪԴ����Ҳ�洢��
        // Device �ϣ��� Device �ϵ����ݶ��Ǿ��� Padding �ģ�����Ϊ
        // cudaMemcpyPeer ����û���ṩ Pitch �汾�ӿڣ����ԣ���������ֱ�ӽ���Դ
        // ����� Padding �ߴ硣
        dstmatCud->pitchWords = srcmatCud->pitchWords;

        // ʹ�� cudaMemcpyPeer ʵ������ Device ������Ϊͬһ�� Device���������
        // ��������Դ������ Device �ϵ�������Ϣ���Ƶ�Ŀ������С�
        cuerrcode = cudaMemcpyPeer(dstmat->matData, curdevid,
                                   srcmat->matData, srcmatCud->deviceId,
                                   datasize);
        if (cuerrcode != cudaSuccess) {
            // �����������ʧ�ܣ����ٱ����˳�ǰ����Ҫ���ɵ�Ŀ��������ݻָ���Ŀ
            // ������С����⣬������ݲ������õģ�����Ҫ�ͷ�����������ݿռ䣬
            // ��ֹ�ڴ�й©��
            if (!reusedata)
                cudaFree(dstmat->matData);
            *dstmatCud = olddstmatCud;
            return CUDA_ERROR;
        }
    }

    // ���˲����Ѿ�˵���µľ������ݿռ��Ѿ��ɹ������벢�������µ����ݣ���ˣ���
    // �����ݿռ��Ѻ����ô�������������ͷŵ��ɵ����ݿռ��Է�ֹ�ڴ�й©�����
    // ��Ϊ������ olddstmatCud �Ǿֲ������������Ӧ��Ԫ���ݻ��ڱ������˳����Զ�
    // �ͷţ�������ᡣ
    if (olddstmatCud.matMeta.matData != NULL) {
        if (olddstmatCud.deviceId < 0) {
            // ��������ݿռ��� Host �ڴ��ϵģ�����Ҫ�������ͷš�
            delete[] olddstmatCud.matMeta.matData;
        } else if (olddstmatCud.deviceId != curdevid) {
            // ��������ݿռ䲻�ǵ�ǰ Device �ڴ��ϵ����� Device �ڴ��ϵ����ݣ�
            // ��Ҳ��Ҫ���������ͷš�
            cudaSetDevice(olddstmatCud.deviceId);
            cudaFree(olddstmatCud.matMeta.matData);
            cudaSetDevice(curdevid);
        } else if (!reusedata) {
            // ��������ݾ��ڵ�ǰ�� Device �ڴ��ϣ������ reusedata δ��λ����
            // �������ͷţ���Ϊһ����λ���ɵ����ݿռ�ͱ����ڳ����µ����ݣ���
            // ���ͷš�
            cudaFree(olddstmatCud.matMeta.matData);
        }
    }

    return NO_ERROR;
}

// Host ��̬������copyToHost�������󿽱��� Host �ڴ��ϣ�
__host__ int MatrixBasicOp::copyToHost(Matrix *mat)
{
    // �������Ƿ�Ϊ NULL��
    if (mat == NULL)
        return NULL_POINTER;

    // ������������� Matrix ��ָ�룬�õ���Ӧ�� MatrixCuda �����ݡ�
    MatrixCuda *matCud = MATRIX_CUDA(mat);

    // ���������ڵĵ�ַ�ռ��Ƿ�Ϸ�������������ڵ�ַ�ռ䲻���� Host ���κ�һ
    // �� Device����ú�������������������󣬱�ʾ�޷�����
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (matCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // ��ȡ��ǰ Device ID��
    int curdevid;
    cudaGetDevice(&curdevid);

    // ���������һ�����������ݵĿվ����򱨴�
    if (mat->matData == NULL || mat->width == 0 || mat->height == 0 || 
        matCud->pitchWords == 0) 
        return INVALID_DATA;

    // ���ڲ�ͬ����������������ݿ�������ǰ�豸�ϡ�
    if (matCud->deviceId < 0) {
        // �������λ�� Host �ڴ��ϣ�����Ҫ�����κβ�����
        return NO_ERROR;

    } else {
        // ������������λ�� Device �ڴ��ϣ�����Ҫ�� Host ���ڴ�ռ��������
        // �䣬Ȼ���������� Padding �󿽱��� Host �ϡ�
        float *hostptr;         // �µ����ݿռ䣬�� Host �ϡ�
        cudaError_t cuerrcode;  // CUDA ���÷��صĴ����롣

        // �� Host ������ռ䡣
        hostptr = new float[mat->width * mat->height];
        if (hostptr == NULL)
            return OUT_OF_MEM;

        // ���豸�л����������ڵ� Device �ϡ�
        cudaSetDevice(matCud->deviceId);

        // ���� Padding ����������
        cuerrcode = cudaMemcpy2D(hostptr, mat->width * sizeof (float),
                                 mat->matData,
                                 matCud->pitchWords * sizeof (float),
                                 mat->width * sizeof (float), mat->height,
                                 cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // �������ʧ�ܣ�����Ҫ�ͷŵ��ո�������ڴ�ռ䣬�Է�ֹ�ڴ�й©��֮
            // �󱨴��ء�
            delete[] hostptr;
            return CUDA_ERROR;
        }

        // �ͷŵ�ԭ���洢�� Device �ڴ��ϵľ������ݡ�
        cudaFree(mat->matData);

        // �� Device �ڴ�Ĳ�����ϣ����豸�л��ص�ǰ Device��
        cudaSetDevice(curdevid);

        // ���¾������ݣ����µ��ڵ�ǰ Device ����������ݺ��������д�����Ԫ��
        // ���С�
        mat->matData = hostptr;
        matCud->deviceId = -1;
        matCud->pitchWords = mat->width;

        // ������ϣ����ء�
        return NO_ERROR;
    }

    // ������ԶҲ���ᵽ�������֧�����������������֧����˵��ϵͳ���ҡ����ڶ�
    // ����������˵����Դ˾䱨�����ɴ����� Warning��������ｫ��ע�͵����Է�
    // ֹ����Ҫ�� Warning��
    //return UNKNOW_ERROR;
}

// Host ��̬������copyToHost�������󿽱��� Host �ڴ��ϣ�
__host__ int MatrixBasicOp::copyToHost(Matrix *srcmat, Matrix *dstmat)
{
    // �����������Ƿ�Ϊ NULL��
    if (srcmat == NULL)
        return NULL_POINTER;

    // ����������Ϊ NULL ���ߺ��������ͬΪһ����������ö�Ӧ�� In-place ��
    // ���ĺ�����
    if (dstmat == NULL || dstmat == srcmat)
        return copyToHost(srcmat);

    // ��ȡ srcmat �� dstmat ��Ӧ�� MatrixCuda ��ָ�롣
    MatrixCuda *srcmatCud = MATRIX_CUDA(srcmat);
    MatrixCuda *dstmatCud = MATRIX_CUDA(dstmat);

    // ������žɵ� dstmat ���ݣ�ʹ���ڿ�������ʧ��ʱ���Իָ�Ϊԭ���Ŀ��õ�����
    // ��Ϣ����ֹϵͳ����һ�����ҵ�״̬��
    MatrixCuda olddstmatCud = *dstmatCud;  // �ɵ� dstmat ����
    bool reusedata = true;                // ��¼�Ƿ�������ԭ���ľ������ݿռ䡣
                                          // ��ֵΪ true����ԭ�������ݿռ䱻��
                                          // �ã�����Ҫ��֮���ͷ����ݣ�������Ҫ
                                          // �ͷžɵĿռ䡣

    // ���Դ������һ���վ����򲻽����κβ�����ֱ�ӱ���
    if (srcmat->matData == NULL || srcmat->width == 0 || srcmat->height == 0 ||
        srcmatCud->pitchWords == 0)
        return INVALID_DATA;

    // ���������ڵĵ�ַ�ռ��Ƿ�Ϸ�������������ڵ�ַ�ռ䲻���� Host ���κ�һ
    // �� Device����ú�������������������󣬱�ʾ�޷�����
    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (srcmatCud->deviceId >= devcnt || dstmatCud->deviceId >= devcnt)
        return OP_OVERFLOW;

    // ��ȡ��ǰ Device ID��
    int curdevid;
    cudaGetDevice(&curdevid);

    // ���Ŀ������д��������ݣ�����Ҫ�����������ԭ�������ݲ��洢�� Host �ϣ�
    // ���߼�ʹ�洢�� Host �ϣ������ݳߴ粻ƥ�䣬����Ҫ�ͷŵ�ԭ������Ŀռ䣬��
    // ������������ʵ��ڴ�ռ䡣�˴��������������ͷŲ�������Ŀ�����ڵ���������
    // ���ִ���ʱ�����Ժܿ�Ļָ� dstmat ��ԭ������Ϣ��ʹ������ϵͳ���ᴦ��һ��
    // ���ҵ�״̬���������������ȷ�� dstmat ���ɹ��ĸ���Ϊ���µ������Ժ󣬲�
    // �������Ľ�ԭ���ľ��������ͷŵ���
    if (dstmatCud->deviceId >= 0) {
        // �������ݴ����� Device �ϣ�����ֱ���ͷŵ�ԭ�������ݿռ䡣
        reusedata = 0;
        dstmat->matData = NULL;
    } else if (!(srcmat->width == dstmat->width &&
                 srcmat->height == dstmat->height)) {
        // �������ݴ����� Host �ϣ�����Ҫ������ݵĳߴ��Ƿ��Դ������ƥ�䡣���
        // �ı�׼��Դ�����Ŀ�����ĳߴ���ͬʱ��������ԭ���Ŀռ䡣
        reusedata = 0;
        dstmat->matData = NULL;
    }

    // ��Ŀ�����ĳߴ����ΪԴ����ĳߴ硣
    dstmat->width = srcmat->width;
    dstmat->height = srcmat->height;  

    // ����Ŀ���������ݴ洢λ��Ϊ Host��
    dstmatCud->deviceId = -1;

    // ���� Host �ڴ��ϵ����ݲ�ʹ�� Padding��������� Padding �ߴ�Ϊ����Ŀ�
    // �ȡ�
    dstmatCud->pitchWords = dstmat->width;

    // ���Ŀ������ matData == NULL��˵��Ŀ�����ԭ��Ҫô��һ���վ���ҪôĿ
    // �����ԭ�������ݿռ䲻���ʣ���Ҫ�������롣��ʱ����ҪΪĿ����������� 
    // Host ������һ�����ʵ����ݿռ䡣
    if (dstmat->matData == NULL) {
        dstmat->matData = new float[srcmat->width * srcmat->height];
        if (dstmat->matData == NULL) {
            // ��������ڴ�Ĳ���ʧ�ܣ����ٱ�����ǰ��Ҫ���ɵ�Ŀ���������
            // �ָ���Ŀ������У��Ա�֤ϵͳ���µĲ��������ڻ��ҡ�
            *dstmatCud = olddstmatCud;
            return OUT_OF_MEM;
        }
    }

    // ���������ݴ�Դ�����п�����Ŀ������С�
    if (srcmatCud->deviceId < 0) {
        // ���Դ�������ݴ洢�� Host �ڴ棬��ֱ��ʹ�� C ��׼֧�ֿ��е� emcpy
        // ��ɿ�����

        // �� srcmat �ڵľ������ݿ����� dstmat �С�memcpy �����ش�����ˣ�û
        // �н��д����顣
        memcpy(dstmat->matData, srcmat->matData, 
               srcmat->width * srcmat->height * sizeof (float));

    } else {
        // ���Դ�������ݴ洢�� Device �ڴ棨�����ǵ�ǰ Device ���������� 
        // Device�������� 2D ��ʽ�Ŀ����������� Padding��
        cudaError_t cuerrcode;                     // CUDA ���÷��صĴ����롣

        // �����л��� srcmat �����������ڵ� Device���Է�������ڴ������
        cudaSetDevice(srcmatCud->deviceId);

        // ����ʹ�� cudaMemcpy2D �� srcmat �д��� Device �ϵ����ݿ����� dstmat
        // ��λ�� Host ���ڴ�ռ����棬�ÿ�����ͬʱ���� Padding��
        cuerrcode = cudaMemcpy2D(dstmat->matData,
                                 dstmatCud->pitchWords * sizeof (float),
                                 srcmat->matData,
                                 srcmatCud->pitchWords * sizeof (float),
                                 srcmat->width * sizeof (float),
                                 srcmat->height,
                                 cudaMemcpyDeviceToHost);
        if (cuerrcode != cudaSuccess) {
            // �����������ʧ�ܣ����ٱ����˳�ǰ����Ҫ���ɵ�Ŀ��������ݻָ���Ŀ
            // ������С����⣬������ݲ������õģ�����Ҫ�ͷ�����������ݿռ䣬
            // ��ֹ�ڴ�й©����󣬻���Ҫ�� Device �л����������������������ס�
            if (!reusedata)
                delete[] dstmat->matData;
            *dstmatCud = olddstmatCud;
            cudaSetDevice(curdevid);
            return CUDA_ERROR;
        }

        // ���ڴ������Ϻ󣬽��豸�л��ص�ǰ�� Device��
        cudaSetDevice(curdevid);
    }

    // ���˲����Ѿ�˵���µľ������ݿռ��Ѿ��ɹ������벢�������µ����ݣ���ˣ���
    // �����ݿռ��Ѻ����ô�������������ͷŵ��ɵ����ݿռ��Է�ֹ�ڴ�й©�����
    // ��Ϊ������ olddstmatCud �Ǿֲ������������Ӧ��Ԫ���ݻ��ڱ������˳����Զ�
    // �ͷţ�������ᡣ
    if (olddstmatCud.matMeta.matData != NULL) {
        if (olddstmatCud.deviceId > 0) {
            // ����������Ǵ洢�� Device �ڴ��ϵ����ݣ�����Ҫ���������ͷš�
            cudaSetDevice(olddstmatCud.deviceId);
            cudaFree(olddstmatCud.matMeta.matData);
            cudaSetDevice(curdevid);
        } else if (!reusedata) {
            // ��������ݾ��� Host �ڴ��ϣ������ reusedata δ��λ�����������
            // �ţ���Ϊһ����λ���ɵ����ݿռ�ͱ����ڳ����µ����ݣ������ͷš�
            delete[] olddstmatCud.matMeta.matData;
        }
    }

    // ������ϣ��˳���
    return NO_ERROR;
}

