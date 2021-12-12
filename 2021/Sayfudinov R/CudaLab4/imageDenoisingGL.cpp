
// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imageDenoising.h"

// includes, project
#include <helper_functions.h> // includes for helper utility functions
#include <helper_cuda.h>      // includes for cuda error checking and initialization

const char *sSDKsample = "CUDA ImageDenoising";

const char *filterMode[] =
{
    "Passthrough",
    "KNN method",
    NULL
};

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "image_passthru.ppm",
    "image_knn.ppm",
   
    NULL
};

const char *sReference[] =
{
    "ref_passthru.ppm",
    "ref_knn.ppm",
    NULL
};

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//Source image on the host side
uchar4 *h_Src;
int imageW, imageH;
GLuint shader;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int  g_Kernel = 0;
bool    g_FPS = false;
bool   g_Diag = false;
StopWatchInterface *timer = NULL;

//Algorithms global parameters
const float noiseStep = 0.025f;
const float  lerpStep = 0.025f;
static float knnNoise = 0.32f;
static float nlmNoise = 1.45f;
static float    lerpC = 0.2f;


const int frameN = 24;
int frameCounter = 0;


#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc   = NULL;
char **pArgv = NULL;

#define MAX_EPSILON_ERROR 5
#define REFRESH_DELAY     10 //ms

void cleanup();

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "<%s>: %3.1f fps", filterMode[g_Kernel], ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        //fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

void runImageFilters(TColor *d_dst)
{
    switch (g_Kernel)
    {
        case 0:
            cuda_Copy(d_dst, imageW, imageH, texImage);
            break;

        case 1:
            if (!g_Diag)
            {
                cuda_KNN(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC, texImage);
            }
            else
            {
                cuda_KNNdiag(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC, texImage);
            }

            break;

            }

    getLastCudaError("Filtering kernel execution failed.\n");
}


void displayFunc(void)
{ 
    sdkStartTimer(&timer);
    TColor *d_dst = NULL;
    size_t num_bytes;

    if (frameCounter++ == 0)
    {
        sdkResetTimer(&timer);
    }

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    getLastCudaError("cudaGraphicsMapResources failed");
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));
    getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

    runImageFilters(d_dst);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // Common display code path
    {
        glClear(GL_COLOR_BUFFER_BIT);

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
        glBegin(GL_TRIANGLES);
        glTexCoord2f(0, 0);
        glVertex2f(-1, -1);
        glTexCoord2f(2, 0);
        glVertex2f(+3, -1);
        glTexCoord2f(0, 2);
        glVertex2f(-1, +3);
        glEnd();
        glFinish();
    }

    if (frameCounter == frameN)
    {
        frameCounter = 0;

        if (g_FPS)
        {
            printf("FPS: %3.1f\n", frameN / (sdkGetTimerValue(&timer) * 0.001));
            g_FPS = false;
        }
    }

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}


void keyboard(unsigned char k, int /*x*/, int /*y*/)
{
    switch (k)
    {
        case 27:
        case 'q':
        case 'Q':
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif

        case '1':
            printf("Passthrough.\n");
            g_Kernel = 0;
            break;

        case '2':
            printf("KNN method \n");
            g_Kernel = 1;
            break;
    
    }
}


int initGL(int *argc, char **argv)
{
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(512 - imageW / 2, 384 - imageH / 2);
    glutCreateWindow(argv[0]);
    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    printf("OpenGL window created.\n");

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

    if (!isGLVersionSupported(1,5) ||
        !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        fflush(stderr);
        return false;
    }

    return 0;
}

// shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

void initOpenGLBuffers()
{
    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src, GL_STREAM_COPY);
    //While a PBO is registered to CUDA, it can't be used
    //as the destination for OpenGL drawing calls.
    //But in our particular case OpenGL is only used
    //to display the content of the PBO, specified by CUDA kernels,
    //so we need to register/unregister it only once.
    // DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(gl_PBO) );
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
                                                 cudaGraphicsMapFlagsWriteDiscard));
    GLenum gl_error = glGetError();

    if (gl_error != GL_NO_ERROR)
    {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        char tmpStr[512];
        // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
        // when the user double clicks on the error line in the Output pane. Like any compile error.
        sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", __FILE__, __LINE__, gluErrorString(gl_error));
        OutputDebugString(tmpStr);
#endif
        fprintf(stderr, "GL Error in file '%s' in line %d :\n", __FILE__, __LINE__);
        fprintf(stderr, "%s\n", gluErrorString(gl_error));
        exit(EXIT_FAILURE);
    }

    printf("PBO created.\n");

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}


void cleanup()
{
    free(h_Src);
    checkCudaErrors(CUDA_FreeArray());
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

    glDeleteProgramsARB(1, &shader);

    sdkDeleteTimer(&timer);
}

void runAutoTest(int argc, char **argv, const char *filename, int kernel_param)
{
    printf("[%s] - (automated testing w/ readback)\n", sSDKsample);

    int devID = findCudaDevice(argc, (const char **)argv);

    // First load the image, so we know what the size of the image (imageW and imageH)
    printf("Allocating host and CUDA memory and loading image file...\n");
    const char *image_path = sdkFindFilePath("portrait_noise.bmp", argv[0]);

    if (image_path == NULL)
    {
        printf("imageDenoisingGL was unable to find and load image file <portrait_noise.bmp>.\nExiting...\n");
        exit(EXIT_FAILURE);
    }

    LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
    printf("Data init done.\n");

    checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

    TColor *d_dst = NULL;
    unsigned char *h_dst = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_dst, imageW*imageH*sizeof(TColor)));
    h_dst = (unsigned char *)malloc(imageH*imageW*4);

    {
        g_Kernel = kernel_param;
        printf("[AutoTest]: %s <%s>\n", sSDKsample, filterMode[g_Kernel]);

        runImageFilters(d_dst);

        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW*imageH*sizeof(TColor), cudaMemcpyDeviceToHost));
        sdkSavePPM4ub(filename, h_dst, imageW, imageH);
    }

    checkCudaErrors(CUDA_FreeArray());
    free(h_Src);

    checkCudaErrors(cudaFree(d_dst));
    free(h_dst);

    printf("\n[%s] -> Kernel %d, Saved: %s\n", sSDKsample, kernel_param, filename);

    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}


int main(int argc, char** argv)
{
    char* dump_file = NULL;


    pArgc = &argc;
    pArgv = argv;

    printf("%s Starting...\n\n", sSDKsample);

    printf("[%s]\n", sSDKsample);
   
    // First load the image, so we know what the size of the image (imageW and imageH)
    printf("Allocating host and CUDA memory and loading image file...\n");
    const char* image_path = sdkFindFilePath("portrait_noise.bmp", argv[0]);

    if (image_path == NULL)
    {
        printf("imageDenoisingGL was unable to find and load image file <portrait_noise.bmp>.\nExiting...\n");
        exit(EXIT_FAILURE);
    }

    LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
    printf("Data init done.\n");

    initGL(&argc, argv);
    findCudaDevice(argc, (const char**)argv);

    checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

    initOpenGLBuffers();

    printf("Press [1] to view noisy image\n");
    printf("Press [2] to view image restored with knn filter\n");
  
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    glutMainLoop();
}
