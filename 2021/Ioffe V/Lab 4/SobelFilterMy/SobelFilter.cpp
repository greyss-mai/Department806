#pragma region includes

//Инклюды библиотек
// OpenGL Graphics includes
#include <helper_gl.h>

#include <GL/freeglut.h>


// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "SobelFilter_kernels.h"

// includes, project
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking

#pragma endregion

#pragma region vars



//То что будет отображаться в шапке при выборе с клавиатуры
const char *filterMode[] =
{
    "No Filtering",
    "Sobel",
    NULL
};

//как часто обновлять экран
#define REFRESH_DELAY     10 //ms

//Нзавание окна
const char *sSDKsample = "CUDA Sobel operator";

//параметры окна и картинки
static int wWidth   = 512; // Window width
static int wHeight  = 512; // Window height
static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height

//таймер для openGL
StopWatchInterface *timer = NULL;


//переменные для работы с отображением
static GLuint pbo_buffer = 0;  // буфер изображения
struct cudaGraphicsResource *cuda_pbo_resource; // буфер изображения CUDA

static GLuint texid = 0;       // айди текстуры
Pixel *pixels = NULL;  // изображение на компе

//текущий режим отображения изображения
enum SobelDisplayMode g_SobelDisplayMode;

#pragma endregion

#pragma region OpenGL functions

//инициализация openGL
void initGL(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("CUDA Edge Detection");

    if (!isGLVersionSupported(1, 5) ||
        !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This code requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_FAILURE);
    }
}

// Функция осуществляющее рендер изображения
void display(void)
{
    sdkStartTimer(&timer);

    // Sobel operation
    Pixel *data = NULL;

    // map Pixel Buffer Object to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&data, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    //Функция которая показыват изображение, полученное либо напрямую, либо пропущенное через фильтр
    sobelFilter(data, imWidth, imHeight, g_SobelDisplayMode);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texid);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imWidth, imHeight,
                    GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glTexCoord2f(0, 0);
    glVertex2f(0, 1);
    glTexCoord2f(1, 0);
    glVertex2f(1, 1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 0);
    glTexCoord2f(0, 1);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glutSwapBuffers();

    sdkStopTimer(&timer);
}

//соытие таймера
void timerEvent(int value)
{
    if(glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

//обработчик нажатия на клавиатуру
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    char temp[256];

    switch (key)
    {
        case 27:
        case 'q':
        case 'Q':
            printf("Shutting down...\n");
           
            glutDestroyWindow(glutGetWindow());
            return;
            break;

        case 'i':
        case 'I':
            g_SobelDisplayMode = SOBELDISPLAY_IMAGE;
            sprintf(temp, "CUDA Edge Detection (%s)", filterMode[g_SobelDisplayMode]);
            glutSetWindowTitle(temp);
            break;

        case 't':
        case 'T':
            g_SobelDisplayMode = SOBELDISPLAY_SOBELTEX;
            sprintf(temp, "CUDA Edge Detection (%s)", filterMode[g_SobelDisplayMode]);
            glutSetWindowTitle(temp);
            break;

        default:
            break;
    }
}

//фунция обработки смены размера окна
void reshape(int x, int y)
{
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

//dispose
void cleanup(void)
{
    cudaGraphicsUnregisterResource(cuda_pbo_resource);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo_buffer);
    glDeleteTextures(1, &texid);
    deleteTexture();

    sdkDeleteTimer(&timer);

}

#pragma endregion




//открывает и подготавливает картинку из файла и записывает ее в буфер
void initializeData(char* file)
{
    //открываем картинку
    GLint bsize;
    unsigned int w, h;
    size_t file_length = strlen(file);

    if (!strcmp(&file[file_length - 3], "pgm"))
    {
        if (sdkLoadPGM<unsigned char>(file, &pixels, &w, &h) != true)
        {
            printf("Failed to load PGM image file: %s\n", file);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        exit(EXIT_FAILURE);
    }

    imWidth = (int)w;
    imHeight = (int)h;
    setupTexture(imWidth, imHeight, pixels);

    memset(pixels, 0x0, sizeof(Pixel) * imWidth * imHeight);


    // сохраняем в буферы
    glGenBuffers(1, &pbo_buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
        sizeof(Pixel) * imWidth * imHeight,
        pixels, GL_STREAM_DRAW);

    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

    if ((GLuint)bsize != (sizeof(Pixel) * imWidth * imHeight))
    {
        printf("Buffer object (%d) has incorrect size (%d).\n", (unsigned)pbo_buffer, (unsigned)bsize);

        exit(EXIT_FAILURE);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // связываем созданный буфер с CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard));

    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, (GL_LUMINANCE),
        imWidth, imHeight, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

}


//ищет заданную нами картинку среди файлов
void loadDefaultImage(char* loc_exec)
{

    printf("Reading image: bird.pgm\n");
    const char* image_filename = "bird.pgm";
    char* image_path = sdkFindFilePath(image_filename, loc_exec);

    if (image_path == NULL)
    {
        printf("Failed to read image file: <%s>\n", image_filename);
        exit(EXIT_FAILURE);
    }

    initializeData(image_path);
    free(image_path);
}

int main(int argc, char **argv)
{

    printf("%s Starting...\n\n", sSDKsample);

    initGL(&argc, argv);
    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    loadDefaultImage(argv[0]);


    printf("I: display Image (no filtering)\n");
    printf("T: display Sobel Edge Detection (Using Texture)\n");
    fflush(stdout);

    glutCloseFunc(cleanup);

    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    glutMainLoop();
}
