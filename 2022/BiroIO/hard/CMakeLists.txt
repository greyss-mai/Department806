CMAKE_MINIMUM_REQUIRED( VERSION 3.10 )
SET( PROJECT_NAME "Lab_1" )

PROJECT( ${PROJECT_NAME} )


# ------------------------------------------------------------------------
#                    Building statuses.
# ------------------------------------------------------------------------
MESSAGE( STATUS "Using c compiler at path: ${CMAKE_C_COMPILER}")
MESSAGE( STATUS "Using cxx compiler at path: ${CMAKE_CXX_COMPILER}")
MESSAGE( STATUS "CPU Version mode: ${LAB_CPU}")
MESSAGE( STATUS "GPU Version mode: ${LAB_GPU}")


INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )


# ------------------------------------------------------------------------

# Find CUDA library
INCLUDE( FindCUDA )
FIND_PACKAGE( CUDA REQUIRED )
INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS} )

# Show CUDA found status
MESSAGE( STATUS "CUDA version: ${CUDA_VERSION_STRING}" )
MESSAGE( STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}" )
MESSAGE( STATUS "CUDA rt library: ${CUDA_CUDART_LIBRARY}" )
MESSAGE( STATUS "CUDA nvcc: ${CUDA_NVCC_EXECUTABLE}" )

SET( CUDA_NVCC_FLAGS "-arch=sm_30" )
SET( CUDA_PROPAGATE_HOST_FLAGS OFF )    

ADD_EXECUTABLE(${PROJECT_NAME} 
    "main.cpp"
    "cpu_impl.cpp"
    "gpu_impl.cu"
)

SET( GPU_LIB_NAME "GPU_IMPL" )
SET( CUDA_SOURCES "gpu_impl.cu" )

CUDA_ADD_LIBRARY( ${GPU_LIB_NAME} ${CUDA_SOURCES} STATIC )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${GPU_LIB_NAME} )