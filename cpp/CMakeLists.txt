cmake_minimum_required(VERSION 3.10)
project(HomomorphicLogisticRegression)

# Use C++17
set(CMAKE_CXX_STANDARD 17)

# ==================== ENVIRONMENT VARIABLES ====================
# SEAL paths
if(NOT DEFINED ENV{SEAL_SRC_DIR})
    message(FATAL_ERROR "SEAL_SRC_DIR is not set!")
endif()

if(NOT DEFINED ENV{SEAL_BUILD_DIR})
    message(FATAL_ERROR "SEAL_BUILD_DIR is not set!")
endif()

# GSL path
if(NOT DEFINED ENV{GSL_INCLUDE_DIR})
    message(FATAL_ERROR "GSL_INCLUDE_DIR is not set!")
endif()

# ==================== INCLUDE DIRECTORIES ====================
include_directories(
    $ENV{SEAL_SRC_DIR}
    $ENV{SEAL_BUILD_DIR}/native/src
    $ENV{GSL_INCLUDE_DIR}
)

# ==================== LINK DIRECTORIES ====================
link_directories(
    $ENV{SEAL_BUILD_DIR}/lib
)

# ==================== EXECUTABLE ====================
add_executable(HomomorphicLogisticRegression
    main.cpp
    HomomorphicLogisticRegression.cpp
)

# ==================== LINK LIBRARIES ====================
# Notice we use 'seal-4.1' because your file is libseal-4.1.a
target_link_libraries(HomomorphicLogisticRegression
    seal-4.1
)