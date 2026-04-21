cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(Aperture LANGUAGES CXX CUDA)

enable_language(CUDA)

Option(CUDA_SANITIZER OFF)

IF (WIN32)
    MESSAGE(STATUS "Now is windows")
    SET(CMAKE_CXX_FLAGS "/openmp /Ot")
    add_compile_definitions($<$<CXX_COMPILER_ID:MSVC>:NOMINMAX _CRT_SECURE_NO_WARNINGS>)
ELSE ()
    MESSAGE(STATUS "Now is linux")
    add_definitions(-DLINUX)
ENDIF ()

Option(GUI ON)
Option(RTX30XX OFF)
Option(TENSORCORE OFF)

if (GUI)
    add_definitions(-DGUI)
    find_package(GLEW)
    find_package(glfw3)
    add_subdirectory(Imgui)
	INCLUDE_DIRECTORIES(Imgui_INCLUDE_DIR)
endif()

if (TENSORCORE)
    add_definitions(-DENBALE_TENSOR)
endif()

file(GLOB_RECURSE CORE_HEADERS
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.hpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cuh")
file(GLOB_RECURSE CORE_SOURCES
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu")

source_group("Core Include" FILES ${CORE_HEADERS}) 
source_group("Core Source" FILES ${CORE_SOURCES})

include_directories(${PROJECT_SOURCE_DIR}/core)
  


add_executable(ApertureRenderer ${CORE_HEADERS} ${CORE_SOURCES} main.cu)
if (CUDA_SANITIZER)
  message(STATUS "CUDA sanitizer ENABLED")
  target_compile_options(ApertureRenderer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -G -O0>
  )
    target_compile_definitions(ApertureRenderer PRIVATE CUDA_SANITIZER_ENABLED=1)
    if (COMPUTE_SANITIZER)
        # memcheck
        add_custom_target(san_memcheck
          COMMAND ${CMAKE_COMMAND} -E env CUDA_LAUNCH_BLOCKING=1
                  "${COMPUTE_SANITIZER}" --error-exitcode=1 --tool=memcheck
                  "$<TARGET_FILE:ApertureRenderer>"
          DEPENDS ApertureRenderer
          USES_TERMINAL
          COMMENT "Compute Sanitizer: memcheck")

        # racecheck
        add_custom_target(san_racecheck
          COMMAND ${CMAKE_COMMAND} -E env CUDA_LAUNCH_BLOCKING=1
                  "${COMPUTE_SANITIZER}" --error-exitcode=1 --tool=racecheck
                  "$<TARGET_FILE:ApertureRenderer>"
          DEPENDS ApertureRenderer
          USES_TERMINAL
          COMMENT "Compute Sanitizer: racecheck")

        # initcheck
        add_custom_target(san_initcheck
          COMMAND ${CMAKE_COMMAND} -E env CUDA_LAUNCH_BLOCKING=1
                  "${COMPUTE_SANITIZER}" --error-exitcode=1 --tool=initcheck
                  "$<TARGET_FILE:ApertureRenderer>"
          DEPENDS ApertureRenderer
          USES_TERMINAL
          COMMENT "Compute Sanitizer: initcheck")

        # synccheck
        add_custom_target(san_synccheck
         COMMAND ${CMAKE_COMMAND} -E env CUDA_LAUNCH_BLOCKING=1
                "${COMPUTE_SANITIZER}" --error-exitcode=1 --tool=synccheck
                "$<TARGET_FILE:ApertureRenderer>"
        DEPENDS ApertureRenderer
        USES_TERMINAL
        COMMENT "Compute Sanitizer: synccheck")

        # all check
        add_custom_target(san_all
        DEPENDS san_memcheck san_racecheck san_initcheck san_synccheck)
    else()
        message(WARNING "compute-sanitizer not found; only compile flags applied.")
    endif()
else()
    message(STATUS "CUDA sanitizer disabled (default)")
endif()

target_include_directories(ApertureRenderer PRIVATE
    ${PROJECT_SOURCE_DIR}/core
)

set_target_properties(ApertureRenderer
PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)

target_compile_definitions(ApertureRenderer PRIVATE _HAS_STD_BYTE=0)


if (RTX30XX)
target_compile_options(ApertureRenderer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168 --extended-lambda>)
else()
target_compile_options(ApertureRenderer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168 --extended-lambda>)
endif()


if (GUI)
    target_link_libraries(ApertureRenderer GLEW::GLEW glfw Imgui) 
  	add_dependencies(ApertureRenderer Imgui)
endif()
