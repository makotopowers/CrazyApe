cmake_minimum_required(VERSION 3.27)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(algorithms)



set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clang-tidy

include(CTest)
enable_testing()

find_package (Eigen3 REQUIRED NO_MODULE)

set(HEADERS
    ${PROJECT_SOURCE_DIR}/include/ridgeRegression.hpp
    ${PROJECT_SOURCE_DIR}/include/OLS.hpp
    ${PROJECT_SOURCE_DIR}/include/lasso.hpp
    ${PROJECT_SOURCE_DIR}/include/PCA.hpp
    ${PROJECT_SOURCE_DIR}/include/makeData.hpp
    ${PROJECT_SOURCE_DIR}/include/utilities.hpp
    ${PROJECT_SOURCE_DIR}/include/configReader.hpp



    )

set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/ridgeRegression.cpp
    ${PROJECT_SOURCE_DIR}/src/OLS.cpp
    ${PROJECT_SOURCE_DIR}/src/lasso.cpp
    ${PROJECT_SOURCE_DIR}/src/PCA.cpp
    ${PROJECT_SOURCE_DIR}/src/makeData.cpp
    ${PROJECT_SOURCE_DIR}/src/utilities.cpp
    ${PROJECT_SOURCE_DIR}/src/configReader.cpp
    )

add_executable(${PROJECT_NAME}
               ${SOURCES}
               main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
# target_link_libraries(${PROJECT_NAME} 

add_subdirectory(tests)