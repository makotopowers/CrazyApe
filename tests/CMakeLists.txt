# include(FetchContent)
# FetchContent_Declare(
#     googletest
#     GIT_REPOSITORY https://github.com/google/googletest.git
#     GIT_TAG release-1.11.0 # Adjust to the desired version
# )

# FetchContent_MakeAvailable(googletest)

# add_executable(test1 testing.cpp )
# target_include_directories(test1 PRIVATE ${CMAKE_SOURCE_DIR}/include)

# target_link_libraries(test1 gtest gtest_main)

# add_test(NAME test1 COMMAND test1)


#----------------------------- fetching ^ 


find_package(GTest REQUIRED)
include(GoogleTest)



add_executable(tests ${SOURCES} testing.cpp)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tests Eigen3::Eigen)
target_link_libraries(tests GTest::GTest GTest::Main)

gtest_discover_tests(tests)


#-------------------------- static ^ (faster)