message("Retrieving ortools...")
Set(FETCHCONTENT_QUIET FALSE)
include(FetchContent)
FetchContent_Declare(
    ortools
    GIT_REPOSITORY https://github.com/google/or-tools.git
    GIT_TAG 6fc1930
    GIT_PROGRESS TRUE
)
set(BUILD_DEPS ON)
set(USE_COINOR OFF)
set(USE_SCIP OFF)
set(BUILD_SCIP OFF)
set(BUILD_DOC OFF)
set(BUILD_FLATZINC OFF)
set(BUILD_SAMPLES OFF)
set(BUILD_EXAMPLES OFF)
FetchContent_MakeAvailable(ortools)
Set(FETCHCONTENT_QUIET TRUE)
