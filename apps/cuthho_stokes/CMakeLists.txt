add_executable(cuthho_stokes cuthho_stokes.cpp)
target_link_libraries(cuthho_stokes ${LINK_LIBS})
install(TARGETS cuthho_stokes RUNTIME DESTINATION bin)
