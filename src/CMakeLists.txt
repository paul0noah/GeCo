add_subdirectory(constraints)
add_subdirectory(product_spaces)
add_subdirectory(energy)

target_sources(${PROJECT_NAME} 
				PRIVATE
					product_graph_generators.cpp
					product_graph_generators.hpp
	)
