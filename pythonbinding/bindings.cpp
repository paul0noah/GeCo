//
//  ShapeMatchModelPyBinds.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 27.06.22.
//

#include "src/product_graph_generators.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "helper/utils.hpp"


namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT

void notimplemented() {
    std::cout << "TODO: IMPL" << std::endl;
}

Eigen::MatrixXi getSurfaceCycles(const Eigen::MatrixXi& F) {
    Eigen::MatrixXi surfaceCycles(F.rows() * 4, 2);
    int idx = 0;
    for (int f = 0; f < F.rows(); f++) {
        surfaceCycles(idx, 0) = F(f, 0);
        surfaceCycles(idx, 1) = F(f, 1);
        idx++;
        surfaceCycles(idx, 0) = F(f, 1);
        surfaceCycles(idx, 1) = F(f, 2);
        idx++;
        surfaceCycles(idx, 0) = F(f, 2);
        surfaceCycles(idx, 1) = F(f, 0);
        idx++;
        surfaceCycles.row(idx) << -1, -1;
        idx++;
    }
    surfaceCycles.conservativeResize(idx-1, 2); // remove last -1 element
    return surfaceCycles;
}

PYBIND11_MODULE(geco, handle) {
    handle.doc() = "geco python bindings";

    handle.def("get_surface_cycles", &getSurfaceCycles);

    py::class_<ProductGraphGenerators, std::shared_ptr<ProductGraphGenerators>> smm(handle, "product_graph_generator");
    smm.def(py::init<Eigen::MatrixXd&, Eigen::MatrixXi&, Eigen::MatrixXd&, Eigen::MatrixXi&, Eigen::MatrixXd&>());

    smm.def("generate", &ProductGraphGenerators::generate);
    smm.def("get_cost_vector", &ProductGraphGenerators::getCostVector);
    smm.def("update_robust_loss_params", &ProductGraphGenerators::updateRobustLossParams);
    smm.def("get_constraint_matrix_vectors", &ProductGraphGenerators::getAVectors);
    smm.def("convert_matching_to_surface_matching", &ProductGraphGenerators::convertEdgeMatching2CycleMatching);
    smm.def("compute_elastic_energy", &ProductGraphGenerators::computeElasticEnergy);
    smm.def("get_rhs", &ProductGraphGenerators::getRHS);
    smm.def("get_constraint_matrix_vectors_intersections", &ProductGraphGenerators::getAleqVectors);

    smm.def("get_product_space", &ProductGraphGenerators::getProductSpace);
    smm.def("decode_result_vector", &ProductGraphGenerators::decodeResultVector);

    smm.def("export", pybind11::overload_cast<>(&ProductGraphGenerators::exportInputs));
    smm.def("export", pybind11::overload_cast<Eigen::MatrixX<bool>>(&ProductGraphGenerators::exportInputs));

    smm.def("set_max_depth", &ProductGraphGenerators::setMaxDepth);
    smm.def("set_resolve_coupling", &ProductGraphGenerators::setResolveCoupling);
    smm.def("set_mean_problem", &ProductGraphGenerators::setMeanProblem);


#if WITH_OR_TOOLS
    smm.def("solve_with_or_tools", &ProductGraphGenerators::solveWithORTools);
#endif
}
