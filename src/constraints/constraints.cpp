#include "constraints.hpp"
#include <vector>
#include <set>
#include <tsl/robin_map.h>
#include <math.h>
#include <igl/per_vertex_normals.h>

Constraints::Constraints(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, Eigen::MatrixXi& productspace, const int numContours, Eigen::MatrixXi& SRCIds, Eigen::MatrixXi& TRGTIds, Eigen::MatrixXi& PLUSMINUSDIR, bool coupling, bool resolveCoupling, bool meanProblem) :
    EX(EX), EY(EY), productspace(productspace), numContours(numContours), SRCIds(SRCIds), TRGTIds(TRGTIds), PLUSMINUSDIR(PLUSMINUSDIR), coupling(coupling), resolveCoupling(resolveCoupling), meanProblem(meanProblem) {
    //nVX = EX.maxCoeff()+1;
    numCouplingConstraints = 0;
}


std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> Constraints::getConstraints() {

    const long nnzVertexEdgeAdjacency = 2 * SRCIds.rows();
    long nnzConstraints = coupling ? 2 * PLUSMINUSDIR.rows() + nnzVertexEdgeAdjacency : nnzVertexEdgeAdjacency;
    if (!meanProblem) nnzConstraints += productspace.rows();

    Eigen::MatrixXi I(nnzConstraints, 1); I.setZero();
    Eigen::MatrixXi J(nnzConstraints, 1); J.setZero();
    Eigen::MatrixXi V(nnzConstraints, 1); V.setZero();

    //Eigen::MatrixXi indexMap;
    if (resolveCoupling) {
        std::cout << "[CONSTR] Resolving coupling" << std::endl;
        couplingDecoder = Eigen::MatrixXi(SRCIds.rows(), 1);
        //indexMap = Eigen::MatrixXi(SRCIds.rows(), 1);

        utils::setLinspaced(couplingDecoder, 0);
        for (int i = 0; i < PLUSMINUSDIR.rows(); i++) {
            const int plusEdgeIdx = PLUSMINUSDIR(i, 0);
            couplingDecoder(plusEdgeIdx) = plusEdgeIdx;

            const int minusEdgeIdx = PLUSMINUSDIR(i, 1);
            couplingDecoder(minusEdgeIdx) = plusEdgeIdx;
        }
        int nonZeroColCounter = 0;
        couplingEncoder = Eigen::MatrixXi(SRCIds.rows(), 1);
        for (int i = 0; i < couplingEncoder.rows(); i++) {
            if (couplingDecoder(i) == i) {
                couplingEncoder(nonZeroColCounter) = couplingDecoder(i);
                couplingDecoder(i) = nonZeroColCounter;
                nonZeroColCounter++;
            }
            else {
                if (i < couplingDecoder(i))std::cout << "coupling decoder not monotonic, this should not happen" << std::endl;
                couplingDecoder(i) = couplingDecoder(couplingDecoder(i));
            }
        }
        couplingEncoder.conservativeResize(nonZeroColCounter, 1);
        std::cout << "[CONSTR] new nonzero num cols " << nonZeroColCounter << std::endl;
    }


    long idx = 0;
    for (long e = 0; e < SRCIds.rows(); e++) {
        I(idx) = SRCIds(e, 0);
        J(idx) = resolveCoupling ? couplingDecoder(e) : e;
        V(idx) = 1;
        idx++;

        I(idx) = TRGTIds(e, 0);
        J(idx) = resolveCoupling ? couplingDecoder(e) : e;
        V(idx) = -1;
        idx++;
    }

    int rowIdx = SRCIds.maxCoeff() + 1;
    if (coupling && !resolveCoupling) {
        for (int i = 0; i < PLUSMINUSDIR.rows(); i++) {
            const int plusEdgeIdx = PLUSMINUSDIR(i, 0);
            I(idx) = rowIdx;
            J(idx) = plusEdgeIdx;
            V(idx) = 1;
            idx++;

            const int minusEdgeIdx = PLUSMINUSDIR(i, 1);
            I(idx) = rowIdx;
            J(idx) = minusEdgeIdx;
            V(idx) = -1;
            idx++;

            rowIdx++;
        }
    }

    if (!meanProblem) {
        tsl::robin_map<EDG, int> edgeToIndex;
        edgeToIndex.reserve(EY.rows());

        int ey_edge_idx = 0;
        for (int e = 0; e < EY.rows(); e++) {
            if (EY(e, 0) == -1) continue;
            EDG edge0(EY.row(e));
            if (edgeToIndex.find(-edge0) != edgeToIndex.end()) {
                if (!resolveCoupling) {
                    edgeToIndex.insert({edge0, ey_edge_idx});
                    ey_edge_idx++;
                }
                else {
                    continue;
                }
            }
            else {
                edgeToIndex.insert({edge0, ey_edge_idx});
                ey_edge_idx++;
            }
        }

        int maxIdx = 0;
        for (int i = 0; i < productspace.rows(); i++) {
            //continue;
            EDG eeedge = EDG(productspace(i, 0), productspace(i, 1));
            if (edgeToIndex.find(eeedge) == edgeToIndex.end()) {
                continue;
            }
            if (productspace(i, 0) == productspace(i, 1))
                std::cout << "This should not happen" << std::endl;
            const int edgeidx = edgeToIndex[eeedge];
            //std::cout << edgeidx << std::endl;
            maxIdx = std::max(maxIdx, edgeidx);
            I(idx) = rowIdx + edgeidx;
            J(idx) = resolveCoupling ? couplingDecoder(i) : i;
            V(idx) = 1;
            idx++;
        }
        RHS = Eigen::MatrixXi(rowIdx + maxIdx + 1, 1);
        RHS.setZero();
        RHS.block(rowIdx, 0, maxIdx+1, 1).setOnes();
        rowIdx += maxIdx+1;
    }
    else {
        RHS = Eigen::MatrixXi(rowIdx, 1);
        RHS.setZero();
    }

    //std::cout << J.maxCoeff() << std::endl;
    I.conservativeResize(idx, 1);
    J.conservativeResize(idx, 1);
    V.conservativeResize(idx, 1);
    return std::make_tuple(I, J, V);
}


std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> Constraints::getLeqConstraints(const int maxDepth) {
    const bool debug = false;
    Eigen::MatrixXi debug_mat;
    if (debug) {
        debug_mat = Eigen::MatrixXi(productspace.rows(), 1);
        debug_mat.setConstant(-1);
    }
    const int numVerticesX = EX.maxCoeff()+1;
    const int nnzConstraints = 10 * EY.rows() * (EX.rows() + numVerticesX) * std::pow(maxDepth / 2.0 + 1.0, 2.0);

    int num_curves = 1;
    for (int i = 0; i < EY.rows(); i++) {
        if (EY(i, 0) == -1)
            num_curves++;
    }

    Eigen::MatrixXi I(nnzConstraints, 1); I.setZero();
    Eigen::MatrixXi J(nnzConstraints, 1); J.setZero();
    Eigen::MatrixXi V(nnzConstraints, 1); V.setZero();

    long numAddedLeq = 0;

    int curveIdx = 0;
    int idxInCurve = 0;
    int idxEX0 = EY(0, 0);
    int idxEX1 = EY(0, 1);
    for (long i = 0; i < productspace.rows(); i++) {
        //std::cout << i << " " << curveIdx << " " << idxInCurve <<  ": " << productspace.row(i) << std::endl;
        // if degenerate 3d edge we do not add this to the constraint since we can "stay in the 3d vertex for multiple consecutive layers"
        if (productspace(i, 3) == productspace(i, 2)) {
            continue;
        }
        if (productspace(i, 0) == -1) {
            continue;
        }

        curveIdx = productspace(i, 6);

        // check if we are still in this curve or not
        /* this code is shit paul and caused headaches...
        if (productspace(i, 1) != productspace(i, 0)) {
            if (idxEX0 != productspace(i, 0) && idxEX1 != productspace(i, 1)) {
                //std::cout << "     " << idxEX0  << " " <<  productspace(i, 0)  << " " <<idxEX1  << " " <<productspace(i, 1) << std::endl;
                idxInCurve++;
                if (EY(idxInCurve, 0) == -1) {
                    //curveIdx++;
                    idxInCurve++;
                }
                idxEX0 = EY(idxInCurve, 0);
                idxEX1 = EY(idxInCurve, 1);
            }
        }
        */

        const long outNodeIdx = productspace(i, 3);

        // out
        I(numAddedLeq, 0) = (int) outNodeIdx + curveIdx * numVerticesX;
        J(numAddedLeq, 0) = (int) i;
        V(numAddedLeq, 0) = 1;
        if (debug) {
            debug_mat(i, 0) = numAddedLeq;
        }
        numAddedLeq++;

        if (idxInCurve > EY.rows()) {
            std::cout << "[CONSTR] error idxInCurve too large" << std::endl;
            break;
        }

        if (curveIdx > num_curves) {
            std::cout << "[CONSTR] error curveIdx too large" << std::endl;
            break;
        }

        if (numAddedLeq > nnzConstraints) {
            std::cout << "[CONSTR] error numAddedLeq > nnzLeqConstraints " << numAddedLeq << " " << nnzConstraints << std::endl;
            break;
        }

    }

    if (debug) {
        std::vector< Eigen::Triplet<int>> constrEntries;
        constrEntries.reserve(numAddedLeq+10);
        for (int i = 0; i < numAddedLeq; i++) {
            constrEntries.push_back(Eigen::Triplet<int>(I(i), J(i), V(i)));
        }
        Eigen::SparseMatrix<int, Eigen::RowMajor> temp(I.maxCoeff()+1, productspace.rows());
        temp.setFromTriplets(constrEntries.begin(), constrEntries.end());

        for (int j = 0; j < temp.rows(); j++) {
            int v_x = -1;
            for (typename Eigen::SparseMatrix<int,  Eigen::RowMajor>::InnerIterator it(temp, 0); it; ++it) {
                const auto pedge = productspace.row(it.index());
                if (v_x == -1) {
                    v_x = pedge(3);
                }
                if ( pedge(3) != v_x) {
                    std::cout << "big problem, non intersection constraints wrong, pedge(3) != v_x " << pedge(3) << " != " << v_x << std::endl;
                }
            }
        }


        for (int i = 0; i < productspace.rows(); i++) {
            if (productspace(i, 2) != productspace(i, 3)) {
                if (debug_mat(i) == -1) {
                    std::cout << "error, no entry for productedge" << std::endl;
                    std::cout << productspace.row(i) << std::endl;
                }
            }
        }
    }

    //std::cout << EY << std::endl;

    //std::cout << idxInCurve << ", " << curveIdx << std::endl;

    I.conservativeResize(numAddedLeq, 1);
    J.conservativeResize(numAddedLeq, 1);
    V.conservativeResize(numAddedLeq, 1);
    return std::make_tuple(I, J, V);
}

typedef std::tuple<int, int> Edge;

namespace std {
    template<> struct hash<Edge> {
        std::size_t operator()(Edge const& edg) const noexcept {
            return std::get<0>(edg) + 10000 + std::get<1>(edg); // this hash function should work great for our purpose
        }
    };
    template<> struct equal_to<Edge>{
        constexpr bool operator()(const Edge &lhs, const Edge &rhs) const {
            return (std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs));
        }
    };
}


std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> Constraints::getOrientVectors(const Eigen::MatrixXd& VX,
                                                                                            const Eigen::MatrixXd& VY,
                                                                                            const Eigen::MatrixXi& FX,
                                                                                            const Eigen::MatrixXi& FY,
                                                                                            const int maxDepth,
                                                                                            const bool angleBased) {
    const double angleThresholdNormalInversion = M_PI_2;
    //if (maxDepth != 0) {
    //     std::cout << prefix << "Cannot create orientation constraints for maxDepth > 0" << std::endl;
    //}
    const long nnzConstraints = 100 * productspace.rows() * std::pow(maxDepth + 1, 2);
    Eigen::MatrixXi I(nnzConstraints, 2); I.setZero();
    Eigen::MatrixXi J(1, 1); J.setZero();
    Eigen::MatrixXi V(1, 1); V.setZero();

    tsl::robin_map<Edge, Edge> next_edge;
    next_edge.reserve(3 * FX.rows());
    for (int f = 0; f < FX.rows(); f++) {
        next_edge[std::make_tuple(FX(f, 0), FX(f, 1))] = std::make_tuple(FX(f, 1), FX(f, 2));
        next_edge[std::make_tuple(FX(f, 1), FX(f, 2))] = std::make_tuple(FX(f, 2), FX(f, 0));
        next_edge[std::make_tuple(FX(f, 2), FX(f, 0))] = std::make_tuple(FX(f, 0), FX(f, 1));
    }

    Eigen::MatrixXd NX, NY;
    igl::per_vertex_normals(VX, FX, NX);
    igl::per_vertex_normals(VY, FY, NY);

    long numAdded = 0;
    int currentCurveIndex = 0, currentCurveStartIndex = 0;
    for (int i = 0; i < productspace.rows(); i++) {
        if (productspace(i, 6) != currentCurveIndex) {
            currentCurveIndex = productspace(i, 6);
            currentCurveStartIndex = i;
        }
        if (productspace(i, 0) == productspace(i, 1) || productspace(i, 2) == productspace(i, 3)) {
            continue; // degenerate edges cannot have normal direction
        }
        const int thisEdgeSourceVertex = productspace(i, 2);

        Eigen::Vector3d edgeXi;
        std::tuple<int, int> nextEdge;
        if (angleBased) {
            edgeXi = VX.row(productspace(i, 3)) - VX.row(productspace(i, 2));
            edgeXi.normalize();
        }
        else {
            const auto result = next_edge.find(std::make_tuple(productspace(i, 2), productspace(i, 3)));
            if (result == next_edge.end()) {
                nextEdge = std::make_tuple(-1, -1);
            }
            else {
                nextEdge = result.value();
            }
        }

        for (int j = currentCurveStartIndex; j < productspace.rows(); j++) {
            if (productspace(j, 6) != currentCurveIndex) {
                break;
            }
            if (productspace(j, 0) == productspace(j, 1) || productspace(j, 2) == productspace(j, 3)) {
                continue; // degenerate edges cannot have normal direction
            }
            if (productspace(i, 1) != productspace(j, 0) || productspace(i, 3) != productspace(j, 2) ) {
                continue;
            }
            if (productspace(i, 2) == productspace(j, 3)) {
                // more or less tri to edge matchings here
                continue;
            }


            //std::cout << productspace.row(i) << "  <-> " << productspace.row(j) << std::endl;
            const int connectingVertex = productspace(i, 3);
            assert(productspace(j, 2) == connectingVertex);

            if (angleBased) {
                Eigen::Vector3d edgeXj = VX.row(productspace(j, 3)) - VX.row(productspace(j, 2));
                edgeXj.normalize();

                Eigen::Vector3d cross = edgeXi.cross(edgeXj);
                const double crossNorm = cross.norm();
                if (crossNorm < 1e-4) {
                    continue;
                }
                cross = cross / crossNorm;

                const double angle = std::abs(acos(cross.dot(NX.row(connectingVertex))));

                if (angle > angleThresholdNormalInversion) {
                    I.row(numAdded) << i, j;
                    numAdded++;
                }
            }
            else {
                const int nextEdgeTargetVertex = std::get<1>(nextEdge);
                if (nextEdgeTargetVertex != productspace(j, 3)) {
                    I.row(numAdded) << i, j;
                    numAdded++;
                }
            }
        }
    }

    I.conservativeResize(numAdded, 2);
    //J.conservativeResize(numAdded, 1);
    //V.conservativeResize(numAdded, 1);
    return std::make_tuple(I, J, V);
}


Eigen::MatrixXi Constraints::getRHS() {
    return RHS;
}

int Constraints::getNumCouplingConstr() {
    return numCouplingConstraints;
}

Eigen::MatrixXi Constraints::getCouplingDecoder() {
    return couplingDecoder;
}
Eigen::MatrixXi Constraints::getCouplingEncoder() {
    return couplingEncoder;
}
