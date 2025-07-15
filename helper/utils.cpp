//
//  utils.cpp
//  helper
//
//  Created by Paul Rötzer on 11.04.21.
//

#include <Eigen/Dense>
#include <math.h>
#include <cmath>
#include "utils.hpp"
#include <igl/per_vertex_normals.h>
#include <igl/repmat.h>
#include <igl/upsample.h>
#include <igl/boundary_loop.h>
#include <igl/boundary_facets.h>

namespace utils {

void addElement2IntVector(Eigen::VectorXi &vec, int val) {
    vec.conservativeResize(vec.rows() + 1, Eigen::NoChange);
    vec(vec.rows()-1) = val;
}

/* function safeLog
 log which is linearly extended below a threshold epsi
 */
float safeLog(const float x) {
    float l;
    if (x > FLOAT_EPSI) {
        l = std::log(x);
    }
    else {
        l = (x - FLOAT_EPSI)/FLOAT_EPSI + std::log(FLOAT_EPSI);
    }
    return l;
}

Eigen::ArrayXf arraySafeLog(const Eigen::ArrayXf X) {
    Eigen::ArrayXf L = X;
    for (int i = 0; i < X.rows(); i++) {
        L(i) = safeLog(X(i));
    }
    return L;
}

float squaredNorm(const Eigen::Vector3f vec) {
    return vec(0)*vec(0) + vec(1)*vec(1) + vec(2)*vec(2);
}


/* function setLinspaced
    creates a increasing vector of fixed step of one
    e.g.
    mat = Eigen::MatrixXi(1, 5);
    setLinspaced(mat, 2);
    creates
    mat = [2 3 4 5 6]
 */
void setLinspaced(Eigen::MatrixXi& mat, int start) {
    assert(mat.rows() == 1 || mat.cols() == 1);
    int length;
    if (mat.rows() == 1) {
        length = mat.cols();
    }
    else {
        length = mat.rows();
    }
    for (int i = 0; i < length; i++) {
        mat(i) = i + start;
    }
}

Eigen::MatrixXi linspaced(int start, int end) {
    return linspaced(start, end, 1);
}

Eigen::MatrixXi linspaced(int start, int end, int step) {
    assert(step > 0);
    assert(end > start);
    int length = (end - start)/step;
    Eigen::MatrixXi A(length, 1);
    for (int i = 0; i < length; i++) {
        A(i, 0) = start + i * step;
    }
    return A;
}

Eigen::MatrixXd normals3d(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, F, N);
    return N;
}


Eigen::MatrixXd normals2d(Eigen::MatrixXd V, const Eigen::MatrixXi& E) {
    /*
     Setup
     */
    if (V.rows() != E.rows()) {
        std::cout << "error: V.rows() != E.rows()" << std::endl;
        return Eigen::MatrixXd(1, 1);
    }

    int zeroDimension = -1;
    for (int i = 0; i < 3; i++) {
        if (std::abs(V.col(i).sum()) < 1e-4) {
            zeroDimension = i;
            break;
        }
    }
    if (zeroDimension == -1) {
        std::cout << "error: points dont lie on xy, xz or yz plane" << std::endl;
        return Eigen::MatrixXd(1, 1);
    }
    if (zeroDimension != 2) {
        Eigen::Vector3<int> resorting;
        if (zeroDimension == 1)
            resorting << 0, 2, 1;
        else
            resorting << 1, 2, 0;

        V = V(Eigen::all, resorting);
    }
    Eigen::MatrixXd edgeVectors = V(E.col(0), Eigen::all) - V(E.col(1), Eigen::all);
    Eigen::MatrixXd edgeLenghts = rowNorm(edgeVectors);
    Eigen::MatrixXd rotationAroundZ(3, 3); rotationAroundZ.setZero();
    rotationAroundZ(0, 1) = 1;
    rotationAroundZ(1, 0) = -1;
    rotationAroundZ(2, 2) = 1;


    /*
     normal computation
     */

    Eigen::MatrixXd edgeN = edgeVectors * rotationAroundZ.transpose();

    edgeN = edgeN.rowwise().normalized();
    // check if normals point outward => bounding box must get bigger
    const float maxXVal = V.col(0).maxCoeff();
    const float minXVal = V.col(0).minCoeff();
    const float maxYVal = V.col(1).maxCoeff();
    const float minYVal = V.col(1).minCoeff();
    const float multiplier = 0.1 * std::min(std::abs(maxXVal - minXVal), std::abs(maxYVal - minYVal));
    const float maxXValmult = (V + multiplier * edgeN).col(0).maxCoeff();
    // flip normals if they dont point outwards
    if (maxXValmult < maxXVal)
        edgeN = -edgeN;

    // transform edge normals to vertex normals by a weighted sum of adjacent edge normals
    Eigen::MatrixXd vertexN(V.rows(), 3);
    for (int i = 1; i < V.rows()+1; i++) {
        const int nextIdx = i % V.rows();
        vertexN.row(nextIdx) = edgeLenghts(i-1) * edgeN.row(i-1) + edgeLenghts(nextIdx) * edgeN.row(nextIdx);
    }
    vertexN = vertexN.rowwise().normalized();

    return vertexN;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::VectorXi, Eigen::MatrixXi, Eigen::VectorX<bool>> getSTAR(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {

    const int originalNumVerts = (int) V.rows();
    Eigen::MatrixXd VV;
    Eigen::MatrixXi FF;
    igl::upsample(V, F, VV, FF, 1);
    const int stageTwoNumVerts = (int) VV.rows();


    Eigen::MatrixXd VVV;
    Eigen::MatrixXi FFF;
    igl::upsample(VV, FF, VVV, FFF, 1);

    Eigen::MatrixXi TTadjacencyFFF;
    igl::triangle_triangle_adjacency(FFF, TTadjacencyFFF);


    Eigen::VectorXi FFFids(FFF.rows());
    FFFids.setConstant(-1);
    // first loop => set triangles adjacent to original vertices
    for (int fff = 0; fff < FFF.rows(); fff++) {
        const int smallestVertexId = FFF.row(fff).minCoeff();
        if (smallestVertexId < originalNumVerts) {
            FFFids(fff) = smallestVertexId;
        }
    }

    // second and third loop => set triangles next to already set triangles
    for (int j = 0; j < 2; j++) {
        Eigen::MatrixXi FFFidsCopy = FFFids;
        for (int fff = 0; fff < FFF.rows(); fff++) {
            for (int i = 0; i < 3; i++) {
                const int neighbouringTriangleId = FFFids(TTadjacencyFFF(fff, i));
                if (neighbouringTriangleId != -1) {
                    FFFidsCopy(fff) = neighbouringTriangleId;
                }
            }
        }
        FFFids = FFFidsCopy;
    }

    // forth loop => set vetexids
    Eigen::MatrixXi VVVids(VVV.rows(), 1);
    VVVids.setConstant(-1);
    for (int fff = 0; fff < FFF.rows(); fff++) {
        for (int i = 0; i < 3; i++) {
            if (FFFids(fff) != -1)
                VVVids(FFF(fff, i)) = FFFids(fff);
        }
    }

    // fith loop => determine inner tri ids
    for (int fff = 0; fff < FFF.rows(); fff++) {
        const int triId0 = FFFids(TTadjacencyFFF(fff, 0));
        const int triId1 = FFFids(TTadjacencyFFF(fff, 1));
        const int triId2 = FFFids(TTadjacencyFFF(fff, 2));
        if (triId0 == -1 && triId1 == -1 && triId2 == -1) {
            const int vertexId0 = VVVids(FFF(fff, 0));
            const int vertexId1 = VVVids(FFF(fff, 1));
            const int vertexId2 = VVVids(FFF(fff, 2));
            Eigen::MatrixXi triIdx;
            igl::find((F.array() == vertexId0).rowwise().sum() > 0 &&
                      (F.array() == vertexId1).rowwise().sum() > 0 &&
                      (F.array() == vertexId2).rowwise().sum() > 0, triIdx);
            FFFids(fff) = originalNumVerts + triIdx(0, 0);
            //std::cout << F.row(triIdx(0, 0)) << ", " << vertexId0 << ", " << vertexId1 << ", " << vertexId2 << ", " << std::endl;
        }
    }


    // sixth loop => set final triangles
    for (int fff = 0; fff < FFF.rows(); fff++) {
        if (FFFids(fff) == -1) {
            int triId = FFFids(TTadjacencyFFF(fff, 0));
            for (int i = 1; i < 3; i++) {
                triId = std::max(FFFids(TTadjacencyFFF(fff, i)), triId);
            }
            FFFids(fff) = triId;
        }
    }

    const int numCycles = (int) (V.rows() + F.rows());


    Eigen::MatrixXi CycEdges(FFF.rows() * 3, 2);
    Eigen::VectorX<bool> Lable(FFF.rows() * 3);
    Lable.setConstant(false);
    std::vector<int> loop;
    int numCycEdges = 0;
    for (int c = 0; c < numCycles; c++) {
        igl::boundary_loop(igl::slice_mask(FFF, FFFids.array() == c, 1), loop);
        Eigen::MatrixXi boundaryEdges(loop.size(), 2);
        for (int i = 0; i < loop.size(); i++) {
            boundaryEdges.row(i) << loop[i], loop[(i+1) % loop.size()];

            if (c < V.rows()) {
                if (loop[i] < VV.rows()) {
                    Lable(numCycEdges+i) = true;
                }
            }
            else {
                if (loop[i] >= VV.rows()) {
                    Lable(numCycEdges+i) = true;
                }
            }

        }

        CycEdges.block(numCycEdges, 0, boundaryEdges.rows(), 2) = boundaryEdges;
        numCycEdges += boundaryEdges.rows();
        CycEdges.row(numCycEdges) << -1, -1;
        numCycEdges++;
    }
    numCycEdges -= 1; // we added one -1, -1 edge too much
    CycEdges.conservativeResize(numCycEdges, 2);
    Lable.conservativeResize(numCycEdges, 1);

    return std::make_tuple(VVV, FFF, FFFids, CycEdges, Lable);
}


} // namespace utils


int findEdge(const tsl::robin_set<EDG> &ELookup, const EDG &edg) {
    auto it = ELookup.find(edg);
    if(it != ELookup.end()) {
        EDG foundEdg = *it;
        return foundEdg.e;
    }
    return -1;
}

namespace std {
    std::size_t hash<EDG>::operator()(EDG const& edg) const noexcept {
        //size_t idx0hash = std::hash<int>()(edg.idx0);
        //size_t idx1hash = std::hash<int>()(edg.idx1) << 1;
        //return idx0hash ^ idx1hash;
        int k1 = edg.idx0;
        int k2 = edg.idx1;
        return (k1 + k2 ) * (k1 + k2 + 1) / 2 + k2;
    }
}
