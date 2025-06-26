//
//  deformationEnergy.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#include "elastic_energy.hpp"
#include <iostream>
#include <tsl/robin_map.h>



/* function getVoronoiArea
         R
         -
        / \
       /   \
      /     \
     /       \
    /_________\
  P             Q
  P is the vertex i for which we want to compute the Voronoi area:
  According to the paper by Meyer:
    A_voronoi = 1/8 * ( norm(PQ)^2 * cot(angle(R)) +  norm(PR)^2 * cot(angle(Q)))
 */
float getVoronoiArea(int i, Shape &shape, int neighboor, Eigen::MatrixXf &cotTriangleAngles) {
    Eigen::Vector3i triangle = shape.getFi(neighboor).transpose();
    Eigen::Vector3f P, Q, R;
    float cotAngleR, cotAngleQ;

    if (i == triangle(0)) {
        P = shape.getVi(triangle(0)).transpose();
        Q = shape.getVi(triangle(1)).transpose();
        R = shape.getVi(triangle(2)).transpose();
        cotAngleQ = cotTriangleAngles(neighboor, 1);
        cotAngleR = cotTriangleAngles(neighboor, 2);
    }
    else if (i == triangle(1)) {
        P = shape.getVi(triangle(1)).transpose();
        Q = shape.getVi(triangle(2)).transpose();
        R = shape.getVi(triangle(0)).transpose();
        cotAngleQ = cotTriangleAngles(neighboor, 2);
        cotAngleR = cotTriangleAngles(neighboor, 0);
    }
    else if (i == triangle(2)) {
        P = shape.getVi(triangle(2)).transpose();
        Q = shape.getVi(triangle(0)).transpose();
        R = shape.getVi(triangle(1)).transpose();
        cotAngleQ = cotTriangleAngles(neighboor, 0);
        cotAngleR = cotTriangleAngles(neighboor, 1);
    }
    return 0.125 * (utils::squaredNorm(P - Q) * cotAngleR + utils::squaredNorm(P - R) * cotAngleQ);
}


/* function getMixedArea

    computes the mixed area as proposed in (2) Fig. 4 (figure shows
    a pseudo-code version of the code which is implemented here)
    => in case of obtuse triangles the Voroni area around vertex i is not
    within the one-ring-neighboorhood or truncated and thus not plausible
    for the curvature computation.
 */
float getMixedArea(int i, Shape &shape, Eigen::VectorXi &oneRingNeighboorhood, Eigen::MatrixXf &triangleAngles, Eigen::MatrixXf &cotTriangleAngles) {

    float Amixed = 0;
    const double piHalf = M_PI * 0.5;

    for (int k = 0; k < oneRingNeighboorhood.rows(); k++) {

        // check if any angle in triangle is obtuse
        // => one angle has to be greater than piHalf
        bool obtuse = false;
        for (int j = 0; j < 3; j++) {
            if (triangleAngles(oneRingNeighboorhood(k), j) > piHalf) {
                obtuse = true;
                break;
            }
        }
        if (obtuse) {
            int idxAngle;
            // find the vertex we are currently talking about
            Eigen::Vector3i currTriangle = shape.getFi(oneRingNeighboorhood(k)).transpose();
            for (int j = 0; j < 3; j++) {
                if (currTriangle(j) == i){
                    idxAngle = j;
                    break;
                }
            }
            float factor = triangleAngles(oneRingNeighboorhood(k), idxAngle) >= piHalf ? 0.5 : 0.25;

            Amixed = Amixed + factor * shape.getTriangleArea(oneRingNeighboorhood(k));
        }
        // non-obtuse triangle
        else {
            Amixed = Amixed + getVoronoiArea(i, shape, oneRingNeighboorhood(k), cotTriangleAngles);
        }
    }
    return Amixed;
}

/* function getAMixed
 */
void getAMixed(Shape& shape, Eigen::MatrixXd& A) {

    Eigen::MatrixXf triangleAngles(shape.getNumFaces(), 3);
    for (int i = 0; i < shape.getNumFaces(); i++) {
        triangleAngles.row(i) = shape.getTriangleAngles(i);
    }

    Eigen::MatrixXf cotTriangleAngles(shape.getNumFaces(), 3);
    // cotan(x) = 1 / tan(x)
    cotTriangleAngles = triangleAngles.array().tan().inverse().matrix();

    Eigen::VectorXi oneRingNeighboorhood;
    Eigen::MatrixXf sumOverOneRingNeighborhood;
    for (int i = 0; i < shape.getV().rows(); i++) {
        oneRingNeighboorhood = shape.getTrianglesAttachedToVertex(i);
        A(i) = getMixedArea(i, shape, oneRingNeighboorhood, triangleAngles, cotTriangleAngles);
    }
}

inline double robustLoss(const double diff, const double alpha, const double c, const double pow) {
    const double x = diff;

    const double xDivCSqr = std::pow((x / c), 2);
    const double absAlphaMinusTwo = std::abs(alpha - 2);

    return absAlphaMinusTwo / alpha * (std::pow(xDivCSqr / absAlphaMinusTwo + 1, 0.5 * alpha) -1);
}

/* function getg()
 According to (1):
 g is actually a matrix
 g = [g0 g1; g2 g3];
 but we make a vector out of it
 => g = [g0 g1 g2 g3];
 according to the paper
 g = [e1; -e2]' * [e1; -e2];
 => g0 = e1' * e1;
    g1 = e1' * e2
    g2 = g1
    g3 = e2' * e2

 Assumes edges12 in the following format:
 edges12 = [edge1_0, edge2_0]
           [edge1_1, edge2_1]
           ...
 where edge1_X and edge2_X in IR^{1x3}
 */
Eigen::ArrayXXf getg(Eigen::ArrayXXf &edges12) {
    Eigen::ArrayXXf g(edges12.rows(), 4);
    g(Eigen::all, 0) = edges12.block(0, 0, edges12.rows(), 3).square().rowwise().sum();
    g(Eigen::all, 3) = edges12.block(0, 3, edges12.rows(), 3).square().rowwise().sum();
    g(Eigen::all, 1) = (edges12.block(0, 0, edges12.rows(), 3).cwiseProduct(- edges12.block(0, 3, edges12.rows(), 3))).rowwise().sum();
    g(Eigen::all, 2) = g(Eigen::all, 1);
    return g;
}

/* function getEdges12()
 returns 2 edges of each triangle in the following form:
 edges12 = [[edge1_0 edge2_0]
            [edge1_1 edge2_1]
            ...]
 */
Eigen::ArrayXXf getEdges12(Shape &shape, Eigen::MatrixXi &FCombo) {
    Eigen::ArrayXXf edges12(FCombo.rows(), 6);
    for (int i = 0; i < FCombo.rows(); i++) {
        edges12.block(i, 0, 1, 3) = (shape.getVi(FCombo(i, 1)) - shape.getVi(FCombo(i, 0))).array();
        edges12.block(i, 3, 1, 3) = (shape.getVi(FCombo(i, 2)) - shape.getVi(FCombo(i, 1))).array();
    }
    return edges12;
}

/* function getG

  G = inv(gA) * gB = [G1 G2; G3 G4]
  for vectorized computation we output
  G = [G1 G2 G3 G4]
  where each GX is a colum vector

  G = inv(gA) * gB
  since we don't want to invert we use the following
  A = [a b; c d]
  inv(A) = 1/(ad - bc) * [d -b; -c a]
  consequently
  gA = [a1 a2; a3 a4];
  gB = [b1 b2; b3 b4];
  => inv(gA) * gB = 1 / (a1*a4 - a2*a3) * [a4 -a2; -a3 a1] * [b1 b2; b3 b4]
                  = 1 / (a1*a4 - a2*a3) * [a4*b1 + -a2*b3, a4*b2 + -a2*b4]
                                          [-a3*b1 + a1*b3, -a3*b2 + a1*b4]
                  = [G1 G2; G3 G4]
 */
Eigen::ArrayXXf getG(Shape &shapeA, Shape &shapeB, Eigen::MatrixXi &FaCombo, Eigen::MatrixXi &FbCombo) {

    Eigen::ArrayXXf edgesA12(FaCombo.rows(), 6); edgesA12 = getEdges12(shapeA, FaCombo);
    Eigen::ArrayXXf edgesB12(FbCombo.rows(), 6); edgesB12= getEdges12(shapeB, FbCombo);

    Eigen::ArrayXXf gA(edgesA12.rows(), 4); gA = getg(edgesA12);
    Eigen::ArrayXXf gB(edgesB12.rows(), 4); gB = getg(edgesB12);

    Eigen::ArrayXf detga =  gA(Eigen::all, 0).cwiseProduct( gA(Eigen::all, 3) ) - gA(Eigen::all, 1).cwiseProduct( gA(Eigen::all, 2) );

    detga = detga.inverse();

    Eigen:: ArrayXXf G(FaCombo.rows(), 4);
    G(Eigen::all, 0) =  ( gA(Eigen::all, 3).cwiseProduct(gB(Eigen::all, 0))
                         - gA(Eigen::all, 1).cwiseProduct(gB(Eigen::all, 2))
                        ).cwiseProduct(detga);
    G(Eigen::all, 1) =  ( gA(Eigen::all, 3).cwiseProduct(gB(Eigen::all, 1))
                         - gA(Eigen::all, 1).cwiseProduct(gB(Eigen::all, 3))
                        ).cwiseProduct(detga);
    G(Eigen::all, 2) = ( -gA(Eigen::all, 2).cwiseProduct(gB(Eigen::all, 0))
                        + gA(Eigen::all, 0).cwiseProduct(gB(Eigen::all, 2))
                       ).cwiseProduct(detga);
    G(Eigen::all, 3) = ( -gA(Eigen::all, 2).cwiseProduct(gB(Eigen::all, 1))
                        + gA(Eigen::all, 0).cwiseProduct(gB(Eigen::all, 3))
                       ).cwiseProduct(detga);
    return G;
}

/* function w = getW(G, mu, lam)
 According to (1)
  W(A) = mu / 2 * tr(A)+ lambda / 4 * det(A) - (mu / 2 + lambda / 4) *
            * log(det(A)) - (mu + lambda / 4)
  A actually is
  A = [A0 A1; A2 A3]
  but due to vectorization we input
  A = [A0 A1 A2 A3]
 */
Eigen::ArrayXf getW(Eigen::ArrayXXf &A, float mu, float lambda) {
    // tr(A) = A0 + A3
    Eigen::ArrayXf trA(A.rows(), 1);
    trA = A(Eigen::all, 0) + A(Eigen::all, 3);

    // det(A) = A0 * A3 - A2 * A1
    Eigen::ArrayXf detA = A(Eigen::all, 0).cwiseProduct(A(Eigen::all, 3)) - A(Eigen::all, 2).cwiseProduct(A(Eigen::all, 1));

    return mu/2 * trA + lambda/4 * detA - (mu/2 + lambda/4) * utils::arraySafeLog(detA) - (mu + lambda/4);
}




/* function get
    computes membrane energy according to (1):
        Membrane Energy between Triangle A and B
        W_mem(A, B) = A_a * W(G_t)
        where
        G_t = g_at^-1 * g_bt;

               -
              / \
         eX1 /   \ eX1
            /  X  \
           /       \
          /_________\
             eX0
        g_t = [eX1; -eX2]' * [eX1; -eX2];
        => no parallel edges in triangle A!!!! (no degenerate cases in A)
        (so g_at is invertible)

        W(A) = mu / 2 * tr(A)+ lambda / 4 * det(A) - (mu / 2 + lambda / 4) *
            * log(det(A)) - (mu + lambda / 4)

 */
float getMembraneEnergy(Shape &shapeA, Eigen::ArrayXf areasA, Shape &shapeB, Eigen::MatrixXi FaCombo, Eigen::MatrixXi FbCombo) {

    // these are some material properties, maybe something we could tune
    float mu = 1;
    float lambda = 1;

    Eigen::ArrayXXf G(FaCombo.rows(), 4);
    G = getG(shapeA, shapeB, FaCombo, FbCombo);

    Eigen::MatrixXf membraneEnergy(FaCombo.rows(), 3);
    membraneEnergy = areasA.cwiseProduct(getW(G, mu, lambda)).matrix();

    return membraneEnergy.sum();
}

/* REFERENCES
  (1) EZUZ, Danielle, et al. Elastic correspondence between triangle meshes.
      In: Computer Graphics Forum. 2019. S. 121-134.
 */


tsl::robin_map<EDG, Eigen::Matrix<int, 1, 2>> getEdgeTriangleAdjacency(const Eigen::MatrixXi& F) {
    Eigen::MatrixXi edgeIndices(3, 2); edgeIndices << 0, 1, 1, 2, 2, 0;

    tsl::robin_map<EDG, Eigen::Matrix<int, 1, 2>> edgeTriangleAdjacency;
    edgeTriangleAdjacency.reserve(3 * F.rows());

    for (int f = 0; f < F.rows(); f++) {
        for (int i = 0; i < 3; i++) {
            EDG edge0(F(f, edgeIndices.row(i)));

            if (edgeTriangleAdjacency.find(-edge0) != edgeTriangleAdjacency.end()) {
                Eigen::Matrix<int, 1, 2> vals = edgeTriangleAdjacency[-edge0];
                vals(1) = f;
                edgeTriangleAdjacency.insert({edge0, vals});
            }
            else {
                Eigen::Matrix<int, 1, 2> vals; vals(0) = f; vals(1) = -2;
                edgeTriangleAdjacency.insert({edge0, vals});
            }
        }
    }
    return edgeTriangleAdjacency;
}

/*



































 */

void ElasticEnergy::computeEnergy() {
    const int numColumnsEnergy = 1;

    Shape shapeX(VX.cast<float>(), FX);
    Shape shapeY(VY.cast<float>(), FY);
    Eigen::ArrayXf AX = shapeX.getTriangleAreas();
    Eigen::ArrayXf AY = shapeY.getTriangleAreas();
    tsl::robin_map<EDG, Eigen::Matrix<int, 1, 2>> edgeTriAdjacencyX = getEdgeTriangleAdjacency(FX);
    tsl::robin_map<EDG, Eigen::Matrix<int, 1, 2>> edgeTriAdjacencyY = getEdgeTriangleAdjacency(FY);
    Eigen::MatrixXd AXmixed(VX.rows(), 1);
    Eigen::MatrixXd AYmixed(VY.rows(), 1);
    getAMixed(shapeX, AXmixed);
    getAMixed(shapeY, AYmixed);
    const double alpha = -0.6;
    const double c = 0.7;
    const double powwer = 2;

    const float mu = 1.0;
    const float lambda = 1.0;
    const Eigen::Matrix<int, 1, 2> defaultVals((Eigen::Matrix<int, 1, 2>() << -1, -1).finished());


    defEnergy = Eigen::MatrixXd(productspace.rows(), numColumnsEnergy);
    defEnergy.setZero();

    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (long i = 0; i < productspace.rows(); i++) {
        double lineIntegralVal = 1.0;
        const long idx2d0 = productspace(i, 0); // y
        const long idx2d1 = productspace(i, 1); // y
        const long idx3d0 = productspace(i, 2); // x
        const long idx3d1 = productspace(i, 3); // x

        const EDG ex = EDG(idx3d0, idx3d1);
        const EDG ey = EDG(idx2d0, idx2d1);

        if (idx2d0 == -1) {
            // handle weird inbetween elements
            defEnergy(i, 0) = 999999999.9;
            continue;
        }



        /*
         beding energy:
            for i-th product edge e_i the bending energy is the following
            c_bend_i = sum(i = 0, 1) (AXmixed(e(i, x)) + AYmixed(e(i, y))) * ( (CX(e(i, x) - CY(e(i, y)) ** 2 )
         */

        defEnergy(i)  = (AXmixed(idx3d0) + AYmixed(idx2d0)) * robustLoss(CurvX(idx3d0) - CurvX(idx2d0), alpha, c, powwer);
        defEnergy(i) += (AXmixed(idx3d1) + AYmixed(idx2d1)) * robustLoss(CurvX(idx3d1) - CurvX(idx2d1), alpha, c, powwer);

        /*
         membrane energy
         case 1: ex and ey non-degenerate
            take neighbouring left and right triangles of edges on x and y and compute mebrane energy left to left + mebrane energy right to right (see roetzer et al. 2022 for that)
            (use symmetric formulation)
         case 2: ex degenerate
            take neighbouring left and right triangles of edge on y and compute membrane energy to ex vertex
            (unsymmetric => multiply by two, see also windheuser et al. 2011)
         case 3: ey degenerate
            take neighbouring left and right triangles of edge on x and compute membrane energy to ex vertex
            (unsymmetric => multiply by two, see also windheuser et al. 2011)
         */
        // case 1
        // TODO: find triangles
        const Eigen::Matrix<int, 1, 2> valsX = edgeTriAdjacencyX.find(ex) != edgeTriAdjacencyX.end() ? edgeTriAdjacencyX[ex] : defaultVals;
        const Eigen::Matrix<int, 1, 2> valsY = edgeTriAdjacencyY.find(ey) != edgeTriAdjacencyY.end() ? edgeTriAdjacencyY[ey] : defaultVals;
        int xleft  = valsX(0);
        int xright = valsX(1);
        int yleft  = valsY(0);
        int yright = valsY(1);

        // 1: non-boundary auf non-boundaray (good, no work to do)
        // 2: x boundary, y non boundary
        if (xleft == -2 || xright == -2) {
            if (xleft == xright)
                std::cout << prefix << "floating edge x in nowhere (no adjacent triangles found), this should not happen" << std::endl;
            xleft = xleft == -2 ? xright : xleft;
            xright = xright == -2 ? xleft : xright;
        }
        // 3: y boundary, x non boundary
        if (yleft == -2 || yright == -2) {
            if (yleft == yright)
                std::cout << prefix << "floating edge y in nowhere (no adjacent triangles found), this should not happen" << std::endl;
            yleft = yleft == -2 ? xright : yleft;
            yright = yright == -2 ? yleft : yright;
        }

        if (idx2d0 != idx2d1 && idx3d0 != idx3d1) {
            const Eigen::MatrixXi triXLeft  = FX.row(xleft);
            const Eigen::MatrixXi triXRight = FX.row(xright);
            const Eigen::MatrixXi triYLeft  = FY.row(yleft);
            const Eigen::MatrixXi triYRight = FY.row(yright);
            Eigen::ArrayXf AXLEFT(1, 1), AXRIGHT(1, 1), AYLEFT(1, 1), AYRIGHT(1, 1);
            AXLEFT << AX(xleft); AXRIGHT << AX(xright); AYLEFT << AY(yleft); AYRIGHT << AY(yright);
            defEnergy(i) += getMembraneEnergy(shapeX, AXLEFT, shapeY, triXLeft, triYLeft);
            defEnergy(i) += getMembraneEnergy(shapeY, AYLEFT, shapeX, triYLeft, triXLeft);
            defEnergy(i) += getMembraneEnergy(shapeX, AXRIGHT, shapeY, triXRight, triYRight);
            defEnergy(i) += getMembraneEnergy(shapeY, AYRIGHT, shapeX, triYRight, triXRight);
        }
        // case 2
        else if (idx3d0 == idx3d1) {
            Eigen::MatrixXi triXDegenerate(1, 3); triXDegenerate << idx3d0, idx3d0, idx3d0;
            const Eigen::MatrixXi triYLeft  = FY.row(yleft);
            const Eigen::MatrixXi triYRight = FY.row(yright);
            Eigen::ArrayXf AYLEFT(1, 1), AYRIGHT(1, 1);
            AYLEFT << AY(yleft); AYRIGHT << AY(yright);
            defEnergy(i) += 2 * getMembraneEnergy(shapeY, AYLEFT, shapeX, triYLeft, triXDegenerate);
            defEnergy(i) += 2 * getMembraneEnergy(shapeY, AYRIGHT, shapeX, triYRight, triXDegenerate);
        }
        // case 3
        else if (idx2d0 == idx2d1) {
            const Eigen::MatrixXi triXLeft  = FX.row(xleft);
            const Eigen::MatrixXi triXRight = FX.row(xright);
            Eigen::ArrayXf AXLEFT(1, 1), AXRIGHT(1, 1);
            AXLEFT << AX(xleft); AXRIGHT << AX(xright);
            Eigen::MatrixXi triYDegenerate(1, 3); triYDegenerate << idx2d0, idx2d0, idx2d0;
            defEnergy(i) += getMembraneEnergy(shapeX, AXLEFT, shapeY, triXLeft, triYDegenerate);
            defEnergy(i) += getMembraneEnergy(shapeX, AXRIGHT, shapeY, triXRight, triYDegenerate);
        }
        else {
            std::cout << prefix << "not implemented error!!!" << std::endl;
        }

        if (std::isnan(defEnergy(i))) {
            std::cout << prefix << "deformation energy is nan. probably your input meshes are not clean (e.g. vertices in same position)" << std::endl;
        }





    }

    if (normalise) {
        const float medianEnergy = utils::median(defEnergy);
        defEnergy  = (defEnergy.array() / std::max(0.001f, medianEnergy)).array();
    }

    computed = true;
}

ElasticEnergy::ElasticEnergy(const Eigen::MatrixXd& VX, const Eigen::MatrixXd& VY, const Eigen::MatrixXd& CurvX, const Eigen::MatrixXd& CurvY, const Eigen::MatrixXi& FX, const Eigen::MatrixXi& FY, const Eigen::MatrixXi& productspace, const bool normalise) :
VX(VX), VY(VY), CurvX(CurvX), CurvY(CurvY), FX(FX), FY(FY), productspace(productspace), normalise(normalise) {
    computed = false;
    prefix = "[ElasticEnergy] ";
}

Eigen::MatrixXd ElasticEnergy::getElasticEnergy() {
    if (!computed) {
        computeEnergy();
    }
    return defEnergy;
}
