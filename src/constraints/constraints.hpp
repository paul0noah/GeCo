//
//  constraints.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 17.04.21.
//

#ifndef constraints_hpp
#define constraints_hpp

#include <Eigen/Sparse>
#include "helper/shape.hpp"
#include "src/product_spaces/product_space.hpp"
#include "helper/utils.hpp"

#define DEBUG_CONSTRAINTS true


class Constraints {
private:
    Eigen::MatrixXi& EX;
    Eigen::MatrixXi& EY;
    Eigen::MatrixXi& productspace;
    Eigen::MatrixXi& SRCIds;
    Eigen::MatrixXi& TRGTIds;
    Eigen::MatrixXi& PLUSMINUSDIR;
    Eigen::MatrixXi RHS;
    Eigen::MatrixXi couplingDecoder;
    Eigen::MatrixXi couplingEncoder;
    bool coupling;
    bool allowOtherSelfIntersections;
    long nVX;
    int numCouplingConstraints;
    int numContours;
    bool resolveCoupling;
    bool meanProblem;

public:
    //Constraints constr(EX, EY, productspace, numContours, SRCIds, TRGTIds, PLUSMINUSDIR, couplingConstraints);
    Constraints(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, Eigen::MatrixXi& productspace, const int numContours, Eigen::MatrixXi& SRCIds, Eigen::MatrixXi& TRGTIds, Eigen::MatrixXi& PLUSMINUSDIR, bool coupling, bool resolveCoupling, bool meanProblem);
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> getConstraints();
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> getLeqConstraints(const int maxDepth);
    Eigen::MatrixXi getRHS();
    int getNumCouplingConstr();

    Eigen::MatrixXi getCouplingDecoder();
    Eigen::MatrixXi getCouplingEncoder();
};
#endif /* constraints_hpp */
