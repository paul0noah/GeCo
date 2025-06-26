//
//  MultiCurveHandler.hpp
//  helper
//
//  Created by Paul Rötzer on 28.04.24.
//

#ifndef MultiCurveHandler_hpp
#define MultiCurveHandler_hpp

#include <Eigen/Dense>
#include "helper/shape.hpp"
#define DEBUG_MCH false

class MultiCurveHandler {
private:
    bool combosComputed;
    Eigen::MatrixXi& EX;
    Eigen::MatrixXi& EY;
    Eigen::MatrixXi productspace;
    Eigen::MatrixXi SRC_IDs;
    Eigen::MatrixXi TRGT_IDs;
    Eigen::MatrixXi PLUSMINUSDIR;
    Eigen::MatrixXi piCycle;
    Eigen::MatrixXi piEY;
    int numContours;
    bool pruneIntralayerEdges;
    std::vector<tsl::robin_set<long>> branchGraph;
    bool verbose;
    std::string prefix;
    int maxDepth;
    int maxCycleLength;

public:
    void init();
    MultiCurveHandler(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY);
    MultiCurveHandler(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, const bool pruneIntralyer, const int maxDepth);
    
    void computeCombinations();
    int getMaxCycleLength();
    Eigen::MatrixXi getProductSpace();
    Eigen::MatrixXi getPiEy();
    Eigen::MatrixXi getPiCycle();
    Eigen::MatrixXi getSRCIds();
    Eigen::MatrixXi getTRGTIds();
    Eigen::MatrixXi getPlusMinusDir();
    int getNumContours() const;
    std::vector<tsl::robin_set<long>> getBranchGraph();
};

#endif /* MultiCurveHandler_hpp */
