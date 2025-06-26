//
//  combinations.hpp
//  helper
//
//  Created by Paul Rötzer on 28.04.21.
//

#ifndef combinations_hpp
#define combinations_hpp

#include <Eigen/Dense>
#include "helper/shape.hpp"
#define DEBUG_COMBINATIONS false
#define NEW_PLUS_MINUS true

class ProductSpace {
private:
    bool combosComputed;
    Eigen::MatrixXi& EX;
    Eigen::MatrixXi& EY;
    Eigen::MatrixXi productspace;
    Eigen::MatrixXi SRC_IDs;
    Eigen::MatrixXi TRGT_IDs;
    Eigen::MatrixXi piEY;
    Eigen::MatrixXi plusMinusDir;
    int numContours;
    bool pruneIntralayerEdges;
    std::vector<tsl::robin_set<long>> branchGraph;
    int maxDepth;

public:
    void init();
    ProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY);
    ProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, const bool pruneIntralyer);
    
    void computeCombinations();
    void setMaxDepth(const int iMaxDepth);
    void setPlusMinusDir(const Eigen::MatrixXi& iPlusMinusDir);
    Eigen::MatrixXi getProductSpace();
    Eigen::MatrixXi getPiEy();
    Eigen::MatrixXi getSRCIds();
    Eigen::MatrixXi getTRGTIds();
    int getNumContours() const;
    std::vector<tsl::robin_set<long>> getBranchGraph();
};

#endif /* combinations_hpp */
