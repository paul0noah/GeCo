//
//  product_graph_generators.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 23.05.21.
//

#ifndef ShapeMatchModel_hpp
#define ShapeMatchModel_hpp

#include "helper/shape.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ctime>
#include "energy/elastic_energy/elastic_energy.hpp"

#define DEBUG_SHAPE_MATCH_MODEL true


class ProductGraphGenerators {
private:
    std::string prefix;
    bool verbose;
    Eigen::MatrixXd VX;
    Eigen::MatrixXi EX;
    Eigen::MatrixXd NormalsX;
    Eigen::MatrixXd VY;
    Eigen::MatrixXi EY;
    Eigen::MatrixXd NormalsY;
    Eigen::MatrixXd FeatDiffMatrix;

    Eigen::MatrixXi AI;
    Eigen::MatrixXi AJ;
    Eigen::MatrixXi AV;
    Eigen::MatrixXi RHS;

    Eigen::MatrixXi AIleq;
    Eigen::MatrixXi AJleq;
    Eigen::MatrixXi AVleq;
    Eigen::MatrixXi RHSleq;

    Eigen::MatrixXi productspace;
    Eigen::MatrixXi piCycle;
    Eigen::MatrixXi SRCIds;
    Eigen::MatrixXi TRGTIds;
    Eigen::MatrixXi PLUSMINUSDIR;
    Eigen::MatrixXi piEy;
    Eigen::MatrixXd energy;
    Eigen::MatrixXi couplingDecoder;
    Eigen::MatrixXi couplingEncoder;
    int nnzColsConstraints;
    bool modelGenerated;
    int numCouplingConstraints;
    bool regularisingCostTerm;
    int numContours;
    bool conjugateGraph;
    std::vector<tsl::robin_set<long>> branchGraph;
    bool normalsGiven;
    double rlAlpha; 
    double rlC;
    double rlPwr;
    int maxDepth;
    int maxCycleLength;
    bool pruneIntralayerEdges;
    bool resolveCouple;
    bool meanProblem;
    std::string costName, timeName;

    void writeToFile();
    
public:
    ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix);
    ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix, bool iConjugateGraph, bool iRegularisingCostTerm);
    ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix, bool iConjugateGraph, bool iRegularisingCostTerm, bool pruneIntralayer);
    ~ProductGraphGenerators();
    void generate();

    Eigen::MatrixXi convertEdgeMatching2CycleMatching(Eigen::MatrixX<bool> resultsVec);

    Eigen::MatrixXd getCostVector();
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> getAVectors();
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> getAleqVectors();
    void setNormals(Eigen::MatrixXd& inormalsX, Eigen::MatrixXd& inormalsY);
    void exportInputs();
    void exportInputs(Eigen::MatrixX<bool> resultsVec);
    void setMaxDepth(const int maxDepth);
    void setResolveCoupling(const bool iResolveCouple);
    void setMeanProblem(const bool iMeanProblem);
    void setCostTimeRatioMode(const std::string costName, const std::string timeName);
    Eigen::MatrixXi getRHS();
    Eigen::MatrixXi getRHSleq();
    Eigen::MatrixXi getProductSpace();
    int getNumCouplingConstraints();
    Eigen::MatrixXi getSortedMatching(const Eigen::MatrixXi& indicatorVector);
    Eigen::MatrixX<bool> decodeResultVector(const Eigen::MatrixX<bool>& iresultsVec);

    
    void updateRobustLossParams(const double alpha, const double c, const double pwr);
    Eigen::MatrixXd computeElasticEnergy(const Eigen::MatrixXi& FX, const Eigen::MatrixXi& FY, const Eigen::MatrixXd& CX, const Eigen::MatrixXd& CY,  const bool normalise);
#if WITH_OR_TOOLS
    Eigen::MatrixXf solveWithORTools(std::string solvername, int timelimit, Eigen::MatrixXd localenergy, float precision);
#endif
};

#endif /* ShapeMatchModel_hpp */
