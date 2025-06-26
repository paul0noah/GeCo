//
//  deformationEnergy.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#ifndef elasticEnergy_hpp
#define elasticEnergy_hpp
#include <Eigen/Dense>
#include "helper/shape.hpp"

#define DEBUG_DEFORMATION_ENERGY false


class ElasticEnergy {
private:
    const Eigen::MatrixXd& VX;
    const Eigen::MatrixXd& VY;
    const Eigen::MatrixXi& FX;
    const Eigen::MatrixXi& FY;
    const Eigen::MatrixXd& CurvX;
    const Eigen::MatrixXd& CurvY;

    const Eigen::MatrixXi& productspace;

    bool lineIntegral;
    
    bool computed;
    Eigen::MatrixXd defEnergy;
    void computeEnergy();
    std::string costName, timeName;
    std::string prefix;
    bool normalise;

public:
    ElasticEnergy(const Eigen::MatrixXd& VX, const Eigen::MatrixXd& VY, const Eigen::MatrixXd& CurvX, const Eigen::MatrixXd& CurvY, const Eigen::MatrixXi& FX, const Eigen::MatrixXi& FY, const Eigen::MatrixXi& productspace, const bool normalise);
    Eigen::MatrixXd getElasticEnergy();

    void setCostTimeRatioMode(const std::string icostName, const std::string itimeName);
};

#endif /* elasticEnergy_hpp */

