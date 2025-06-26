//
//  product_graph_generators.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 23.05.21.
//

#include "product_graph_generators.hpp"
#include <fstream>
#include <iostream>
#include <cstdio>
#include "helper/utils.hpp"
#include <chrono>
#include <filesystem>
#include <algorithm>
#include "src/product_spaces/multi_curve_handler.hpp"
#include "src/energy/deformationEnergy.hpp"
#include "src/constraints/constraints.hpp"


void ProductGraphGenerators::generate() {


    if (verbose) std::cout << prefix << "Generating Surface Cycles for normal product space..." << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    pruneIntralayerEdges = true;
    const std::string pruneOutput = pruneIntralayerEdges ? " (pruning intralayer edges)" : "";
    if (verbose) std::cout << prefix << "  > Product Space" << pruneOutput << pruneOutput << std::endl;
    MultiCurveHandler mcccombos(EX, EY, pruneIntralayerEdges, maxDepth);
    productspace = mcccombos.getProductSpace();
    numContours = mcccombos.getNumContours();
    SRCIds = mcccombos.getSRCIds();
    TRGTIds = mcccombos.getTRGTIds();
    PLUSMINUSDIR = mcccombos.getPlusMinusDir();
    piCycle = mcccombos.getPiCycle();
    maxCycleLength = mcccombos.getMaxCycleLength();

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << "  Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "  [ms])" << std::endl;

    if (verbose) std::cout << "[ShapeMM]   > Constraints" << std::endl;
    const bool couplingConstraints = true;
    Constraints constr(EX, EY, productspace, numContours, SRCIds, TRGTIds, PLUSMINUSDIR, couplingConstraints, resolveCouple, meanProblem);
    const auto constrVectors = constr.getConstraints();
    AI  = std::get<0>(constrVectors);
    AJ  = std::get<1>(constrVectors);
    AV  = std::get<2>(constrVectors);
    if (resolveCouple) {
        couplingDecoder = constr.getCouplingDecoder();
        couplingEncoder = constr.getCouplingEncoder();
        nnzColsConstraints = couplingEncoder.rows();
    }
    RHS = constr.getRHS();
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << "  Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "  [ms])" << std::endl;
    if (verbose) std::cout << prefix << "  > Energies" << std::endl;
    DeformationEnergy defEnergy(VX, VY, productspace, FeatDiffMatrix, regularisingCostTerm);
    defEnergy.setCostTimeRatioMode(costName, timeName);

    energy = defEnergy.getDeformationEnergy();
    
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << "  Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "  [ms])" << std::endl;
    if (verbose) std::cout << prefix << "Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count() << "  [ms])" << std::endl;

    modelGenerated = true;
}



ProductGraphGenerators::ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix, bool iConjugateGraph, bool iRegularisingCostTerm, bool pruneIntralayer) {
    prefix = "[ShapeMM] ";
    VX = iVX;
    EX = iEX;
    VY = iVY;
    EY = iEY;
    AIleq = Eigen::MatrixXi(0, 0);
    FeatDiffMatrix = iFeatDiffMatrix;
    modelGenerated = false;
    verbose = true;
    numCouplingConstraints = 0;
    conjugateGraph = iConjugateGraph;
    regularisingCostTerm = iRegularisingCostTerm;
    numContours = 0;
    rlAlpha = 0.7; 
    rlC = 0.6;
    rlPwr = 4;
    maxDepth = 4;
    pruneIntralayerEdges = pruneIntralayer;
    costName = "vanilla";
    timeName = "";
    resolveCouple = false;
    meanProblem = false;
}

ProductGraphGenerators::ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix, bool iConjugateGraph, bool iRegularisingCostTerm) : ProductGraphGenerators(iVX, iEX, iVY, iEY, iFeatDiffMatrix, iConjugateGraph, iRegularisingCostTerm, true) {
}

ProductGraphGenerators::ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix) :
ProductGraphGenerators(iVX, iEX, iVY, iEY, iFeatDiffMatrix, false, false, true){

}

void ProductGraphGenerators::setNormals(Eigen::MatrixXd& inormalsX, Eigen::MatrixXd& inormalsY){
    if (inormalsX.rows() != VX.rows() || inormalsX.cols() != 3) {
        std::cout << prefix << "error: normals for shape X do not contain as many entries as expected. Assumed shape = |VX| x 3" << std::endl;
        return;
    }
    if (inormalsY.rows() != VY.rows() || inormalsY.cols() != 3) {
        std::cout << prefix << "error: normals for shape Y do not contain as many entries as expected. Assumed shape = |VY| x 3" << std::endl;
        return;
    }
    NormalsX = inormalsX;
    NormalsY = inormalsY;
    normalsGiven = true;
}


ProductGraphGenerators::~ProductGraphGenerators() {
    
}


void ProductGraphGenerators::setMaxDepth(const int imaxDepth) {
    maxDepth = imaxDepth;
}

Eigen::MatrixXd ProductGraphGenerators::getCostVector() {
    if (!modelGenerated) {
        generate();
    }
    if (resolveCouple) {
        Eigen::MatrixXd e(nnzColsConstraints, 1);
        e.setZero();
        for (int i = 0; i < energy.rows(); i++) {
            e(couplingDecoder(i)) += energy(i);
        }
        return e;
    }
    return energy;
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> ProductGraphGenerators::getAVectors() {
    if (!modelGenerated) {
        generate();
    }
    return std::make_tuple(AI, AJ, AV);
}

Eigen::MatrixXi ProductGraphGenerators::getRHS() {
    if (!modelGenerated) {
        generate();
    }
    return RHS;
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> ProductGraphGenerators::getAleqVectors() {
    if (AIleq.rows() == 0) {
        if (verbose) std::cout << prefix << "Generating non-intersection constraints..." << std::endl;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        Constraints constr(EX, EY, productspace, numContours, SRCIds, TRGTIds, PLUSMINUSDIR, true, resolveCouple, meanProblem);
        const auto constrVectors = constr.getLeqConstraints(maxDepth);
        AIleq = std::get<0>(constrVectors);
        AJleq = std::get<1>(constrVectors);
        AVleq = std::get<2>(constrVectors);
        if (resolveCouple) {
            for (int i = 0; i < AJleq.rows(); i++) {
                AJleq(i, 0) = couplingDecoder(AJleq(i, 0));
            }
        }
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        if (verbose) std::cout << prefix << "Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "  [ms])" << std::endl;
    }
    return std::make_tuple(AIleq, AJleq, AVleq);
}

Eigen::MatrixXi ProductGraphGenerators::getRHSleq() {
    if (!modelGenerated) {
        generate();
    }
    return RHSleq;
}

Eigen::MatrixXi ProductGraphGenerators::getProductSpace() {
    if (!modelGenerated) {
        generate();
    }
    if (resolveCouple) {
        return productspace(couplingEncoder.col(0), Eigen::all);
    }
    return productspace;
}

int ProductGraphGenerators::getNumCouplingConstraints() {
    return numCouplingConstraints;
}

void ProductGraphGenerators::setResolveCoupling(const bool iResolveCouple) {
    resolveCouple = iResolveCouple;
}

void ProductGraphGenerators::setMeanProblem(const bool iMeanProblem) {
    meanProblem = iMeanProblem;
}

Eigen::MatrixXi ProductGraphGenerators::getSortedMatching(const Eigen::MatrixXi& indicatorVector) {
    const long maxNumEdgesOnLevel = productspace.rows() / EY.rows();


    const long numCycleConstr = EY.rows();
    const long numInOutConstr = EY.rows() * VX.rows();
    std::vector<Eigen::Triplet<int>> in_out_entries;
    in_out_entries.reserve(AI.rows());
    for (long i = 0; i < AI.rows(); i++) {
        if (AI(i) >= numCycleConstr && AI(i) < numCycleConstr + numInOutConstr) {
            in_out_entries.push_back(Eigen::Triplet<int>(AI(i) - (int) numCycleConstr, AJ(i), AV(i)));
        }
    }

    Eigen::SparseMatrix<int, Eigen::RowMajor> in_out_rm(numInOutConstr, productspace.rows());
    in_out_rm.setFromTriplets(in_out_entries.begin(), in_out_entries.end());
    Eigen::SparseMatrix<int, Eigen::ColMajor> in_out_cm(numInOutConstr, productspace.rows());
    in_out_cm.setFromTriplets(in_out_entries.begin(), in_out_entries.end());


    long firstNonZeroIdx = -1;
    long numMatches = 0;
    for (long i = 0; i < indicatorVector.rows(); i++) {
        if (indicatorVector(i) == 1) {
            numMatches++;
            if (firstNonZeroIdx == -1)
                firstNonZeroIdx = i;
        }
    }
    Eigen::MatrixXi matchingSorted(numMatches, 4);
    matchingSorted = -matchingSorted.setOnes();
    Eigen::MatrixXi nodeUsed(numInOutConstr, 1); nodeUsed.setZero();
    //Eigen::MatrixXi sortedIndices(numMatches, 1); sortedIndices = -sortedIndices.setOnes();

    long idx = firstNonZeroIdx;
    for (long i = 0; i < numMatches; i++) {
        //sortedIndices(i, 1) = (int) idx;
        //std::cout << idx << ": " <<productspace.row(idx) << std::endl;
        matchingSorted.row(i) = productspace.row(idx);
        long row = -1;
        int currentVal = 0;
        bool newNodeFound = false;
        for (typename Eigen::SparseMatrix<int, Eigen::ColMajor>::InnerIterator it(in_out_cm, idx); it; ++it) {
            if (nodeUsed(it.row(), 0) == 0 && it.value() == -1) {
                //std::cout << "  " << it.value() << std::endl;
                row = it.row();
                nodeUsed(row, 0) = 1;
                currentVal = it.value();
                newNodeFound = true;
                break;
            }
        }
        if (!newNodeFound) {
            if (DEBUG_SHAPE_MATCH_MODEL) std::cout << prefix << "Did not find new node, aborting" << std::endl;
            long numadded = 0;
            for (long ii = 0; ii < indicatorVector.rows(); ii++) {
                if (indicatorVector(ii) == 1) {
                    matchingSorted.row(numadded) = productspace.row(ii);
                    numadded++;
                }
            }
            break;
        }

        for (typename Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(in_out_rm, row); it; ++it) {
            if (it.value() == -currentVal && indicatorVector(it.col(), 0) > 0) {
                idx = it.col();
                break;
            }
        }
    }

    return matchingSorted;
}

void ProductGraphGenerators::updateRobustLossParams(const double alpha, const double c, const double pwr) {
    if (!conjugateGraph) {
        return;
    }
    if (modelGenerated) {
        std::cout << prefix << "error: cannot update robust loss params after model has been generated" << std::endl;
        return;
    }
    rlAlpha = alpha; 
    rlC = c;
    rlPwr = pwr;
}

void ProductGraphGenerators::writeToFile() {
    utils::writeMatrixToFile(VX, "VX");
    utils::writeMatrixToFile(VY, "VY");
    utils::writeMatrixToFile(EX, "EX");
    utils::writeMatrixToFile(EY, "EY");
    utils::writeMatrixToFile(FeatDiffMatrix, "FeatDiffMatrix");
    if (conjugateGraph) {
        utils::writeMatrixToFile(NormalsX, "NX");
        utils::writeMatrixToFile(NormalsY, "NY");
    }
}


Eigen::MatrixX<bool> ProductGraphGenerators::decodeResultVector(const Eigen::MatrixX<bool>& iresultsVec) {
    Eigen::MatrixX<bool> resultsVec;
    if (productspace.rows() != iresultsVec.rows()) {
        resultsVec = Eigen::MatrixX<bool>(productspace.rows(), 1);
        for (int i = 0; i < resultsVec.rows(); i++) {
            resultsVec(i) = iresultsVec(couplingDecoder(i));
        }
    }
    else {
        resultsVec = iresultsVec;
    }
    return resultsVec;
}

Eigen::MatrixXi ProductGraphGenerators::convertEdgeMatching2CycleMatching(Eigen::MatrixX<bool> iresultsVec) {
    const Eigen::MatrixX<bool> resultsVec = decodeResultVector(iresultsVec);
    const int numMatches = resultsVec.nonZeros();
    Eigen::MatrixXi surfaceMatching(numMatches, 2 * maxCycleLength);
    surfaceMatching.setConstant(-1);

    const bool removeDegenerateY = false;
    if (removeDegenerateY) {
        std::cout << prefix << "WARNING: removing all degenerate Y matches" << std::endl;
    }

    Eigen::MatrixXi matches(numMatches+1, 5);
    matches.setConstant(-1);
    int nthmatch = 0;
    for (int i = 0; i < productspace.rows(); i++) {
        if (resultsVec(i)) {
            if (removeDegenerateY && productspace(i, 0) == productspace(i, 1))
                continue;
            matches.row(nthmatch) << productspace(i, 0), productspace(i, 1), productspace(i, 2), productspace(i, 3), piCycle(i);
            nthmatch++;
        }
    }

    int curveIdx = 0;
    int lastCurveIdx = 0;
    int numCurveElements = 0;
    std::vector<Eigen::MatrixXi> sortedCycleMatchings; sortedCycleMatchings.reserve(numContours);

    for (int i = 0; i < matches.rows(); i++) {
        if (matches(i, 4) == curveIdx) {
            numCurveElements++;
        }
        else {
            Eigen::MatrixXi curveMatching = matches.block(lastCurveIdx, 0, numCurveElements, 5);
            //std::cout << "-+-+-+-+-+-+" << std::endl << curveMatching << std::endl << "-+-+-+-+-+-+" << std::endl;
            // removing back forth edges
            Eigen::MatrixX<bool> keepEdge(numCurveElements, 1); keepEdge.setConstant(true);
            for (int i = 0; i < curveMatching.rows(); i++) {
                if (!keepEdge(i)) continue;
                const bool isDeg2d = curveMatching(i, 0) == curveMatching(i, 1);
                if (isDeg2d) {
                    bool anyBackForth = false;
                    for (int j = i+1; j < curveMatching.rows(); j++) {
                        const bool nextIsSameDeg2d = curveMatching(j, 0) == curveMatching(j, 1) && curveMatching(i, 0) == curveMatching(j, 0);
                        if (!nextIsSameDeg2d) continue;
                        const bool isBackForthPair = curveMatching(i, 2) == curveMatching(j, 3) && curveMatching(i, 3) == curveMatching(j, 2);
                        if (isBackForthPair) {
                            keepEdge(i) = false;
                            keepEdge(j) = false;
                            break;
                        }
                    }
                }
            }
            int edgeIdx = 0;
            for (int i = 0; i < curveMatching.rows(); i++) {
                if (!keepEdge(i)) continue;
                curveMatching.row(edgeIdx) = curveMatching.row(i);
                edgeIdx++;
            }
            curveMatching.conservativeResize(edgeIdx, curveMatching.cols());
            numCurveElements = curveMatching.rows();

            Eigen::MatrixXi permutation(1, numCurveElements);
            utils::setLinspaced(permutation, 0);

            for (int j = 1; j < curveMatching.rows(); j++) {
                const int previdx2d0 = curveMatching(permutation(0, j-1), 0);
                const int previdx3d0 = curveMatching(permutation(0, j-1), 2);
                const int previdx2d1 = curveMatching(permutation(0, j-1), 1);
                const int previdx3d1 = curveMatching(permutation(0, j-1), 3);

                const int curridx2d0 = curveMatching(permutation(0, j), 0);
                const int curridx3d0 = curveMatching(permutation(0, j), 2);
                const int curridx3d1 = curveMatching(permutation(0, j), 3);

                //const bool notIsBackForth3d = !(previdx2d0 == previdx2d1 && previdx3d0 == curridx3d1);
                if (previdx2d1 == curridx2d0 && previdx3d1 == curridx3d0) {
                    continue;
                }
                else {
                    for (int k = j+1; k < curveMatching.rows(); k++) {
                        const int curridx2d0 = curveMatching(permutation(0, k), 0);
                        const int curridx3d0 = curveMatching(permutation(0, k), 2);
                        if (previdx2d1 == curridx2d0 && previdx3d1 == curridx3d0) {
                            const int temp = permutation(0, j);
                            permutation(0, j) = permutation(0, k);
                            permutation(0, k) = temp;
                            break;
                        }
                    }
                }
            }

            //std::cout << "after" << std::endl;
            //std::cout << curveMatching(permutation.row(0), Eigen::all) << std::endl;
            const Eigen::MatrixXi sortedCurveMatching = curveMatching(permutation.row(0), Eigen::all);
            bool failed = false;
            for (int i = 0; i < sortedCurveMatching.rows(); i++) {
                const int next = (i+1) % sortedCurveMatching.rows();
                if (sortedCurveMatching(i, 1) != sortedCurveMatching(next, 0)) failed = true;
                if (sortedCurveMatching(i, 3) != sortedCurveMatching(next, 2)) failed = true;
            }
            if (failed) {
                std::cout << "sorting failed for" << std::endl;
                std::cout << sortedCurveMatching << std::endl;
                std::cout << "-----------------" << std::endl;
            }



            matches.block(lastCurveIdx, 0, numCurveElements, 5) = sortedCurveMatching;
            sortedCycleMatchings.push_back(sortedCurveMatching);
            lastCurveIdx = i;
            curveIdx = matches(i, 4);
            numCurveElements = 1;
        }
    }
    matches.conservativeResize(nthmatch, matches.cols());


    const bool triTriOutpout = true;
    int nthsurface = 0;
    int nthelementPerSurface = 1;

    if (triTriOutpout) {
        int prevIdx = 0;
        surfaceMatching.conservativeResize(surfaceMatching.rows(), 6);
        for (const Eigen::MatrixXi& sortedCycleMatching : sortedCycleMatchings) {
            const int numEdgesInCycle = sortedCycleMatching.rows();
            if (numEdgesInCycle < 3) {
                std::cout << "sortedCycleMatching.rows() < 3, this should not happen" << std::endl;
            }

            if (numEdgesInCycle == 3) {
                for (int i = 0; i < 3; i++) {
                    surfaceMatching(nthsurface, i) = sortedCycleMatching(i, 0);
                    surfaceMatching(nthsurface, i + maxCycleLength) = sortedCycleMatching(i, 2);
                }
                nthsurface++;
            }
            else {
                // we need more than one triangle to close the surface
                for (int k = 0; k < numEdgesInCycle-2; k++) {
                    surfaceMatching(nthsurface, 0) = sortedCycleMatching(0, 0);
                    surfaceMatching(nthsurface, 0 + maxCycleLength) = sortedCycleMatching(0, 2);
                    for (int i = 1; i < 3; i++) {
                        surfaceMatching(nthsurface, i) = sortedCycleMatching(k + i, 0);
                        surfaceMatching(nthsurface, i + maxCycleLength) = sortedCycleMatching(k + i, 2);
                    }
                    nthsurface++;
                }
            }
            //std::cout << "++++  " << nthsurface - prevIdx<< std::endl;
            //std::cout << surfaceMatching.block(prevIdx, 0, nthsurface - prevIdx, 6) << std::endl;
            prevIdx = nthsurface;
        }
    }
    else {
        surfaceMatching(nthsurface, 0) = matches(0, 0);
        surfaceMatching(nthsurface, maxCycleLength) = matches(0, 2);
        for (int i = 1; i < matches.rows(); i++) {
            if (matches(i-1, 4) != matches(i, 4)) {
                nthelementPerSurface = 0;
                nthsurface++;
            }
            surfaceMatching(nthsurface, nthelementPerSurface) = matches(i, 0);
            surfaceMatching(nthsurface, nthelementPerSurface + maxCycleLength) = matches(i, 2);
            nthelementPerSurface++;

        }
    }

    surfaceMatching.conservativeResize(nthsurface, surfaceMatching.cols());

    return surfaceMatching;
}

void ProductGraphGenerators::exportInputs() {
    utils::writeMatrixToFile(EX, "EX");
    utils::writeMatrixToFile(VX, "VX");
    utils::writeMatrixToFile(EY, "EY");
    utils::writeMatrixToFile(VY, "VY");
    utils::writeMatrixToFile(FeatDiffMatrix, "FeatDiffMatrix");
}

void ProductGraphGenerators::exportInputs(Eigen::MatrixX<bool> resultsVec) {
    utils::writeMatrixToFile(resultsVec, "resultsVec");
    utils::writeMatrixToFile(EX, "EX");
    utils::writeMatrixToFile(VX, "VX");
    utils::writeMatrixToFile(EY, "EY");
    utils::writeMatrixToFile(VY, "VY");
    utils::writeMatrixToFile(FeatDiffMatrix, "FeatDiffMatrix");
}



Eigen::MatrixXd ProductGraphGenerators::computeElasticEnergy(const Eigen::MatrixXi& FX, const Eigen::MatrixXi& FY, const Eigen::MatrixXd& CX, const Eigen::MatrixXd& CY, const bool normalise) {
    if (!modelGenerated)
        generate();
    ElasticEnergy elastice(VX, VY, CX, CY, FX, FY, productspace, normalise);
    if (resolveCouple) {
        const Eigen::MatrixXd elastics = elastice.getElasticEnergy();
        Eigen::MatrixXd e(nnzColsConstraints, 1);
        e.setZero();
        for (int i = 0; i < elastics.rows(); i++) {
            e(couplingDecoder(i)) += elastics(i);
        }
        return e;
    }
    return elastice.getElasticEnergy();
}


#if WITH_OR_TOOLS
#include "ortools/base/init_google.h"
#include "ortools/init/init.h"
#include "ortools/linear_solver/linear_solver.h"
Eigen::MatrixXf ProductGraphGenerators::solveWithORTools(std::string solvername, int timelimit, Eigen::MatrixXd localenergy, float precision) {
    namespace gor = operations_research;
    if (!modelGenerated)
        generate();

    const Eigen::MatrixXi localProductspace = getProductSpace();
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> AVEC = getAVectors();
    const Eigen::MatrixXi AI = std::get<0>(AVEC);
    const Eigen::MatrixXi AJ = std::get<1>(AVEC);
    const Eigen::MatrixXi AV = std::get<2>(AVEC);
    // build a mat
    std::vector<Eigen::Triplet<int>> entriesA;
    entriesA.reserve(AI.rows());
    for (long i = 0; i < AI.rows(); i++) {
        entriesA.push_back(Eigen::Triplet<int>(AI(i, 0), AJ(i, 0), AV(i, 0)));
    }
    Eigen::SparseMatrix<int, Eigen::RowMajor> A(AI.maxCoeff()+1, localenergy.rows()); A.setFromTriplets(entriesA.begin(), entriesA.end());


    std::unique_ptr<gor::MPSolver> solver(gor::MPSolver::CreateSolver(solvername));
    std::string pdlp_params_str = "termination_criteria: { eps_optimal_relative: "+ std::to_string(precision) +"}";//eps_optimal_absolute: " + std::to_string(precision) +
    solver->SetSolverSpecificParametersAsString(pdlp_params_str);
    if (!solver) {
        std::cout << prefix << "Could not create solver " << solvername;
        exit(0);
    }
    solver->EnableOutput();
    const double infinity = solver->infinity();

    // add vars
    const long numVars = productspace.rows();
    std::vector<gor::MPVariable* const> vars; vars.reserve(numVars);
    for (int i = 0; i < localProductspace.rows(); i++) {
        vars.push_back(solver->MakeNumVar(0.0, 1, "x" + std::to_string(i)));
    }

    std::cout << solver->NumVariables() << std::endl;

    // add constraints x >= 0 and x <= 0
    for (int i = 0; i < localenergy.rows(); i++) {
        gor::MPConstraint* const ct = solver->MakeRowConstraint(0.0, 1.0, "cx" + std::to_string(i));
        ct->SetCoefficient(vars[i], 1);
    }

    if (meanProblem) {
        // 1^Tx = 1
        gor::MPConstraint* const ct = solver->MakeRowConstraint(1.0, 1.0, "callvars");
        for (int i = 0; i < localenergy.rows(); i++) {
            ct->SetCoefficient(vars[i], 1);
        }
    }


    // add surface-cycle constraints
    for (long k = 0; k < A.outerSize(); ++k) {
        const float rhs = RHS(k);
        gor::MPConstraint* const ct = solver->MakeRowConstraint(rhs, rhs, "csc" + std::to_string(k));
        for (typename Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(A, k); it; ++it) {
            ct->SetCoefficient(vars[it.index()], it.value());
       }
    }


    // add objective
    gor::MPObjective* objective = solver->MutableObjective();

    for (int i = 0; i < localenergy.rows(); i++) {
        objective->SetCoefficient(vars[i], localenergy(i));
    }
    objective->SetMinimization();


    // solve
    solver->set_time_limit((long)timelimit * 1000);
    std::cout << prefix << "Solving with " << solver->SolverVersion();
    const gor::MPSolver::ResultStatus result_status = solver->Solve();
    if (result_status != gor::MPSolver::OPTIMAL) {
      std::cout << prefix << "The problem does not have an optimal solution!\n";
      if (result_status == gor::MPSolver::FEASIBLE) {
          std::cout << prefix << "A potentially suboptimal solution was found\n";
      }
      else {
        std::cout << prefix << "The solver could not solve the problem.\n";
      }
    }
    else {
        std::cout << prefix << "Found optimal solution" << std::endl;
    }

    Eigen::MatrixXf output(localenergy.rows(), 1);
    for (int i = 0; i < localenergy.rows(); i++) {
        output(i) = vars[i]->solution_value();
    }
    return output;
}
#endif
