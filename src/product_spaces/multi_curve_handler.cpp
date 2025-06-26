//
//  MultiCurveHandler.cpp
//  helper
//
//  Created by Paul Rötzer on 28.11.24.
//

#include "multi_curve_handler.hpp"
#include "helper/utils.hpp"
#include <tsl/robin_set.h>
#include "helper/utils.hpp"
#include "product_space.hpp"


/*function computeCombinations(...)
 Consider the edge matrix of shape X (both orientations bc triangle mesh) and the one of shape Y (single edge orientation bc 3D contour)
 -> EX is the triangle mesh
 -> EY are the contour multiple contours divided by -1 -1 rows

 -> Productspace is esentially the edges in the product graph
*/
void MultiCurveHandler::computeCombinations() {

    int numCurves = 1;
    int numEdges = 0;
    int lastCycleStartIndex = 0;
    int numEdgePairs = 0;

    tsl::robin_set<EDG> ELookup; ELookup.reserve(EY.rows());
    Eigen::MatrixXi EinvLookup(EY.rows(), 1); EinvLookup.setConstant(-1);
    Eigen::MatrixXi EPlusMinus(EY.rows(), 1); EPlusMinus.setConstant(0);
    Eigen::MatrixXi ERealIdx(EY.rows(), 2); ERealIdx.setConstant(-1);
    const long maxNumExpectedCycles = EY.rows() / 3;
    std::vector<Eigen::MatrixXi> CyclesEY; CyclesEY.reserve(maxNumExpectedCycles);
    std::vector<Eigen::MatrixXi> CyclesEYPlusMinus; CyclesEYPlusMinus.reserve(maxNumExpectedCycles);
    std::vector<Eigen::MatrixXi> CEYtoRealEY; CEYtoRealEY.reserve(maxNumExpectedCycles);
    Eigen::MatrixXi cycleStartEndIndex(maxNumExpectedCycles, 2);
    for (int e = 0; e < EY.rows(); e++) {
        ERealIdx(e, 0) = numEdges;
        if (EY(e, 0) == -1 && EY(e, 1) == -1) {
            numCurves++;

            const int cycleLength = e - lastCycleStartIndex;
            CyclesEY.push_back(EY.block(lastCycleStartIndex, 0, cycleLength, 2));
            CyclesEYPlusMinus.push_back(EPlusMinus.block(lastCycleStartIndex, 0, cycleLength, 1));
            CEYtoRealEY.push_back(ERealIdx.block(lastCycleStartIndex, 0, cycleLength, 2));
            lastCycleStartIndex = e + 1;
        }
        else {
            const int edgeIdx = findEdge(ELookup, EDG(EY(e, 1), EY(e, 0)));
            if (edgeIdx == -1) {
                ELookup.insert(EDG(EY(e, 0), EY(e, 1), e));
                EPlusMinus(e) = +1;
            }
            else {
                numEdgePairs++;
                if (e < edgeIdx) std::cout << prefix << "e < edgeIdx, this should not happen" << std::endl;
                EinvLookup(e, 0) = edgeIdx;
                EPlusMinus(e) = -1;
                ERealIdx(e, 1) = ERealIdx(edgeIdx, 0);
                if (EinvLookup(edgeIdx, 0) == -1) {
                    EinvLookup(edgeIdx, 0) = e;
                }
                else {
                    std::cout << prefix << "EinvLookup(edgeIdx) != -1, this should not happen" << std::endl;
                }
            }

            numEdges++;
        }
    }
    const int lastCycleLength = EY.rows() - lastCycleStartIndex;
    CyclesEY.push_back(EY.block(lastCycleStartIndex, 0, lastCycleLength, 2));
    CyclesEYPlusMinus.push_back(EPlusMinus.block(lastCycleStartIndex, 0, lastCycleLength, 1));
    CEYtoRealEY.push_back(ERealIdx.block(lastCycleStartIndex, 0, lastCycleLength, 2));
    if (verbose) std::cout << prefix << "Found >> " << numCurves << " << cycles, >> " << numEdgePairs << " << edge pairs and >> " << numEdges << " << edges" << std::endl;
    numContours = numCurves;

    if (DEBUG_MCH) {
        std::cout << "Cycles: " << std::endl;
        for (const Eigen::MatrixXi& cyc : CyclesEY) {
            std::cout << cyc << std::endl;
            std::cout << "++++++" << std::endl;
        }

        std::cout << "Pair edges: " << std::endl;
        for (int e = 0; e < EinvLookup.rows(); e++) {
            if (EinvLookup(e, 0) != -1) {
                if (e > EinvLookup(e, 0)) {
                    std::cout << "neg: ";
                }
                std::cout << EY.row(e) << ", " <<  EY.row(EinvLookup(e, 0)) << std::endl;
            }
        }
    }



    const int numVerticesX = EX.maxCoeff() + 1;
    const long numIntraLayerEdges = numEdges * EX.rows() * maxDepth;
    const long DISTBOUNDCONSTANT = NEW_PLUS_MINUS ? std::pow(maxDepth / 2.0 + 1.0, 2.0)  :  (1 + maxDepth);
    long numInterLayerEdges = numEdges * (EX.rows() + numVerticesX) * DISTBOUNDCONSTANT;

    const long numProductEdges = numIntraLayerEdges + numInterLayerEdges;
    productspace = Eigen::MatrixXi(numProductEdges, 7);
    productspace.setConstant(-1);
    SRC_IDs = Eigen::MatrixXi(numProductEdges, 1);
    TRGT_IDs = Eigen::MatrixXi(numProductEdges, 1);
    const long numEdgesPerCoupleBlock = EX.rows() * maxDepth + (EX.rows() + numVerticesX) * DISTBOUNDCONSTANT;
    const long numCouples = numEdgePairs * numEdgesPerCoupleBlock;
    PLUSMINUSDIR = Eigen::MatrixXi(numCouples, 2);
    piCycle = Eigen::MatrixXi(numProductEdges, 1);
    piCycle.setConstant(-1);

    long oldEdgeIdx = 0;
    long oldMaxIdx = 0;
    long oldCoupleIdx = 0;
    maxCycleLength = 0;
    for (int c = 0; c < CyclesEY.size(); c++) {
        if (CyclesEY.at(c).rows() > maxCycleLength) {
            maxCycleLength = CyclesEY.at(c).rows();
        }
        const bool pruneIntraLayer = true;
        ProductSpace p(EX, CyclesEY.at(c), pruneIntraLayer);
        p.setMaxDepth(maxDepth);
        p.setPlusMinusDir(CyclesEYPlusMinus.at(c));
        const Eigen::MatrixXi projectiononCycleEY = p.getPiEy();

        const long numNewEdges = p.getProductSpace().rows();

        productspace.block(oldEdgeIdx, 0, numNewEdges, p.getProductSpace().cols()) = p.getProductSpace();
        Eigen::MatrixXi cycIndicator(numNewEdges, 1) ; cycIndicator.setConstant(c);
        productspace.block(oldEdgeIdx, p.getProductSpace().cols(), numNewEdges, 1) = cycIndicator;
        SRC_IDs.block(oldEdgeIdx, 0, numNewEdges, 1) = p.getSRCIds().array() + oldMaxIdx;
        TRGT_IDs.block(oldEdgeIdx, 0, numNewEdges, 1) = p.getTRGTIds().array() + oldMaxIdx;
        piCycle.block(oldEdgeIdx, 0, numNewEdges, 1).setConstant(c);

        /*/std::cout << projectiononCycleEY << std::endl;
        Eigen::MatrixXi temp(p.getProductSpace().rows(), 8); temp.setConstant(-1);
        temp.block(0, 0, p.getProductSpace().rows(), 5) = p.getProductSpace();
        temp.block(0, 6, p.getProductSpace().rows(), 1) = p.getSRCIds();
        temp.block(0, 7, p.getProductSpace().rows(), 1) = p.getTRGTIds();
        std::cout << temp << std::endl;
        std::cout << "p.getProductSpace()" << std::endl;
         */


        /*std::cout << "-+-+-+-+-+-+-+--+-+-+--+" << std::endl;
        std::cout <<  productspace.block(0, 0, oldEdgeIdx + numNewEdges, p.getProductSpace().cols()) << std::endl;
        std::cout << "-+-+-+-+-+-+-+--+-+-+--+" << std::endl;
        */
        if ((CyclesEYPlusMinus.at(c).array() == -1).any()) {
            for (int i = 0; i < CyclesEYPlusMinus.at(c).rows(); i++) {
                if ( CyclesEYPlusMinus.at(c)(i) == -1) {
                    const long minusStartIdx = CEYtoRealEY.at(c)(i, 0); assert(minusStartIdx != -1);
                    const long plusStartIdx = CEYtoRealEY.at(c)(i, 1); assert(plusStartIdx != -1);
                    const long minusStartIdxPspace = minusStartIdx * numEdgesPerCoupleBlock;
                    const long plusStartIdxPspace = plusStartIdx * numEdgesPerCoupleBlock;
                    for (long j = 0; j < numEdgesPerCoupleBlock; j++) {
                        const long idxPlusPEdge = plusStartIdxPspace + j;
                        const long idxMinusPEdge = minusStartIdxPspace + j;
                        PLUSMINUSDIR(j + oldCoupleIdx, 0) = idxPlusPEdge;
                        PLUSMINUSDIR(j + oldCoupleIdx, 1) = idxMinusPEdge;

                        if (DEBUG_MCH) {
                            const Eigen::Vector<int, 6> reverse(( Eigen::Vector<int, 6>() << 1, 0, 3, 2, 5, 4).finished());
                            if (!utils::allEqual(productspace.row(idxPlusPEdge), productspace(idxMinusPEdge, reverse))) {
                                std::cout << "coupling not correct for pedge " << productspace.row(idxPlusPEdge) << " <--> " << productspace.row(idxMinusPEdge) << std::endl;
                            }
                        }
                    }
                    oldCoupleIdx += numEdgesPerCoupleBlock;
                }
            }
        }

        oldEdgeIdx += numNewEdges;
        oldMaxIdx += p.getSRCIds().maxCoeff() + 1;
    }
    combosComputed = true;
}



MultiCurveHandler::MultiCurveHandler(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, const bool pruneIntralyer, const int maxDepth) : EX(EX), EY(EY), maxDepth(maxDepth) {
    combosComputed = false;
    numContours = 1;
    pruneIntralayerEdges = pruneIntralyer;
    verbose = true;
    prefix = "[MCHandler] ";
    if (NEW_PLUS_MINUS && maxDepth % 2 != 0) {
        std::cout << prefix << "maxDepth needs to be a multiple of two, exiting" << std::endl;
        exit(0);
    }
}

MultiCurveHandler::MultiCurveHandler(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY) : MultiCurveHandler(EX, EY, false, 0) {
}


Eigen::MatrixXi MultiCurveHandler::getProductSpace() {
    if (!combosComputed) {
        computeCombinations();
    }
    return productspace;
}

Eigen::MatrixXi MultiCurveHandler::getPiEy() {
    if (!combosComputed) {
        computeCombinations();
    }
    return piEY;
}

Eigen::MatrixXi MultiCurveHandler::getPiCycle() {
    if (!combosComputed) {
        computeCombinations();
    }
    return piCycle;
}

Eigen::MatrixXi MultiCurveHandler::getSRCIds() {
    if (!combosComputed) {
        computeCombinations();
    }
    return SRC_IDs;
}

Eigen::MatrixXi MultiCurveHandler::getTRGTIds() {
    if (!combosComputed) {
        computeCombinations();
    }
    return TRGT_IDs;
}

Eigen::MatrixXi MultiCurveHandler::getPlusMinusDir() {
    if (!combosComputed) {
        computeCombinations();
    }
    return PLUSMINUSDIR;
}

int MultiCurveHandler::getNumContours() const {
    return numContours;
}

int MultiCurveHandler::getMaxCycleLength() {
    if (!combosComputed) {
        computeCombinations();
    }
    return maxCycleLength;
}

std::vector<tsl::robin_set<long>> MultiCurveHandler::getBranchGraph() {
    return branchGraph;
}
