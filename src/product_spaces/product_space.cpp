//
//  combinations.cpp
//  helper
//
//  Created by Paul Rötzer on 28.04.21.
//

#include "product_space.hpp"
#include "helper/utils.hpp"


/*function computeCombinations(...)
 Consider the edge matrix of shape X (both orientations bc triangle mesh) and the one of shape Y (single edge orientation bc 3D contour)
 -> EX is the triangle mesh
 -> EY is the contour (just a single contour, no multiple)

 -> Productspace is esentially the edges in the product graph
*/
void ProductSpace::computeCombinations() {
    const int numVerticesX = EX.maxCoeff() + 1;
    //const long numVerticesY = EY.rows();
    long numIntraLayerEdges, numInterLayerEdges;
    if (pruneIntralayerEdges) {
        // maxDepth == 0 means no intralayer
        numIntraLayerEdges = EY.rows() * EX.rows() * maxDepth;
        if (NEW_PLUS_MINUS) {
            numInterLayerEdges = EY.rows() * (EX.rows() + numVerticesX) * std::pow(maxDepth / 2.0 + 1.0, 2.0);
        }
        else {
            numInterLayerEdges = EY.rows() * (EX.rows() + numVerticesX) * (1 + maxDepth);
        }
    }
    else {
        numIntraLayerEdges = EY.rows() * EX.rows();
        numInterLayerEdges = EY.rows() * (EX.rows() + numVerticesX);
    }

    const long numElementsPSpace = numIntraLayerEdges + numInterLayerEdges;
    productspace = Eigen::MatrixXi(numElementsPSpace, 6);
    SRC_IDs  = Eigen::MatrixXi(numElementsPSpace, 1);
    TRGT_IDs = Eigen::MatrixXi(numElementsPSpace, 1);
    productspace.setZero(); SRC_IDs.setZero(); TRGT_IDs.setZero();
    piEY = Eigen::MatrixXi(numElementsPSpace, 2);
    piEY.setZero();

    const long numEdgesY = EY.rows();
    long numadded = 0;
    int currentstartidx = 0;

    if (pruneIntralayerEdges) {
        int numIntralayer = 0;
        int numInterDeg = 0;
        int numInterNonDeg = 0;
        for (int i = 0; i < numEdgesY; i++) {
            const long numAddedBefore = numadded;
            const bool isPlusDirection = plusMinusDir.size() == 0 ? true : plusMinusDir(i) > 0;
            if (NEW_PLUS_MINUS) {
                /*
                    O O O -> P P P, now  O O O -> P P P
                    ^        ^             ^        ^        BASE
                 */
                
                const long middle = maxDepth / 2;
                const long target_id_base_idx = ((i+1) % numEdgesY) * numVerticesX * (1 + maxDepth) + numVerticesX * middle;
                const long source_id_base_idx = i * numVerticesX * (1 + maxDepth) + numVerticesX * middle;

                /* Wiring for plus and minus different for duplicating intralayers, here maxDepth = 2, maxDepth NEEDS to be multiple of 2
                                     --------------<---------------
                  Minus Dir:        /            ---------<--------\
                                   /            /                   \
                               ###/         +-+/         +-+         \+-+
                               #0#----<-----|0|----<-----|1|----<-----|1|
                               ###\         +-+         /+-+          +-+
                                   \                   /
                                    ---------<--------/

                                     -------------->---------------
                  Plus Dir:         /            --------->--------\
                                   /            /                   \
                               +-+/         +-+/         +-+         \###
                               |0|---->-----|0|---->-----|1|---->-----#1#
                               +-+\         +-+         /+-+          ###
                                   \                   /
                                    --------->--------/

                 */
                Eigen::MatrixXi currentEY = EY.row(i);

                // intralayer edges (via duplicated layers)
                for (int d = 0; d < middle; d++) { // middle = maxDepth / 2
                    const int dinv = middle - 1 - d; // d is maximum middle - 1

                    for (int vertexY = 0; vertexY < 2; vertexY++) { // first iter here is 0 --> 0, second iter is 1 --> 1
                        for (int j = 0; j < EX.rows(); j++) {

                            if (isPlusDirection) {
                                if (vertexY == 0) {
                                    productspace.row(numadded) << currentEY(vertexY), currentEY(vertexY), EX(j, 0), EX(j, 1), d, d+1;
                                    SRC_IDs(numadded)  = (int) ( source_id_base_idx +  d    * numVerticesX + EX(j, 0) );
                                    TRGT_IDs(numadded) = (int) ( source_id_base_idx + (d+1) * numVerticesX + EX(j, 1) );
                                }
                                else {
                                    productspace.row(numadded) << currentEY(vertexY), currentEY(vertexY), EX(j, 0), EX(j, 1), middle - d, middle - d - 1;
                                    SRC_IDs(numadded)  = (int) ( target_id_base_idx - (middle - d    ) * numVerticesX + EX(j, 0) );
                                    TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - d - 1) * numVerticesX + EX(j, 1) );
                                }
                            }
                            else {
                                // vertexY is inverted here!!!!

                                if (vertexY == 0) {
                                    productspace.row(numadded) << currentEY(1-vertexY), currentEY(1-vertexY), EX(j, 1), EX(j, 0), dinv+1, dinv;
                                    SRC_IDs(numadded)  = (int) ( target_id_base_idx - (middle - dinv    ) * numVerticesX + EX(j, 1) );
                                    TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - dinv - 1) * numVerticesX + EX(j, 0) );
                                }
                                else {
                                    productspace.row(numadded) << currentEY(1-vertexY), currentEY(1-vertexY), EX(j, 1), EX(j, 0), dinv, dinv+1;
                                    SRC_IDs(numadded)  = (int) ( source_id_base_idx +  (dinv)  * numVerticesX + EX(j, 1) );
                                    TRGT_IDs(numadded) = (int) ( source_id_base_idx +  (dinv+1)   * numVerticesX + EX(j, 0) );
                                }
                            }

                            piEY(numadded, 0) = i;
                            piEY(numadded, 1) = i;
                            numadded++;
                            numIntralayer++;
                        }
                    }
                }


                // interlayer edges (from every duplicate to every other duplicate)
                if (isPlusDirection) {
                    for (int d = 0; d <= middle; d++) { // middle = maxDepth / 2
                        const int dinv  = middle -  d;

                        for (int dd = 0; dd <= middle; dd++) { // middle = maxDepth / 2
                            const int ddinv = middle - dd;
                            // non degenerates
                            for (int j = 0; j < EX.rows(); j++) {
                                productspace.row(numadded) << currentEY(0), currentEY(1), EX(j, 0), EX(j, 1), d, middle - dd;
                                SRC_IDs(numadded)  = (int) ( source_id_base_idx +            d  * numVerticesX + EX(j, 0) );
                                TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - dd) * numVerticesX + EX(j, 1) );
                                piEY(numadded, 0) = i;
                                piEY(numadded, 1) = i;
                                numadded++;
                                numInterNonDeg++;
                            }

                            // degenerates
                            for (int j = 0; j < numVerticesX; j++) {
                                productspace.row(numadded) << currentEY(0), currentEY(1), j, j, d, middle - dd;
                                SRC_IDs(numadded)  = (int) ( source_id_base_idx +            d  * numVerticesX + j );
                                TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - dd) * numVerticesX + j );

                                piEY(numadded, 0) = i;
                                piEY(numadded, 1) = i;
                                numadded++;
                                numInterDeg++;
                            }
                        }
                    }
                }
                /*
                 !isPlusDirection
                 */
                else {
                    for (int dd = 0; dd <= middle; dd++) { // middle = maxDepth / 2
                        const int ddinv = middle - dd;
                        for (int d = 0; d <= middle; d++) { // middle = maxDepth / 2
                            const int dinv  = middle -  d;
                            // non degenerates
                            for (int j = 0; j < EX.rows(); j++) {
                                productspace.row(numadded) << currentEY(0), currentEY(1), EX(j, 1), EX(j, 0), dinv, middle - ddinv;
                                SRC_IDs(numadded)  = (int) ( source_id_base_idx +            dinv  * numVerticesX + EX(j, 1) );
                                TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - ddinv) * numVerticesX + EX(j, 0) );
                                piEY(numadded, 0) = i;
                                piEY(numadded, 1) = i;
                                numadded++;
                                numInterNonDeg++;
                            }

                            // degenerates
                            for (int j = 0; j < numVerticesX; j++) {
                                productspace.row(numadded) << currentEY(0), currentEY(1), j, j, dinv, middle - ddinv;
                                SRC_IDs(numadded)  = (int) ( source_id_base_idx +            dinv  * numVerticesX + j );
                                TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - ddinv) * numVerticesX + j );
                                piEY(numadded, 0) = i;
                                piEY(numadded, 1) = i;
                                numadded++;
                                numInterDeg++;
                            }

                        }
                    }
                }

                /*/ interlayer edges (from every duplicate to every other duplicate)
                for (int d = 0; d <= middle; d++) { // middle = maxDepth / 2
                    const int dinv  = middle -  d;

                    for (int dd = 0; dd <= middle; dd++) { // middle = maxDepth / 2
                        const int ddinv = middle - dd;

                        // non degenerates
                        for (int j = 0; j < EX.rows(); j++) {
                            if (isPlusDirection) {
                                productspace.row(numadded) << currentEY(0), currentEY(1), EX(j, 0), EX(j, 1), d, middle - dd;
                                SRC_IDs(numadded)  = (int) ( source_id_base_idx +            d  * numVerticesX + EX(j, 0) );
                                TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - dd) * numVerticesX + EX(j, 1) );
                            }
                            else {
                                productspace.row(numadded) << currentEY(0), currentEY(1), EX(j, 1), EX(j, 0), dinv, middle - ddinv;
                                SRC_IDs(numadded)  = (int) ( source_id_base_idx +            dinv  * numVerticesX + EX(j, 1) );
                                TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - ddinv) * numVerticesX + EX(j, 0) );
                            }
                            piEY(numadded, 0) = i;
                            piEY(numadded, 1) = i;
                            numadded++;
                            numInterNonDeg++;
                        }

                        // degenerates
                        for (int j = 0; j < numVerticesX; j++) {
                            if (isPlusDirection) {
                                productspace.row(numadded) << currentEY(0), currentEY(1), j, j, d, middle - dd;
                                SRC_IDs(numadded)  = (int) ( source_id_base_idx +            d  * numVerticesX + j );
                                TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - dd) * numVerticesX + j );
                            }
                            else {
                                productspace.row(numadded) << currentEY(0), currentEY(1), j, j, dinv, middle - ddinv;
                                SRC_IDs(numadded)  = (int) ( source_id_base_idx +            dinv  * numVerticesX + j );
                                TRGT_IDs(numadded) = (int) ( target_id_base_idx - (middle - ddinv) * numVerticesX + j );
                            }
                            piEY(numadded, 0) = i;
                            piEY(numadded, 1) = i;
                            numadded++;
                            numInterDeg++;
                        }
                    }
                }*/
            }
            /*



             ONLY USED IF NEW_PLUS_MINUS is false




             */
            else {
                const long target_id_base_idx = ((i+1) % numEdgesY) * numVerticesX * (1 + maxDepth);
                const long source_id_base_idx = i * numVerticesX * (1 + maxDepth);
                /* Wiring for plus and minus different for duplicating intralayers, here maxDepth = 2
                                     --------------<---------------
                  Minus Dir:        /            ---------<--------\
                                   /            /                   \
                               ###/         +-+/         +-+         \+-+
                               #0#----<-----|0|----<-----|0|----<-----|1|
                               ###          +-+          +-+          +-+
                                     -------------->---------------
                  Plus Dir:         /            --------->--------\
                                   /            /                   \
                               +-+/         +-+/         +-+         \###
                               |0|---->-----|0|---->-----|0|---->-----#1#
                               +-+          +-+          +-+          ###
                 */
                Eigen::MatrixXi currentEY = EY.row(i);

                for (int d = 0; d < maxDepth+1; d++) {
                    const int indexOffset = i * numVerticesX * (1 + maxDepth) + numVerticesX * d;
                    const int indexOffsetMinusDir = i * numVerticesX * (1 + maxDepth) + numVerticesX * maxDepth - numVerticesX * d;


                    for (int j = 0; j < EX.rows(); j++) {
                        // intralayer (via duplicated layers)
                        if (d < maxDepth) {
                            if (isPlusDirection) {
                                productspace.row(numadded) << currentEY(0), currentEY(0), EX(j, 0), EX(j, 1), d;
                                SRC_IDs(numadded)  = (int) ( indexOffset + EX(j, 0) );
                                TRGT_IDs(numadded) = (int) ( indexOffset + numVerticesX + EX(j, 1) );
                            }
                            else {// TODO:
                                productspace.row(numadded) << currentEY(1), currentEY(1), EX(j, 1), EX(j, 0), d;
                                SRC_IDs(numadded)  = (int) ( indexOffsetMinusDir + EX(j, 1) );
                                if (d == 0 && i == numEdgesY-1)
                                    TRGT_IDs(numadded) = (int) ( EX(j, 0) );
                                else
                                    TRGT_IDs(numadded) = (int) ( indexOffsetMinusDir + numVerticesX + EX(j, 0) );
                            }

                            piEY(numadded, 0) = i;
                            piEY(numadded, 1) = i;
                            numadded++;
                            numIntralayer++;
                        }

                        // interlayer (non-degenerate from every duplicated layer)
                        if (isPlusDirection) {
                            productspace.row(numadded) << currentEY(0), currentEY(1), EX(j, 0), EX(j, 1), d;
                            SRC_IDs(numadded)  = (int) ( indexOffset + EX(j, 0) );
                            TRGT_IDs(numadded) = (int) ( target_id_base_idx + EX(j, 1) );
                        }
                        else {// TODO:
                            productspace.row(numadded) << currentEY(0), currentEY(1), EX(j, 1), EX(j, 0), d;
                            SRC_IDs(numadded)  = (int) ( source_id_base_idx + EX(j, 1) );
                            if (d == 0 && i == numEdgesY-1)
                                TRGT_IDs(numadded) = (int) ( EX(j, 0) );
                            else
                                TRGT_IDs(numadded) = (int) ( indexOffsetMinusDir + numVerticesX + EX(j, 0) );
                        }
                        piEY(numadded, 0) = i;
                        if (i+1 < numEdgesY) {
                            if (EY(i+1, 0) == -1) {
                                piEY(numadded, 1) = currentstartidx;
                            }
                            else {
                                piEY(numadded, 1) = i+1;
                            }
                        }
                        else {
                            piEY(numadded, 1) = currentstartidx;
                        }
                        numadded++;
                        numInterNonDeg++;
                    }


                    // interlayer (degenerate)
                    for (int j = 0; j < numVerticesX; j++) {
                        productspace.row(numadded) << currentEY(0), currentEY(1), j, j, d;
                        if (isPlusDirection) {
                            SRC_IDs(numadded)  = (int) ( indexOffset + j );
                            TRGT_IDs(numadded) = (int) ( target_id_base_idx + j);
                        }
                        else { // TODO:
                            SRC_IDs(numadded)  = (int) ( source_id_base_idx + j );
                            if (d == 0 && i == numEdgesY-1)
                                TRGT_IDs(numadded) = (int) ( j );
                            else
                                TRGT_IDs(numadded) = (int) ( indexOffsetMinusDir + numVerticesX + j);
                        }
                        piEY(numadded, 0) = i;
                        if (i+1 < numEdgesY) {
                            piEY(numadded, 1) = i+1;
                        }
                        else {
                            piEY(numadded, 1) = currentstartidx;
                        }
                        numadded++;
                        numInterDeg++;
                    }
                }
            }
        }
        if (DEBUG_COMBINATIONS) {
            Eigen::MatrixXi numEdgesPerVertex(numEdgesY * numVerticesX * (maxDepth + 1), 2); numEdgesPerVertex.setZero();
            Eigen::MatrixXi perVertexInAndOutIds(numEdgesY * numVerticesX * (maxDepth + 1), 2); perVertexInAndOutIds.setConstant(-1);
            for (int iii = 0; iii < numadded; iii++) {
                numEdgesPerVertex(SRC_IDs(iii), 0) += 1;
                numEdgesPerVertex(TRGT_IDs(iii), 1) += 1;

                if (perVertexInAndOutIds(SRC_IDs(iii), 0) == -1) {
                    perVertexInAndOutIds.row(SRC_IDs(iii)) << productspace(iii, 0), productspace(iii, 2);
                }
                else {
                    const bool firstEqual  = perVertexInAndOutIds(SRC_IDs(iii), 0) == productspace(iii, 0);
                    const bool secondEqual = perVertexInAndOutIds(SRC_IDs(iii), 1) == productspace(iii, 2);
                    if (! (firstEqual && secondEqual)) {
                        std::cout << "Err src: " << productspace.row(iii) << " epxected " << perVertexInAndOutIds.row(SRC_IDs(iii)) << std::endl;
                    }
                }

                if (perVertexInAndOutIds(TRGT_IDs(iii), 0) == -1) {
                    perVertexInAndOutIds.row(TRGT_IDs(iii)) << productspace(iii, 1), productspace(iii, 3);
                }
                else {
                    const bool firstEqual  = perVertexInAndOutIds(TRGT_IDs(iii), 0) == productspace(iii, 1);
                    const bool secondEqual = perVertexInAndOutIds(TRGT_IDs(iii), 1) == productspace(iii, 3);
                    if (! (firstEqual && secondEqual)) {
                        std::cout << "Err trgt: " << productspace.row(iii) << " epxected " << perVertexInAndOutIds.row(TRGT_IDs(iii)) << std::endl;
                    }
                }

            }

            

            std::cout << "numEdgesPerVertex" << std::endl;
            std::cout << numEdgesPerVertex.transpose() << std::endl;

            std::cout << "numInterDeg " << numInterDeg << std::endl;
            std::cout << "numInterNonDeg " << numInterNonDeg << std::endl;
            std::cout << "numIntralayer " << numIntralayer << std::endl;
        }
    }
    /*
     OLD IMPL
     */
    else {

        for (int i = 0; i < numEdgesY; i++) {

            const long target_id_base_idx = (i+1) % numEdgesY;
            for (int j = 0; j < EX.rows(); j++) {

                // intralayer
                if (!pruneIntralayerEdges) {
                    productspace.row(numadded) << EY(i, 0), EY(i, 0), EX(j, 0), EX(j, 1), 0;
                    SRC_IDs(numadded)  = (int) ( i * numVerticesX + EX(j, 0) );
                    TRGT_IDs(numadded) = (int) ( i * numVerticesX + EX(j, 1) );
                    piEY(numadded, 0) = i;
                    piEY(numadded, 1) = i;
                    numadded++;
                }

                // interlayer
                productspace.row(numadded) << EY(i, 0), EY(i, 1), EX(j, 0), EX(j, 1), 0;
                SRC_IDs(numadded)  = (int) ( i * numVerticesX + EX(j, 0) );
                TRGT_IDs(numadded) = (int) ( target_id_base_idx * numVerticesX + EX(j, 1) );
                piEY(numadded, 0) = i;
                if (i+1 < numEdgesY) {
                    if (EY(i+1, 0) == -1) {
                        piEY(numadded, 1) = currentstartidx;
                    }
                    else {
                        piEY(numadded, 1) = i+1;
                    }
                }
                else {
                    piEY(numadded, 1) = currentstartidx;
                }
                numadded++;

            }

            for (int j = 0; j < numVerticesX; j++) {
                // interlayer
                productspace.row(numadded) << EY(i, 0), EY(i, 1), j, j, 0;
                SRC_IDs(numadded)  = (int) ( i * numVerticesX + j );
                TRGT_IDs(numadded) = (int) ( target_id_base_idx * numVerticesX + j);
                piEY(numadded, 0) = i;
                if (i+1 < numEdgesY) {
                    /*if (EY(i+1, 0) == -1) {
                     piEY(numadded, 1) = currentstartidx;
                     }
                     else {*/
                    piEY(numadded, 1) = i+1;
                    //}
                }
                else {
                    piEY(numadded, 1) = currentstartidx;
                }
                numadded++;
            }

        }
    }
    if (DEBUG_COMBINATIONS) std::cout << "[COMBOS] Detected " << numContours << " closed contours" << std::endl;
    productspace.conservativeResize(numadded, productspace.cols());
    SRC_IDs.conservativeResize(numadded, 1);
    TRGT_IDs.conservativeResize(numadded, 1);
    piEY.conservativeResize(numadded, 2);
    combosComputed = true;

    if (DEBUG_COMBINATIONS && !pruneIntralayerEdges) {
        const int numNodesPerLayer = numVerticesX;
        for (long i = 0; i < SRC_IDs.rows(); i++) {
            const int srcId  = SRC_IDs(i);
            const int trgtId = TRGT_IDs(i);
            const Eigen::MatrixXi pedge = productspace.row(i);
            const int src2d = srcId / numNodesPerLayer;
            const int src3d = srcId - src2d * numNodesPerLayer;
            const int trgt2d = trgtId / numNodesPerLayer;
            const int trgt3d = trgtId - trgt2d * numNodesPerLayer;
            Eigen::MatrixXi pedgeReconstructed(1, 4);
            pedgeReconstructed << src2d, trgt2d, src3d, trgt3d;
            if (!utils::allEqual(pedge, pedgeReconstructed)) {
                std::cout << "Original: " << pedge << std::endl;
                std::cout << "Reconstr: " << pedgeReconstructed << std::endl;
            }
        }
    }

    if (pruneIntralayerEdges) {
        branchGraph = std::vector<tsl::robin_set<long>>();
        branchGraph.reserve(numVerticesX);
        for (int j = 0; j < numVerticesX; j++) {
            branchGraph.push_back(tsl::robin_set<long>());
        }
        for (int j = 0; j < EX.rows(); j++) {
            branchGraph.at(EX(j, 0)).insert(EX(j, 1));
        }
    }
}



ProductSpace::ProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, const bool pruneIntralyer) : EX(EX), EY(EY) {
    combosComputed = false;
    numContours = 1;
    pruneIntralayerEdges = pruneIntralyer;
    maxDepth = 1;
}

ProductSpace::ProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY) : ProductSpace(EX, EY, false) {
}


Eigen::MatrixXi ProductSpace::getProductSpace() {
    if (!combosComputed) {
        computeCombinations();
    }
    return productspace;
}

Eigen::MatrixXi ProductSpace::getPiEy() {
    if (!combosComputed) {
        computeCombinations();
    }
    return piEY;
}

Eigen::MatrixXi ProductSpace::getSRCIds() {
    if (!combosComputed) {
        computeCombinations();
    }
    return SRC_IDs;
}

Eigen::MatrixXi ProductSpace::getTRGTIds() {
    if (!combosComputed) {
        computeCombinations();
    }
    return TRGT_IDs;
}


int ProductSpace::getNumContours() const {
    return numContours;
}

std::vector<tsl::robin_set<long>> ProductSpace::getBranchGraph() {
    return branchGraph;
}

void ProductSpace::setPlusMinusDir(const Eigen::MatrixXi& iPlusMinusDir) {
    plusMinusDir = iPlusMinusDir;
}

void ProductSpace::setMaxDepth(const int iMaxDepth) {
    // 0 means no intralayer connections
    maxDepth = iMaxDepth;
}
