#ifndef STATS_FS_CUDA
#define STATS_FS_CUDA
#include <string>
#include <cmath>
#include <iostream>

class stats_fs_cuda
{
    private:
        float simTime;
        float nbrTime;
        float sfTime;
        float mcTime;
        float totalTime;
        int numberOfFrames;
    public:
        stats_fs_cuda() : simTime(0.0f), nbrTime(0.0f), sfTime(0.0f), mcTime(0.0f), totalTime(0.0f) {}
        void addSimTime(float time) { simTime += time; }
        void addNbrTime(float time) { nbrTime += time; }
        void addSfTime(float time) { sfTime += time; }
        void addMcTime(float time) { mcTime += time; }
        void addTotalTime(float time) { totalTime += time; }
        void setNumberOfFrames(int frames) { numberOfFrames = frames; }
        int getNumberOfFrames() const { return numberOfFrames; }
        void calculateAvgTimes() {
            if (numberOfFrames > 0) {
                totalTime = simTime + nbrTime + sfTime + mcTime;
                simTime = simTime / numberOfFrames;
                nbrTime = nbrTime / numberOfFrames;
                sfTime = sfTime / numberOfFrames;
                mcTime = mcTime / numberOfFrames;
            }
        };
        float getSimTime() const { return simTime; }
        float getNbrTime() const { return nbrTime; }
        float getSfTime() const { return sfTime; }
        float getMcTime() const { return mcTime; }
        float getTotalTime() const { return totalTime; }
        void printAvgTimes() const {
            std::cout << "Number of Frames: " << numberOfFrames << std::endl;
            std::cout << "Average Times (ms):" << std::endl;
            std::cout << "Simulation: " << simTime  << " ms" << std::endl;
            std::cout << "Neighbor Index: " << nbrTime  << " ms" << std::endl;
            std::cout << "Scalar Field: " << sfTime  << " ms" << std::endl;
            std::cout << "Marching Cubes: " << mcTime << " ms" << std::endl;
            std::cout << "Total Frame Time: " << totalTime  << " ms" << std::endl;
        }
};

#endif