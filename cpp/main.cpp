// USAGE: ./app -f <Frame_Rate> -t <Total Frames>
//        ./app -f <Frame_Rate>
//        ./app -t <Total Frames>
//        ./app

#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp" 
#include "opencv2/imgproc.hpp" 
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"


int main (int argc, char** argv) {
    
    // Variables:
    cv::VideoCapture capture;
    int framerate = 30;
    int numframes = 0;
    int totalframes = 600;

    int i = 0;
    int j = 0;

    

    // Code:
    // Set framerate and/or total frames to capture if given
    if (argc == 5) {
        framerate = std::stoi(argv[2]);
        totalframes = std::stoi(argv[4]);
    } 
    if (argc == 3) {
        if (argv[1] == std::string("-f")) {
            framerate = std::stoi(argv[2]);
        }
        if (argv[1] == std::string("-t")) {
            totalframes = std::stoi(argv[2]);
        }
    }
    
    std::cout << "Frames to capture: " << totalframes << '\n';
    std::cout << "Frame rate       : " << framerate << '\n';
        

    // CONNECT TO CAMERA
    
    while (true) {
        for ( ; i < (j+10); i++) {
            // take and store image
            numframes++;
        }

        for ( ; j < i; j++) {
            // get rgb signal channels
        }



        // Stores 60 frames
        if (i == 60){
            i = 0;
            j = 0;
        }
        
        if (numframes == totalframes) {
            break;
        }
        
    }

    return 0;

}