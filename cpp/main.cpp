// USAGE: ./app -f <Frame_Rate> -t <Total Frames>
//        ./app -f <Frame_Rate>
//        ./app -t <Total Frames>
//        ./app

#include <iostream>
#include <cstring>

/* TOFIX: causes seg fault
struct image {
    static const int x = 640, y = 480;
    int r[x*y], g[x*y], b[(x*y)];
};
*/


int main (int argc, char** argv) {
    
    // Variables:
    int framerate = 30;
    int totalframes = 600;
    // image img[60];

    int i = 0;
    int j = 0;
    int z = 0;

    // Code:
    // Set framerate if given
    if (argc == 5) {
        framerate = std::stoi(argv[2]);
        totalframes = std::stoi(argv[4]);
    } 
    if (argc == 3) {
        if (std::strcmp(argv[1], "-f") == 0) {
            framerate = std::stoi(argv[2]);
        }
        if (std::strcmp(argv[1], "-t") == 0) {
            totalframes = std::stoi(argv[2]);
        }
    }
    std::cout << "Frames to capture: " << totalframes << '\n';
    std::cout << "Frame rate       : " << framerate << '\n';
        

    // CONNECT TO CAMERA
    
    while (true) {
        for ( ; i < (j+10); i++) {
            // take and store image
            z++;
        }
        // std::cerr << i << '\n';
        // std::cerr << totalframes << '\n';

        for ( ; j < i; j++) {
            // get rgb signal channels
        }



        // Stores 60 frames
        if (i == 60){
            i = 0;
            j = 0;
        }
        
        // std::cerr << z << '\n'; 
        if (z == totalframes) {
            break;
        }
        
    }
    return 0;

}