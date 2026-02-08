/** @file assignment.cu
 * @author Eric Jameson
 * @brief CUDA file containing Module 3 assignment for EN605.617
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

/** @brief Input image file name */
#define INPUT_FILE "jhu.ppm"
/** @brief Input image size in bytes (excluding header) */
#define IMAGE_SIZE 1920 * 1080 * 3

/** @brief Output file name for the host image for part 1 */
#define OUTPUT_FILE_HOST "jhu_negative_host.ppm"
/** @brief Output file name for the host image for part 2 */
#define OUTPUT_FILE_HOST_2 "jhu_channel_branch_host.ppm"
/** @brief Output file name for the device image for part 1 */
#define OUTPUT_FILE_DEVICE "jhu_negative_device.ppm"
/** @brief Output file name for the device image for part 2 */
#define OUTPUT_FILE_DEVICE_2 "jhu_channel_branch_device.ppm"

/** @brief Helper to print the header with thread and block information.
 *
 * @param total_threads Requested (or adjusted) number of total threads
 * @param block_size Requested number of threads per block
 * @param num_blocks Requested number of blocks
 * @param warning Flag to indicated if the number of total threads has been adjusted to accomodate
 * the requested block_size
 */
void print_header(int total_threads, int block_size, int num_blocks, bool warning) {
    std::cout << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl
              << "| EN605.617 Module 3 Assignment                                  Eric Jameson |"
              << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl;

    std::cout << "| Total Threads:    "
              << std::setw(80 - 23 - std::to_string(total_threads).length()) << " " << total_threads
              << " |" << std::endl
              << "| Block Size:       " << std::setw(80 - 23 - std::to_string(block_size).length())
              << " " << block_size << " |" << std::endl
              << "| Number of Blocks: " << std::setw(80 - 23 - std::to_string(num_blocks).length())
              << " " << num_blocks << " |" << std::endl;

    if (warning) {
        std::cout
            << "| " << std::setw(77) << " |" << std::endl
            << "| Warning: Chosen thread count is not evenly divisible by the block size, so  |"
            << std::endl
            << "| the total number of threads has been rounded up to " << total_threads << "."
            << std::setw(80 - 55 - std::to_string(total_threads).length()) << " |" << std::endl;
    }

    std::cout << "+-----------------------------------------------------------------------------+"
              << std::endl;
}

/** @brief Helper to print the timing information for each part.
 *
 * @param host_microseconds The number of microseconds it took the host code to execute
 * @param device_microseconds The number of microseconds it took the device code to execute
 */
void print_timings(int host_microseconds, int device_microseconds) {
    int max_width = std::max(
        {std::to_string(host_microseconds).length(), std::to_string(device_microseconds).length()});
    std::cout << "| Time elapsed   (Host): " << std::setw(80 - 26 - max_width) << host_microseconds
              << " μs |" << std::endl
              << "| Time elapsed (Device): " << std::setw(80 - 26 - max_width)
              << device_microseconds << " μs |" << std::endl;
}

/** @brief Host code to mirror the bytes corresponding to the R/G/B values of an image.
 *
 * @param[out] dst Destination to write the inverted values to
 * @param src Source to read the original input from
 * @param image_size Size of the \p src array
 */
void invert_pixels_host(unsigned char* dst, const unsigned char* src, const size_t image_size) {
    for (size_t i = 0; i < image_size; i++) {
        dst[i] = 255 - src[i];
    }
}

/** @brief Device kernel to mirror the bytes corresponding to the R/G/B values of an image.
 *
 * @param[out] dst Destination to write the inverted values to
 * @param src Source to read the original input from
 * @param image_size Size of the \p src array
 */
__global__ void invert_pixels_device(unsigned char* dst, const unsigned char* src,
                                     const size_t image_size) {
    const unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < image_size) {
        dst[thread_idx] = 255 - src[thread_idx];
    }
}

/** @brief Host code to manipulate the R/G/B values in an image based on their channel.
 *
 * @param[out] dst Destination to write the manipulated values to
 * @param src Source to read the original input from
 * @param image_size Size of the \p src array
 */
void channel_branch_host(unsigned char* dst, const unsigned char* src, const size_t image_size) {
    for (size_t i = 0; i < image_size; i++) {
        if (i % 3 == 0) {
            dst[i] = 255 - src[i];
        } else if (i % 3 == 1) {
            dst[i] = src[i] / 2;
        } else {
            dst[i] = src[i];
        }
    }
}

/** @brief Device kernel to manipulate the R/G/B values in an image based on their channel.
 *
 * @param[out] dst Destination to write the manipulated values to
 * @param src Source to read the original input from
 * @param image_size Size of the \p src array
 */
__global__ void channel_branch_device(unsigned char* dst, const unsigned char* src,
                                      const size_t image_size) {
    const unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < image_size) {
        if (thread_idx % 3 == 0) {
            dst[thread_idx] = 255 - src[thread_idx];
        } else if (thread_idx % 3 == 1) {
            dst[thread_idx] = src[thread_idx] / 2;
        } else {
            dst[thread_idx] = src[thread_idx];
        }
    }
}

/** @brief Main function for the module 3 assignment. Handles command-line input, reads input
 * image, and runs both part 1 and part 2 functions/kernels on the input image. Displays
 * timing data for each part.
 *
 * @param argc Number of arguments passed to this program
 * @param argv Array of arguments passed to this program
 * @return Returns 0 on success.
 */
int main(int argc, char* argv[]) {
    //-------------------------------------------------------------------------------------------//
    // Handle command-line input and print header                                                //
    //-------------------------------------------------------------------------------------------//
    int total_threads = IMAGE_SIZE;
    int block_size = 192;

    if (argc >= 2) {
        total_threads = atoi(argv[1]);
    }
    if (argc >= 3) {
        block_size = atoi(argv[2]);
    }

    bool warning = false;
    int num_blocks = total_threads / block_size;
    if (total_threads % block_size != 0) {
        num_blocks++;
        total_threads = num_blocks * block_size;
        warning = true;
    }

    print_header(total_threads, block_size, num_blocks, warning);

    //-------------------------------------------------------------------------------------------//
    // Read input image                                                                          //
    //-------------------------------------------------------------------------------------------//
    std::ifstream in(INPUT_FILE, std::ios::binary);
    std::string magic;
    int width, height, maxval;

    // Read header
    in >> magic;
    char c;
    in >> width >> height >> maxval;
    in.get(c);

    // Read pixel data
    const size_t image_size = width * height * 3;
    std::vector<unsigned char> pixels(image_size);
    in.read(reinterpret_cast<char*>(pixels.data()), image_size);

    //-------------------------------------------------------------------------------------------//
    // PART 1 - No branching                                                                     //
    //-------------------------------------------------------------------------------------------//

    /** HOST **/
    std::vector<unsigned char> output_pixels_host(image_size);
    auto start_host = std::chrono::high_resolution_clock::now();
    invert_pixels_host(output_pixels_host.data(), pixels.data(), image_size);
    auto stop_host = std::chrono::high_resolution_clock::now();

    // Output image to file
    std::ofstream out_host(OUTPUT_FILE_HOST, std::ios::binary);
    out_host << "P6\n" << width << " " << height << "\n" << maxval << "\n";
    out_host.write(reinterpret_cast<const char*>(output_pixels_host.data()), image_size);

    /** DEVICE **/
    std::vector<unsigned char> output_pixels_device(image_size);

    // Allocate and copy memory to device
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, image_size);
    cudaMalloc(&d_dst, image_size);
    cudaMemcpy(d_src, pixels.data(), image_size, cudaMemcpyHostToDevice);

    // Set up timing and run kernel
    cudaEvent_t start_device, stop_device;
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);
    cudaEventRecord(start_device);
    invert_pixels_device<<<num_blocks, block_size>>>(d_dst, d_src, image_size);

    // Compute elapsed time
    cudaEventRecord(stop_device);
    cudaEventSynchronize(stop_device);
    float device_ms;
    cudaEventElapsedTime(&device_ms, start_device, stop_device);
    cudaEventDestroy(start_device);
    cudaEventDestroy(stop_device);

    // Copy output back to host and clean up
    cudaMemcpy(output_pixels_device.data(), d_dst, image_size, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);

    // Output image to file
    std::ofstream out_device(OUTPUT_FILE_DEVICE, std::ios::binary);
    out_device << "P6\n" << width << " " << height << "\n" << maxval << "\n";
    out_device.write(reinterpret_cast<const char*>(output_pixels_device.data()), image_size);

    /** SUMMARY **/

    // Print report for part 1
    std::cout << "| PART 1                                                                      |"
              << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl;

    print_timings(
        std::chrono::duration_cast<std::chrono::microseconds>(stop_host - start_host).count(),
        static_cast<int>(roundf(device_ms * 1000.0f)));

    std::cout << "| " << std::setw(77) << " |" << std::endl
              << "| Host image output to " << OUTPUT_FILE_HOST
              << std::setw(80 - 24 - std::string(OUTPUT_FILE_HOST).length()) << " |" << std::endl
              << "| Device image output to " << OUTPUT_FILE_DEVICE
              << std::setw(80 - 26 - std::string(OUTPUT_FILE_DEVICE).length()) << " |" << std::endl;

    if (output_pixels_host != output_pixels_device) {
        std::cout
            << "| " << std::setw(77) << " |" << std::endl
            << "| Warning! Difference found between host and device output!                   |"
            << std::endl;
    } else {
        std::cout
            << "| " << std::setw(77) << " |" << std::endl
            << "| Output images are identical!                                                |"
            << std::endl;
    }

    //-------------------------------------------------------------------------------------------//
    // PART 2 - Branch on red/green/blue channel                                                 //
    //-------------------------------------------------------------------------------------------//

    /** HOST **/
    start_host = std::chrono::high_resolution_clock::now();
    channel_branch_host(output_pixels_host.data(), pixels.data(), image_size);
    stop_host = std::chrono::high_resolution_clock::now();

    // Output image to file
    std::ofstream out_host2(OUTPUT_FILE_HOST_2, std::ios::binary);
    out_host2 << "P6\n" << width << " " << height << "\n" << maxval << "\n";
    out_host2.write(reinterpret_cast<const char*>(output_pixels_host.data()), image_size);

    /** DEVICE **/

    // Allocate and copy memory to device
    cudaMalloc(&d_src, image_size);
    cudaMalloc(&d_dst, image_size);
    cudaMemcpy(d_src, pixels.data(), image_size, cudaMemcpyHostToDevice);

    // Set up timing and run kernel
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);
    cudaEventRecord(start_device);
    channel_branch_device<<<num_blocks, block_size>>>(d_dst, d_src, image_size);

    // Compute elapsed time
    cudaEventRecord(stop_device);
    cudaEventSynchronize(stop_device);
    cudaEventElapsedTime(&device_ms, start_device, stop_device);
    cudaEventDestroy(start_device);
    cudaEventDestroy(stop_device);

    // Copy output back to host and clean up
    cudaMemcpy(output_pixels_device.data(), d_dst, image_size, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);

    // Output image to file
    std::ofstream out_device2(OUTPUT_FILE_DEVICE_2, std::ios::binary);
    out_device2 << "P6\n" << width << " " << height << "\n" << maxval << "\n";
    out_device2.write(reinterpret_cast<const char*>(output_pixels_device.data()), image_size);

    /** SUMMARY **/

    // Print report for part 2
    std::cout << "+-----------------------------------------------------------------------------+"
              << std::endl
              << "| PART 2                                                                      |"
              << std::endl
              << "+-----------------------------------------------------------------------------+"
              << std::endl;

    print_timings(
        std::chrono::duration_cast<std::chrono::microseconds>(stop_host - start_host).count(),
        static_cast<int>(roundf(device_ms * 1000.0f)));

    std::cout << "| " << std::setw(77) << " |" << std::endl
              << "| Host image output to " << OUTPUT_FILE_HOST_2
              << std::setw(80 - 24 - std::string(OUTPUT_FILE_HOST_2).length()) << " |" << std::endl
              << "| Device image output to " << OUTPUT_FILE_DEVICE_2
              << std::setw(80 - 26 - std::string(OUTPUT_FILE_DEVICE_2).length()) << " |"
              << std::endl;

    if (output_pixels_host != output_pixels_device) {
        std::cout
            << "| " << std::setw(77) << " |" << std::endl
            << "| Warning! Difference found between host and device output!                   |"
            << std::endl;
    } else {
        std::cout
            << "| " << std::setw(77) << " |" << std::endl
            << "| Output images are identical!                                                |"
            << std::endl;
    }
    std::cout << "+-----------------------------------------------------------------------------+"
              << std::endl;
    return 0;
}
